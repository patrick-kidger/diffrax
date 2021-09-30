import functools as ft

import equinox as eqx
import jax
import jax.lax as lax
import jax.numpy as jnp

from ..misc import ravel_pytree, rms_norm
from ..solution import RESULTS
from .base import AbstractNonlinearSolver


# TODO: factor out parts of this code into the base class as we get more solvers.


def _small(diffsize):
    # TODO: make a more careful choice here -- the existence of this function is
    # pretty ad-hoc.
    resolution = 10 ** (2 - jnp.finfo(diffsize.dtype).precision)
    return diffsize < resolution


def _diverged(rate):
    return ~jnp.isfinite(rate) | (rate > 2)


def _converged(factor, tol):
    return (factor > 0) & (factor < tol)


# TODO: different implementations do slightly different things. Not clear which is
# optimal, so all of the following should be considered up-for-debate.
#
# (1) The value `scale`.
#  - SciPy uses a vector-valued `scale` for each element of `x`, computed on the
#    initial value.
#  - OrdinaryDiffEq.jl uses a scalar-valued `scale`, recomputed on every
#    iteration.
#  Just to keep life interesting, we're using a scalar-valued `scale` computed on
#  the initial value.
#
# (2) The divergence criteria.
#  - SciPy uses the very strict criteria of checking whether `rate >= 1` for
#    divergence (plus an additional criterion comparing against tolerance.)
#  - OrdinaryDiffEq.jl just checks whether `rate > 2`.
#  We follow OrdinaryDiffEq.jl's more permissive approach here.
#
# (3) The minimum number of loops.
#  - SciPy uses a minimum of two loops, to be able to check the convergence criterion.
#  - OrdinaryDiffEq.jl allow just one loop if the first `diffsize < 1e-5`.
#  The 1e-5 criterion seems a bit suspect if high tolerances are demanded, so we follow
#  SciPy on this one, but switching to additional use our `_small' criteria for
#  stopping after a single loop may well be desirable from an efficiency standpoint.
#
# (4) Enhancements to the solver.
#  - There's a lot of ways to do nonlinear solves (w/ line search etc.) They're
#    probably overkill here (there's a reason no-one else uses them after all) but this
#    deserves more thought on our part than just "other standard implementations don't
#    use them"!
class NewtonNonlinearSolver(AbstractNonlinearSolver):
    # Default values taken from SciPy's Radau.
    max_steps: int = 6
    rtol: float = 1e-3
    atol: float = 1e-6
    tol: float = 1e-6
    norm: callable = rms_norm
    tolerate_nonconvergence: bool = False

    def __post_init__(self):
        if self.max_steps < 2:
            raise ValueError(f"max_steps must be at least 2. Got {self.max_steps}")

    def __call__(self, fn, x, args):
        # TODO: not sure this will give the desired behaviour if differentiating wrt
        # integers or complexes.
        diff_args, nondiff_args = eqx.partition(args, eqx.is_inexact_array)
        return _newton_solve(self, fn, x, nondiff_args, diff_args)


@ft.partial(jax.custom_jvp, nondiff_argnums=(0, 1, 2, 3))
def _newton_solve(self, fn, x, nondiff_args, diff_args):
    args = eqx.combine(nondiff_args, diff_args)
    scale = self.atol + self.rtol * self.norm(x)
    flat, unflatten = ravel_pytree(x)
    curried = lambda z: ravel_pytree(fn(unflatten(z), *args))[0]

    def cond_fn(val):
        _, step, diffsize, diffsize_prev = val
        rate = diffsize / diffsize_prev
        factor = diffsize * rate / (1 - rate)
        step_okay = step < self.max_steps
        not_small = ~_small(diffsize)
        not_diverged = ~_diverged(rate)
        not_converged = ~_converged(factor, self.tol)
        return step_okay & not_small & not_diverged & not_converged

    def body_fn(val):
        flat, step, diffsize, _ = val
        jac = jax.jacfwd(curried)(flat)
        fx = curried(flat)
        diff = jnp.linalg.solve(jac, fx)
        flat = flat - diff
        diffsize_prev = diffsize
        diffsize = self.norm(diff / scale)
        val = (flat, step + 1, diffsize, diffsize_prev)
        return val

    # Unconditionally execute two loops to fill in diffsize and diffsize_prev.
    val = (flat, 0, None, None)
    val = body_fn(val)
    val = body_fn(val)
    val = lax.while_loop(cond_fn, body_fn, val)
    flat, _, diffsize, diffsize_prev = val

    if self.tolerate_nonconvergence:
        result = RESULTS.successful
    else:
        rate = diffsize / diffsize_prev
        factor = diffsize * rate / (1 - rate)
        converged = _converged(factor, self.tol)
        diverged = _diverged(rate)
        small = _small(diffsize)
        result = jnp.where(
            converged, RESULTS.successful, RESULTS.implicit_nonconvergence
        )
        result = jnp.where(diverged, RESULTS.implicit_divergence, result)
        result = jnp.where(small, RESULTS.successful, result)
    return unflatten(flat), result


# TODO: I think the jacfwd and the jvp can probably be combined, as they both
# basically do the same thing. That might improve efficiency via parallelism.
@_newton_solve.defjvp
def _newton_solve_jvp(self, fn, x, nondiff_args, diff_args, tang_diff_args):
    (diff_args,) = diff_args
    (tang_diff_args,) = tang_diff_args
    root, result = _newton_solve(self, fn, x, nondiff_args, diff_args)

    flat_root, unflatten_root = ravel_pytree(root)
    args = eqx.combine(nondiff_args, diff_args)

    def _for_jac(_root):
        _root = unflatten_root(_root)
        _out = fn(_root, *args)
        _out, _ = ravel_pytree(_out)
        return _out

    jac_flat_root = jax.jacfwd(_for_jac)(flat_root)

    flat_diff_args, unflatten_diff_args = ravel_pytree(diff_args)
    flat_tang_diff_args, _ = ravel_pytree(tang_diff_args)

    def _for_jvp(_diff_args):
        _diff_args = unflatten_diff_args(_diff_args)
        _args = eqx.combine(nondiff_args, _diff_args)
        _out = fn(root, *_args)
        _out, _ = ravel_pytree(_out)
        return _out

    _, jvp_flat_diff_args = jax.jvp(_for_jvp, (flat_diff_args,), (flat_tang_diff_args,))

    tang_root = -jnp.linalg.solve(jac_flat_root, jvp_flat_diff_args)
    tang_root = unflatten_root(tang_root)
    return (root, result), (tang_root, 0)
