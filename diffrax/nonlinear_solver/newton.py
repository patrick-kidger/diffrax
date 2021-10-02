from typing import Optional, Tuple

import equinox as eqx
import jax
import jax.lax as lax
import jax.numpy as jnp
import jax.scipy as jsp

from ..custom_types import Bool, PyTree, Scalar
from ..misc import ravel_pytree, rms_norm
from ..solution import RESULTS
from .base import AbstractNonlinearSolver, LU_Jacobian


def _small(diffsize: Scalar) -> Bool:
    # TODO: make a more careful choice here -- the existence of this function is
    # pretty ad-hoc.
    resolution = 10 ** (2 - jnp.finfo(diffsize.dtype).precision)
    return diffsize < resolution


def _diverged(rate: Scalar) -> Bool:
    return ~jnp.isfinite(rate) | (rate > 2)


def _converged(factor: Scalar, tol: Scalar) -> Bool:
    return (factor > 0) & (factor < tol)


# TODO: different implementations do slightly different things. Not clear which is
# optimal, so both of the following should be considered up-for-debate.
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

# TODO: there's some of tricks for improving Newton's method, specifically when
# solving implicit ODEs. Several have already been implemented. Some remaing ones:
# - The minimum number of iterations: it's possible to use only a single iteration.
# - Choice of initial guess.
# - Transform Jacobians into W, in particular when treating FIRKs.
class NewtonNonlinearSolver(AbstractNonlinearSolver):
    """Newton's method for root-finding. (Also known as Newton--Raphson.)"""

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

    def _solve(
        self,
        fn: callable,
        x: PyTree,
        jac: Optional[LU_Jacobian],
        nondiff_args: PyTree,
        diff_args: PyTree,
    ) -> Tuple[PyTree, RESULTS]:
        args = eqx.combine(nondiff_args, diff_args)
        scale = self.atol + self.rtol * self.norm(x)
        flat, unflatten = ravel_pytree(x)
        if flat.size == 0:
            return x, RESULTS.successful
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
            fx = curried(flat)
            if jac is None:
                _jac = jax.jacfwd(curried)(flat)
                diff = jsp.linalg.solve(_jac, fx)
            else:
                diff = jsp.linalg.lu_solve(jac, fx)
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
