from typing import Callable, Optional, Tuple

import equinox as eqx
import jax
import jax.flatten_util as fu
import jax.lax as lax
import jax.numpy as jnp
import jax.scipy as jsp

from ..custom_types import Bool, Int, PyTree, Scalar
from ..misc import rms_norm
from ..solution import RESULTS
from .base import AbstractNonlinearSolver, LU_Jacobian, NonlinearSolution


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
# - (+The entire space of quasi-Newton methods.)
class NewtonNonlinearSolver(AbstractNonlinearSolver):
    """Newton's method for root-finding. (Also known as Newton--Raphson.)

    Also supports the quasi-Newton chord method.

    !!! info

        If using this as part of a implicit ODE solver, then:

        - An adaptive step size controller should be used (e.g.
          `diffrax.PIDController`). This will allow smaller steps to be made if the
          nonlinear solver fails to converge.
        - As a general rule, the values for `rtol` and `atol` should be set to the same
          values as used for the adaptive step size controller. (And this will happen
          automatically by default.)
        - The value for `kappa` should usually be left alone.

    !!! warning

        Note that backpropagation through `__call__` may not produce accurate values if
        `tolerate_nonconvergence=True`, as the backpropagation calculation implicitly
        assumes that the forward pass converged.
    """

    max_steps: Optional[Int] = 10
    kappa: Scalar = 1e-2
    norm: Callable = rms_norm
    tolerate_nonconvergence: bool = False

    def __post_init__(self):
        if self.max_steps is not None and self.max_steps < 2:
            raise ValueError("max_steps must be at least 2.")

    def _solve(
        self,
        fn: callable,
        x: PyTree,
        jac: Optional[LU_Jacobian],
        nondiff_args: PyTree,
        diff_args: PyTree,
    ) -> Tuple[PyTree, RESULTS]:
        args = eqx.combine(nondiff_args, diff_args)
        if self.rtol is None or self.atol is None:
            raise ValueError(
                "The `rtol` and `atol` tolerances for `NewtonNonlinearSolver` default "
                "to the `rtol` and `atol` used with an adaptive step size "
                "controller (such as `diffrax.PIDController`). Either use an "
                "adaptive step size controller, or specify these tolerances "
                "manually.\n"
                "Note that this changed in Diffrax version 0.2.0. If you want to match "
                "the previous defaults then specify `rtol=1e-3`, `atol=1e-6`. For "
                "example:\n"
                "```\n"
                "diffrax.NewtonNonlinearSolver(rtol=1e-3, atol=1e-6)\n"
                "```\n"
            )
        scale = self.atol + self.rtol * self.norm(x)
        flat, unflatten = fu.ravel_pytree(x)
        if flat.size == 0:
            return NonlinearSolution(root=x, num_steps=0, result=RESULTS.successful)
        curried = lambda z: fu.ravel_pytree(fn(unflatten(z), args))[0]

        def cond_fn(val):
            _, step, diffsize, diffsize_prev = val
            at_least_two = step < 2
            rate = diffsize / diffsize_prev
            factor = diffsize * rate / (1 - rate)
            if self.max_steps is None:
                step_okay = True
            else:
                step_okay = step < self.max_steps
            not_small = ~_small(diffsize)
            not_diverged = ~_diverged(rate)
            not_converged = ~_converged(factor, self.kappa)
            return at_least_two | (step_okay & not_small & not_diverged & not_converged)

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

        val = (flat, 0, 0.0, 0.0)
        val = lax.while_loop(cond_fn, body_fn, val)
        flat, num_steps, diffsize, diffsize_prev = val

        if self.tolerate_nonconvergence:
            result = RESULTS.successful
        else:
            rate = diffsize / diffsize_prev
            factor = diffsize * rate / (1 - rate)
            converged = _converged(factor, self.kappa)
            diverged = _diverged(rate)
            small = _small(diffsize)
            result = jnp.where(
                converged, RESULTS.successful, RESULTS.implicit_nonconvergence
            )
            result = jnp.where(diverged, RESULTS.implicit_divergence, result)
            result = jnp.where(small, RESULTS.successful, result)
        root = unflatten(flat)
        return NonlinearSolution(root=root, num_steps=num_steps, result=result)


NewtonNonlinearSolver.__init__.__doc__ = """
**Arguments:**

- `rtol`: The relative tolerance for determining convergence. Defaults to the same
    `rtol` as passed to an adaptive step controller if one is used.
- `atol`: The absolute tolerance for determining convergence. Defaults to the same
    `atol` as passed to an adaptive step controller if one is used.
- `max_steps`: The maximum number of steps allowed. If more than this are required then
    the iteration fails. Set to `None` to allow an arbitrary number of steps.
- `kappa`: The kappa value for determining convergence.
- `norm`: A function `PyTree -> Scalar`, which is called to determine the size of the
    current value. (Used in determining convergence.)
- `tolerate_nonconvergence`: Whether to return an error code if the iteration fails to
    converge (or to silently pretend it was successful).
"""
