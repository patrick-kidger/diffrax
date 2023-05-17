import equinox as eqx
import jax
import jax.flatten_util as jfu
import jax.numpy as jnp

from ..solution import RESULTS
from .base import AbstractNonlinearSolver, NonlinearSolution


class AffineNonlinearSolver(AbstractNonlinearSolver):
    """Finds the fixed point of f(x)=0, where f(x) = Ax + b is affine.

    !!! Warning

        This solver only exists temporarily. It is deliberately undocumented and will be
        removed shortly, in favour of a more comprehensive approach to performing linear
        and nonlinear solves.
    """

    def _solve(self, fn, x, jac, nondiff_args, diff_args):
        del jac
        args = eqx.combine(nondiff_args, diff_args)
        flat, unflatten = jfu.ravel_pytree(x)
        zero = jnp.zeros_like(flat)
        flat_fn = lambda z: jfu.ravel_pytree(fn(unflatten(z), args))[0]
        b = flat_fn(zero)
        A = jax.jacfwd(flat_fn)(zero)
        out = -jnp.linalg.solve(A, b)
        out = unflatten(out)
        return NonlinearSolution(root=out, num_steps=0, result=RESULTS.successful)

    @staticmethod
    def jac(fn, x, args):
        return None
