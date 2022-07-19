import abc
from typing import Callable, Optional, Tuple, TypeVar

import equinox as eqx
import jax
import jax.flatten_util as fu
import jax.lax as lax
import jax.numpy as jnp
import jax.scipy as jsp

from ..custom_types import Int, PyTree, Scalar
from ..misc import implicit_jvp
from ..solution import RESULTS


LU_Jacobian = TypeVar("LU_Jacobian")


class NonlinearSolution(eqx.Module):
    root: PyTree
    num_steps: Int
    result: RESULTS


def _primal(diff_args, closure):
    self, fn, x, jac, nondiff_args = closure
    nsol = self._solve(fn, x, jac, nondiff_args, diff_args)
    return nsol.root, eqx.tree_at(lambda s: s.root, nsol, None)


def _rewrite(root, _, diff_args, closure):
    _, fn, _, _, nondiff_args = closure
    args = eqx.combine(diff_args, nondiff_args)
    return fn(root, args)


class AbstractNonlinearSolver(eqx.Module):
    """Abstract base class for all nonlinear root-finding algorithms.

    Subclasses will be differentiable via the implicit function theorem.
    """

    rtol: Optional[Scalar] = None
    atol: Optional[Scalar] = None

    @abc.abstractmethod
    def _solve(
        self,
        fn: Callable,
        x: PyTree,
        jac: Optional[LU_Jacobian],
        nondiff_args: PyTree,
        diff_args: PyTree,
    ) -> Tuple[PyTree, RESULTS]:
        pass

    def __call__(
        self, fn: Callable, x: PyTree, args: PyTree, jac: Optional[LU_Jacobian] = None
    ) -> NonlinearSolution:
        """Find `z` such that `fn(z, args) = 0`.

        Gradients will be computed with respect to `args`. (And in particular not with
        respect to either `fn` or `x` -- the latter has zero derivative by definition
        anyway.)

        **Arguments:**

        - `fn`: A function `PyTree -> PyTree` to find the root of.
            (With input and output PyTrees of the same structure.)
        - `x`: An initial guess for the location of the root.
        - `args`: Arbitrary PyTree parameterising `fn`.
        - `jac`: As returned by `self.jac(...)`. Many root finding algorithms use the
            Jacobian `d(fn)/dx` as part of their iteration. Often they will
            recompute a Jacobian at every step (for example this is done in the
            "standard" Newton solver). In practice computing the Jacobian may be
            expensive, and it may be enough to use a single value for the Jacobian
            held constant throughout the iteration. (This is a quasi-Newton method
            known as the chord method.) For the former behaviour, do not pass
            `jac`. To get the latter behaviour, do pass `jac`.

        **Returns:**

        A `NonlinearSolution` object, with attributes `root`, `num_steps`, `result`.
        `root` (hopefully) solves `fn(root, args) = 0`. `num_steps` is the number of
        steps taken in the nonlinear solver. `result` is a status code indicating
        whether the solver managed to converge or not.
        """

        x = lax.stop_gradient(x)
        diff_args, nondiff_args = eqx.partition(args, eqx.is_inexact_array)
        closure = (self, fn, x, jac, nondiff_args)
        root, nsol_no_root = implicit_jvp(_primal, _rewrite, diff_args, closure)
        return eqx.tree_at(
            lambda s: s.root, nsol_no_root, root, is_leaf=lambda z: z is None
        )

    @staticmethod
    def jac(fn: Callable, x: PyTree, args: PyTree) -> LU_Jacobian:
        """Computes the LU decomposition of the Jacobian `d(fn)/dx`.

        Arguments as [`diffrax.AbstractNonlinearSolver.__call__`][].
        """

        flat, unflatten = fu.ravel_pytree(x)
        curried = lambda z: fu.ravel_pytree(fn(unflatten(z), args))[0]
        if not jnp.issubdtype(flat, jnp.inexact):
            # Handle integer arguments
            flat = flat.astype(jnp.float32)
        return jsp.linalg.lu_factor(jax.jacfwd(curried)(flat))
