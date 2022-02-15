import abc
from typing import Callable, Optional, Tuple, TypeVar

import equinox as eqx
import jax
import jax.flatten_util as fu
import jax.numpy as jnp
import jax.scipy as jsp

from ..custom_types import Int, PyTree, Scalar
from ..misc import fixed_custom_jvp
from ..solution import RESULTS


LU_Jacobian = TypeVar("LU_Jacobian")


class NonlinearSolution(eqx.Module):
    root: PyTree
    num_steps: Int
    result: RESULTS


class AbstractNonlinearSolver(eqx.Module):
    """Abstract base class for all nonlinear root-finding algorithms.

    Subclasses will be differentiable via the implicit function theorem.
    """

    rtol: Optional[Scalar] = None
    atol: Optional[Scalar] = None

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        # Note that this breaks the descriptor protocol so we have to pass self
        # manually in __call__.
        cls._solve = fixed_custom_jvp(cls._solve, nondiff_argnums=(0, 1, 2, 3, 4))
        cls._solve.defjvp(_root_solve_jvp)

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

        # TODO: switch from is_inexact_array to is_perturbed once JAX issue #9567 is
        # fixed.
        diff_args, nondiff_args = eqx.partition(args, eqx.is_inexact_array)
        return self._solve(self, fn, x, jac, nondiff_args, diff_args)

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


# TODO: I think the jacfwd and the jvp can probably be combined, as they both
# basically do the same thing. That might improve efficiency via parallelism.
# TODO: support differentiating wrt `fn`? This isn't terribly hard -- just pass it as
# part of `diff_args` and use a custom "apply" instead of `fn`. However I can see that
# stating "differentiating wrt `fn` is allowed" might result in confusion if an attempt
# is made to differentiate wrt anything `fn` closes over. (Which is the behaviour of
# `lax.custom_root`. Such closure-differentiation is "magical" behaviour that I won't
# ever put into code I write; if differentiating wrt "closed over values" is expected
# then it's much safer to require that `fn` be a PyTree a la Equinox, but at time of
# writing that isn't yet culturally widespread enough.)
def _root_solve_jvp(
    self: AbstractNonlinearSolver,
    fn: callable,
    x: PyTree,
    jac: Optional[LU_Jacobian],
    nondiff_args: PyTree,
    diff_args: PyTree,
    tang_diff_args: PyTree,
):
    """JVP for differentiably solving for the root of a function, via the implicit
    function theorem.

    Gradients are computed with respect to diff_args.

    This is a lot like lax.custom_root -- we just use less magic. Rather than creating
    gradients for whatever the function happened to close over, we create gradients for
    just diff_args.
    """

    (diff_args,) = diff_args
    (tang_diff_args,) = tang_diff_args
    solution = self._solve(self, fn, x, jac, nondiff_args, diff_args)
    root = solution.root

    flat_root, unflatten_root = fu.ravel_pytree(root)
    args = eqx.combine(nondiff_args, diff_args)

    def _for_jac(_root):
        _root = unflatten_root(_root)
        _out = fn(_root, args)
        _out, _ = fu.ravel_pytree(_out)
        return _out

    jac_flat_root = jax.jacfwd(_for_jac)(flat_root)

    flat_diff_args, unflatten_diff_args = fu.ravel_pytree(diff_args)
    flat_tang_diff_args, _ = fu.ravel_pytree(tang_diff_args)

    def _for_jvp(_diff_args):
        _diff_args = unflatten_diff_args(_diff_args)
        _args = eqx.combine(nondiff_args, _diff_args)
        _out = fn(root, _args)
        _out, _ = fu.ravel_pytree(_out)
        return _out

    _, jvp_flat_diff_args = jax.jvp(_for_jvp, (flat_diff_args,), (flat_tang_diff_args,))

    tang_root = -jnp.linalg.solve(jac_flat_root, jvp_flat_diff_args)
    tang_root = unflatten_root(tang_root)
    return solution, NonlinearSolution(root=tang_root, num_steps=0, result=0)
