from typing import Any, cast, Generic, Optional, Union

import equinox as eqx
import jax
import jax.tree_util as jtu
from equinox.internal import ω
from jaxtyping import PyTree

from ._adjoint import AbstractAdjoint, ImplicitAdjoint
from ._custom_types import Args, Aux, Fn, MaybeAuxFn, SolverState, Y
from ._iterate import AbstractIterativeSolver, iterative_solve
from ._least_squares import AbstractLeastSquaresSolver
from ._minimise import AbstractMinimiser
from ._misc import inexact_asarray, NoneAux, OutAsArray
from ._root_find import AbstractRootFinder, root_find
from ._solution import Solution


class AbstractFixedPointSolver(
    AbstractIterativeSolver[Y, Y, Aux, SolverState], strict=True
):
    """Abstract base class for all fixed point solvers."""


def _rewrite_fn(fixed_point, _, inputs):
    fixed_point_fn, _, _, args, *_ = inputs
    del inputs
    f_val, _ = fixed_point_fn(fixed_point, args)
    return (f_val**ω - fixed_point**ω).ω


# Keep `optx.implicit_jvp` is happy.
if _rewrite_fn.__globals__["__name__"].startswith("jaxtyping"):
    _rewrite_fn = _rewrite_fn.__wrapped__  # pyright: ignore[reportFunctionMemberAccess]


class _ToRootFn(eqx.Module, Generic[Y, Aux]):
    fixed_point_fn: Fn[Y, Y, Aux]

    def __call__(self, y: Y, args: Args) -> tuple[Y, Aux]:
        out, aux = self.fixed_point_fn(y, args)
        if (
            eqx.tree_equal(jax.eval_shape(lambda: y), jax.eval_shape(lambda: out))
            is not True
        ):
            raise ValueError(
                "The input and output of `fixed_point_fn` must have the same structure"
            )
        return (out**ω - y**ω).ω, aux


@eqx.filter_jit
def fixed_point(
    fn: MaybeAuxFn[Y, Y, Aux],
    solver: Union[
        AbstractFixedPointSolver,
        AbstractRootFinder,
        AbstractLeastSquaresSolver,
        AbstractMinimiser,
    ],
    y0: Y,
    args: PyTree[Any] = None,
    options: Optional[dict[str, Any]] = None,
    *,
    has_aux: bool = False,
    max_steps: Optional[int] = 256,
    adjoint: AbstractAdjoint = ImplicitAdjoint(),
    throw: bool = True,
    tags: frozenset[object] = frozenset(),
) -> Solution[Y, Aux]:
    """Find a fixed-point of a function.

    Given a nonlinear function `fn(y, args)` which returns a pytree of arrays of the
    same shape as `y`, this returns the value `z` such that `fn(z, args) = z`.

    **Arguments:**

    - `fn`: The function to find the fixed-point of. This should take two arguments
        `fn(y, args)`, and return a pytree of arrays of the same shape as the input `y`.
    - `solver`: The root-finder to use. This can be either an
        [`optimistix.AbstractFixedPointSolver`][] or
        [`optimistix.AbstractRootFinder`][], or
        [`optimistix.AbstractLeastSquaresSolver`][], or
        [`optimistix.AbstractMinimiser`][]. If `solver` is a root-finder then it will
        will attempt to find the root of `fn(y, args) - y`. If `solver` is a
        least-squares or minimisation algorithm, then it will attempt to minimise
        `sum((fn(y, args) - y)^2)`.
    - `y0`: An initial guess for what `y` may be. Used to start the iterative process of
        finding the fixed point; using good initial guesses is often important.
    - `args`: Passed as the `args` of `fn(y, args)`.
    - `options`: Individual solvers may accept additional runtime arguments.
        See each individual solver's documentation for more details.
    - `has_aux`: If `True`, then `fn` may return a pair, where the first element is its
        function value, and the second is just auxiliary data. Keyword only argument.
    - `max_steps`: The maximum number of steps the solver can take. Keyword only
        argument.
    - `adjoint`: The adjoint method used to compute gradients through the fixed-point
        solve. Keyword only argument.
    - `throw`: How to report any failures. (E.g. an iterative solver running out of
        steps, or encountering divergent iterates.) If `True` then a failure will raise
        an error. If `False` then the returned solution object will have a `result`
        field indicating whether any failures occured. (See [`optimistix.Solution`][].)
        Keyword only argument.
    - `tags`: Lineax [tags](https://docs.kidger.site/lineax/api/tags/) describing the
        any structure of the Jacobian of `y -> fn(y, args) - y` with respect to y. (That
        is, the structure of the matrix `dfn/dy - I`.) Used with
        [`optimistix.ImplicitAdjoint`][] to implement the implicit function theorem as
        efficiently as possible. Keyword only argument.

    **Returns:**

    An [`optimistix.Solution`][] object.
    """

    if not has_aux:
        fn = NoneAux(fn)  # pyright: ignore
    fn = OutAsArray(fn)

    if isinstance(
        solver, (AbstractRootFinder, AbstractLeastSquaresSolver, AbstractMinimiser)
    ):
        del tags
        return root_find(
            _ToRootFn(fn),
            solver,
            y0,
            args,
            options,
            has_aux=True,
            max_steps=max_steps,
            adjoint=adjoint,
            throw=throw,
        )
    else:
        y0 = jtu.tree_map(inexact_asarray, y0)
        fn = eqx.filter_closure_convert(fn, y0, args)  # pyright: ignore
        fn = cast(Fn[Y, Y, Aux], fn)
        f_struct, aux_struct = fn.out_struct
        if eqx.tree_equal(jax.eval_shape(lambda: y0), f_struct) is not True:
            raise ValueError(
                "The input and output of `fixed_point_fn` must have the same structure"
            )
        if options is None:
            options = {}
        return iterative_solve(
            fn,
            solver,
            y0,
            args,
            options,
            max_steps=max_steps,
            adjoint=adjoint,
            throw=throw,
            tags=tags,
            f_struct=f_struct,
            aux_struct=aux_struct,
            rewrite_fn=_rewrite_fn,
        )
