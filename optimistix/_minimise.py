from typing import Any, cast, Optional

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
from jaxtyping import PyTree, Scalar

from ._adjoint import AbstractAdjoint, ImplicitAdjoint
from ._custom_types import Aux, Fn, MaybeAuxFn, SolverState, Y
from ._iterate import AbstractIterativeSolver, iterative_solve
from ._misc import inexact_asarray, NoneAux, OutAsArray
from ._solution import Solution


class AbstractMinimiser(
    AbstractIterativeSolver[Y, Scalar, Aux, SolverState], strict=True
):
    """Abstract base class for all minimisers."""


def _rewrite_fn(minimum, _, inputs):
    minimise_fn, _, _, args, *_ = inputs
    del inputs

    def min_no_aux(x):
        f_val, _ = minimise_fn(x, args)
        return f_val

    return jax.grad(min_no_aux)(minimum)


# Keep `optx.implicit_jvp` is happy.
if _rewrite_fn.__globals__["__name__"].startswith("jaxtyping"):
    _rewrite_fn = _rewrite_fn.__wrapped__  # pyright: ignore[reportFunctionMemberAccess]


@eqx.filter_jit
def minimise(
    fn: MaybeAuxFn[Y, Scalar, Aux],
    # no type parameters, see https://github.com/microsoft/pyright/discussions/5599
    solver: AbstractMinimiser,
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
    """Minimise a function.

    This minimises a nonlinear function `fn(y, args)` which returns a scalar value.

    **Arguments:**

    - `fn`: The objective function. This should take two arguments: `fn(y, args)` and
        return a scalar.
    - `solver`: The minimiser solver to use. This should be an
        [`optimistix.AbstractMinimiser`][].
    - `y0`: An initial guess for what `y` may be.
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
        any structure of the Hessian of `fn` with respect to `y`. Used with
        [`optimistix.ImplicitAdjoint`][] to implement the implicit function theorem as
        efficiently as possible. Keyword only argument.

    **Returns:**

    An [`optimistix.Solution`][] object.
    """

    y0 = jtu.tree_map(inexact_asarray, y0)
    if not has_aux:
        fn = NoneAux(fn)  # pyright: ignore
    fn = OutAsArray(fn)
    fn = eqx.filter_closure_convert(fn, y0, args)  # pyright: ignore
    fn = cast(Fn[Y, Scalar, Aux], fn)
    f_struct, aux_struct = fn.out_struct
    if options is None:
        options = {}

    if not (
        isinstance(f_struct, jax.ShapeDtypeStruct)
        and f_struct.shape == ()
        and jnp.issubdtype(f_struct.dtype, jnp.floating)
    ):
        raise ValueError(
            "minimisation function must output a single floating-point scalar."
        )

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
        aux_struct=aux_struct,
        f_struct=f_struct,
        rewrite_fn=_rewrite_fn,
    )
