from typing import Any, cast, Optional, Union

import equinox as eqx
import jax.numpy as jnp
import jax.tree_util as jtu
from equinox import AbstractVar
from jaxtyping import PyTree

from ._adjoint import AbstractAdjoint, ImplicitAdjoint
from ._custom_types import Aux, Fn, MaybeAuxFn, Out, SolverState, Y
from ._iterate import AbstractIterativeSolver, iterative_solve
from ._least_squares import AbstractLeastSquaresSolver, least_squares
from ._minimise import AbstractMinimiser, minimise
from ._misc import inexact_asarray, NoneAux, OutAsArray, tree_full_like
from ._solution import Solution


class AbstractRootFinder(
    AbstractIterativeSolver[Y, Out, Aux, SolverState], strict=True
):
    """Abstract base class for all root finders."""


def _rewrite_fn(root, _, inputs):
    root_fn, _, _, args, *_ = inputs
    del inputs
    f_val, _ = root_fn(root, args)
    return f_val


# Keep `optx.implicit_jvp` is happy.
if _rewrite_fn.__globals__["__name__"].startswith("jaxtyping"):
    _rewrite_fn = _rewrite_fn.__wrapped__  # pyright: ignore[reportFunctionMemberAccess]


def _to_minimise_fn(root_fn, norm, y, args):
    root, aux = root_fn(y, args)
    return norm(root), (root, aux)


def _to_lstsq_fn(root_fn, y, args):
    root, aux = root_fn(y, args)
    return root, (root, aux)


# Adds an additional termination condition, that `fn(y, args)` be near zero.
class _ToRoot(AbstractIterativeSolver):
    solver: AbstractVar[AbstractIterativeSolver]

    @property  # pyright: ignore
    def rtol(self):
        return self.solver.rtol

    @property  # pyright: ignore
    def atol(self):
        return self.solver.atol

    @property  # pyright: ignore
    def norm(self):
        return self.solver.norm

    def init(self, fn, y, args, options, f_struct, aux_struct, tags):
        orig_f_struct, _ = aux_struct
        init_state = self.solver.init(fn, y, args, options, f_struct, aux_struct, tags)
        f_inf = tree_full_like(orig_f_struct, jnp.inf)
        return (init_state, f_inf)

    def step(self, fn, y, args, options, state, tags):
        state, _ = state
        new_y, new_state, (f, aux) = self.solver.step(fn, y, args, options, state, tags)
        return new_y, (new_state, f), (f, aux)

    def terminate(self, fn, y, args, options, state, tags):
        state, f = state
        terminate, result = self.solver.terminate(fn, y, args, options, state, tags)
        # No rtol, because `rtol * 0 = 0`.
        near_zero = self.norm(f) < self.atol
        return terminate & near_zero, result

    def postprocess(self, fn, y, aux, args, options, state, tags, result):
        state, _ = state
        return self.solver.postprocess(fn, y, aux, args, options, state, tags, result)


class _MinimToRoot(AbstractMinimiser, _ToRoot):
    solver: AbstractMinimiser

    # Redeclare these three to work around the Equinox bug fixed here:
    # https://github.com/patrick-kidger/equinox/pull/544
    @property  # pyright: ignore
    def rtol(self):
        return self.solver.rtol

    @property  # pyright: ignore
    def atol(self):
        return self.solver.atol

    @property  # pyright: ignore
    def norm(self):
        return self.solver.norm


class _LstsqToRoot(AbstractLeastSquaresSolver, _ToRoot):
    solver: AbstractLeastSquaresSolver

    # Redeclare these three to work around the Equinox bug fixed here:
    # https://github.com/patrick-kidger/equinox/pull/544
    @property  # pyright: ignore
    def rtol(self):
        return self.solver.rtol

    @property  # pyright: ignore
    def atol(self):
        return self.solver.atol

    @property  # pyright: ignore
    def norm(self):
        return self.solver.norm


@eqx.filter_jit
def root_find(
    fn: MaybeAuxFn[Y, Out, Aux],
    # no type parameters, see https://github.com/microsoft/pyright/discussions/5599
    solver: Union[AbstractRootFinder, AbstractLeastSquaresSolver, AbstractMinimiser],
    y0: Y,
    args: PyTree = None,
    options: Optional[dict[str, Any]] = None,
    *,
    has_aux: bool = False,
    max_steps: Optional[int] = 256,
    adjoint: AbstractAdjoint = ImplicitAdjoint(),
    throw: bool = True,
    tags: frozenset[object] = frozenset(),
) -> Solution[Y, Aux]:
    """Solve a root-finding problem.

    Given a nonlinear function `fn(y, args)` which returns a pytree of arrays,
    this returns the value `z` such that `fn(z, args) = 0`.

    **Arguments:**

    - `fn`: The function to find the roots of. This should take two arguments:
        `fn(y, args)` and return a pytree of arrays not necessarily of the same shape
        as the input `y`.
    - `solver`: The root-finder to use. This should be an
        [`optimistix.AbstractRootFinder`][],
        [`optimistix.AbstractLeastSquaresSolver`][], or
        [`optimistix.AbstractMinimiser`][]. If it is a least-squares solver or a
        minimiser, then the value `sum(fn(y, args)^2)` is minimised.
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
        any structure of the Jacobian of `fn` with respect to `y`. Used with some
        solvers (e.g. [`optimistix.Newton`][]), and with some adjoint methods (e.g.
        [`optimistix.ImplicitAdjoint`][]) to improve the efficiency of linear solves.
        Keyword only argument.

    **Returns:**

    An [`optimistix.Solution`][] object.
    """

    if not has_aux:
        fn = NoneAux(fn)  # pyright: ignore
    fn = OutAsArray(fn)

    if isinstance(solver, AbstractMinimiser):
        del tags
        sol = minimise(
            eqx.Partial(_to_minimise_fn, fn, solver.norm),
            _MinimToRoot(solver),  # pyright: ignore
            y0,
            args,
            options,
            has_aux=True,
            max_steps=max_steps,
            adjoint=adjoint,
            throw=throw,
        )
        _, aux = sol.aux
        sol = eqx.tree_at(lambda s: s.aux, sol, aux)
        return sol
    elif isinstance(solver, AbstractLeastSquaresSolver):
        del tags
        return least_squares(
            eqx.Partial(_to_lstsq_fn, fn),
            _LstsqToRoot(solver),  # pyright: ignore
            y0,
            args,
            options,
            has_aux=True,
            max_steps=max_steps,
            adjoint=adjoint,
            throw=throw,
        )
        _, aux = sol.aux
        sol = eqx.tree_at(lambda s: s.aux, sol, aux)
        return sol
    else:
        y0 = jtu.tree_map(inexact_asarray, y0)
        fn = eqx.filter_closure_convert(fn, y0, args)  # pyright: ignore
        fn = cast(Fn[Y, Out, Aux], fn)
        f_struct, aux_struct = fn.out_struct
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
