import abc
from typing import Any, Generic

import equinox as eqx
import jax
import jax.numpy as jnp
from equinox import AbstractVar
from equinox.internal import ω
from jaxtyping import Array, Bool, PyTree, Scalar

from .._custom_types import Aux, Fn, Out, SolverState, Y
from .._fixed_point import AbstractFixedPointSolver
from .._iterate import AbstractIterativeSolver
from .._least_squares import AbstractLeastSquaresSolver
from .._minimise import AbstractMinimiser
from .._misc import sum_squares, tree_full_like, tree_where
from .._root_find import AbstractRootFinder
from .._solution import RESULTS


class _BestSoFarState(eqx.Module, Generic[Y, Aux, SolverState], strict=True):
    best_y: Y
    best_aux: Aux
    best_loss: Scalar
    state: SolverState


def _auxmented(fn, y, args):
    out, aux = fn(y, args)
    return out, (out, aux)


class _AbstractBestSoFarSolver(
    AbstractIterativeSolver, Generic[Y, Out, Aux], strict=True
):
    solver: AbstractVar[AbstractIterativeSolver[Y, Out, tuple[Out, Aux], Any]]

    @abc.abstractmethod
    def _to_loss(self, y: Y, f: Out) -> Scalar:
        ...

    @property  # pyright: ignore
    def rtol(self):
        return self.solver.rtol

    @property  # pyright: ignore
    def atol(self):
        return self.solver.atol

    @property  # pyright: ignore
    def norm(self):
        return self.solver.norm

    def init(
        self,
        fn: Fn[Y, Out, Aux],
        y: Y,
        args: PyTree,
        options: dict[str, Any],
        f_struct: PyTree[jax.ShapeDtypeStruct],
        aux_struct: PyTree[jax.ShapeDtypeStruct],
        tags: frozenset[object],
    ) -> _BestSoFarState:
        aux = tree_full_like(aux_struct, 0)
        loss = jnp.array(jnp.inf)
        auxmented_fn = eqx.Partial(_auxmented, fn)
        state = self.solver.init(
            auxmented_fn, y, args, options, f_struct, (f_struct, aux_struct), tags
        )
        return _BestSoFarState(best_y=y, best_aux=aux, best_loss=loss, state=state)

    def step(
        self,
        fn: Fn[Y, Out, Aux],
        y: Y,
        args: PyTree,
        options: dict[str, Any],
        state: _BestSoFarState,
        tags: frozenset[object],
    ) -> tuple[Y, _BestSoFarState, Aux]:
        auxmented_fn = eqx.Partial(_auxmented, fn)
        new_y, new_state, (f, new_aux) = self.solver.step(
            auxmented_fn, y, args, options, state.state, tags
        )
        loss = self._to_loss(y, f)
        pred = loss < state.best_loss
        best_y = tree_where(pred, y, state.best_y)
        best_aux = tree_where(pred, new_aux, state.best_aux)
        best_loss = jnp.where(pred, loss, state.best_loss)
        new_state = _BestSoFarState(
            best_y=best_y, best_aux=best_aux, best_loss=best_loss, state=new_state
        )
        return new_y, new_state, new_aux

    def terminate(
        self,
        fn: Fn[Y, Out, Aux],
        y: Y,
        args: PyTree,
        options: dict[str, Any],
        state: _BestSoFarState,
        tags: frozenset[object],
    ) -> tuple[Bool[Array, ""], RESULTS]:
        auxmented_fn = eqx.Partial(_auxmented, fn)
        return self.solver.terminate(auxmented_fn, y, args, options, state.state, tags)

    def postprocess(
        self,
        fn: Fn[Y, Out, Aux],
        y: Y,
        aux: Aux,
        args: PyTree,
        options: dict[str, Any],
        state: _BestSoFarState,
        tags: frozenset[object],
        result: RESULTS,
    ) -> tuple[Y, Aux, dict[str, Any]]:
        return state.best_y, state.best_aux, {}


class BestSoFarMinimiser(  # pyright: ignore
    _AbstractBestSoFarSolver[Y, Scalar, Aux],
    AbstractMinimiser[Y, Aux, _BestSoFarState],
    strict=True,
):
    """Wraps another minimiser, to return the best-so-far value. That is, it makes a
    copy of the best `y` seen, and returns that.
    """

    solver: AbstractMinimiser[Y, tuple[Scalar, Aux], Any]

    # Explicitly declare to keep pyright happy.
    def __init__(self, solver: AbstractMinimiser[Y, tuple[Scalar, Aux], Any]):
        self.solver = solver

    def _to_loss(self, y: Y, f: Scalar) -> Scalar:
        return f

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


BestSoFarMinimiser.__init__.__doc__ = """**Arguments:**

- `solver`: the minimiser to wrap.  
"""


class BestSoFarLeastSquares(  # pyright: ignore
    _AbstractBestSoFarSolver[Y, Out, Aux],
    AbstractLeastSquaresSolver[Y, Out, Aux, _BestSoFarState],
    strict=True,
):
    """Wraps another least-squares solver, to return the best-so-far value. That is, it
    makes a copy of the best `y` seen, and returns that.
    """

    solver: AbstractLeastSquaresSolver[Y, Out, tuple[Out, Aux], Any]

    # Explicitly declare to keep pyright happy.
    def __init__(
        self, solver: AbstractLeastSquaresSolver[Y, Out, tuple[Out, Aux], Any]
    ):
        self.solver = solver

    def _to_loss(self, y: Y, f: Out) -> Scalar:
        return sum_squares(f)

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


BestSoFarLeastSquares.__init__.__doc__ = """**Arguments:**

- `solver`: the least-squares solver to wrap.  
"""


class BestSoFarRootFinder(  # pyright: ignore
    _AbstractBestSoFarSolver[Y, Out, Aux],
    AbstractRootFinder[Y, Out, Aux, _BestSoFarState],
    strict=True,
):
    """Wraps another root-finder, to return the best-so-far value. That is, it
    makes a copy of the best `y` seen, and returns that.
    """

    solver: AbstractRootFinder[Y, Out, tuple[Out, Aux], Any]

    # Explicitly declare to keep pyright happy.
    def __init__(self, solver: AbstractRootFinder[Y, Out, tuple[Out, Aux], Any]):
        self.solver = solver

    def _to_loss(self, y: Y, f: Out) -> Scalar:
        return sum_squares(f)

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


BestSoFarRootFinder.__init__.__doc__ = """**Arguments:**

- `solver`: the root-finder solver to wrap.  
"""


class BestSoFarFixedPoint(  # pyright: ignore
    _AbstractBestSoFarSolver[Y, Y, Aux],
    AbstractFixedPointSolver[Y, Aux, _BestSoFarState],
    strict=True,
):
    """Wraps another fixed-point solver, to return the best-so-far value. That is, it
    makes a copy of the best `y` seen, and returns that.
    """

    solver: AbstractFixedPointSolver[Y, tuple[Y, Aux], Any]

    # Explicitly declare to keep pyright happy.
    def __init__(self, solver: AbstractFixedPointSolver[Y, tuple[Y, Aux], Any]):
        self.solver = solver

    def _to_loss(self, y: Y, f: Y) -> Scalar:
        return sum_squares((y**ω - f**ω).ω)

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


BestSoFarFixedPoint.__init__.__doc__ = """**Arguments:**

- `solver`: the fixed-point solver to wrap.  
"""
