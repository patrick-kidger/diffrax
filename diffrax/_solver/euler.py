from typing_extensions import TypeAlias

from equinox.internal import ω
from jaxtyping import ArrayLike, PyTree

from .._custom_types import BoolScalarLike, DenseInfo, RealScalarLike
from .._local_interpolation import LocalLinearInterpolation
from .._solution import RESULTS
from .._term import AbstractTerm
from .base import AbstractItoSolver


_ErrorEstimate: TypeAlias = None
_SolverState: TypeAlias = None


class Euler(AbstractItoSolver):
    """Euler's method.

    1st order explicit Runge--Kutta method. Does not support adaptive step sizing. Uses
    1 stage. Uses 1st order local linear interpolation for dense/ts output.

    When used to solve SDEs, converges to the Itô solution.
    """

    term_structure = AbstractTerm
    interpolation_cls = LocalLinearInterpolation

    def order(self, terms):
        return 1

    def strong_order(self, terms):
        return 0.5

    def init(
        self,
        terms: AbstractTerm,
        t0: RealScalarLike,
        t1: RealScalarLike,
        y0: PyTree[ArrayLike],
        args: PyTree,
    ) -> _SolverState:
        return None

    def step(
        self,
        terms: AbstractTerm,
        t0: RealScalarLike,
        t1: RealScalarLike,
        y0: PyTree[ArrayLike],
        args: PyTree,
        solver_state: _SolverState,
        made_jump: BoolScalarLike,
    ) -> tuple[PyTree, _ErrorEstimate, DenseInfo, _SolverState, RESULTS]:
        del solver_state, made_jump
        control = terms.contr(t0, t1)
        y1 = (y0**ω + terms.vf_prod(t0, y0, args, control) ** ω).ω
        dense_info = dict(y0=y0, y1=y1)
        return y1, None, dense_info, None, RESULTS.successful

    def func(
        self,
        terms: AbstractTerm,
        t0: RealScalarLike,
        y0: PyTree,
        args: PyTree,
    ) -> PyTree:
        return terms.vf(t0, y0, args)
