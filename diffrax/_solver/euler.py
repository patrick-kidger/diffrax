from collections.abc import Callable
from typing import ClassVar
from typing_extensions import TypeAlias

from equinox.internal import ω

from .._custom_types import Args, BoolScalarLike, DenseInfo, RealScalarLike, VF, Y
from .._local_interpolation import LocalLinearInterpolation
from .._solution import RESULTS
from .._term import AbstractTerm
from .base import _PathState, AbstractItoSolver


_ErrorEstimate: TypeAlias = None
_SolverState: TypeAlias = None


class Euler(AbstractItoSolver):
    """Euler's method.

    1st order explicit Runge--Kutta method. Does not support adaptive step sizing. Uses
    1 stage. Uses 1st order local linear interpolation for dense/ts output.

    When used to solve SDEs, converges to the Itô solution.
    """

    term_structure: ClassVar = AbstractTerm
    interpolation_cls: ClassVar[
        Callable[..., LocalLinearInterpolation]
    ] = LocalLinearInterpolation

    def order(self, terms):
        return 1

    def strong_order(self, terms):
        return 0.5

    def init(
        self,
        terms: AbstractTerm,
        t0: RealScalarLike,
        t1: RealScalarLike,
        y0: Y,
        args: Args,
    ) -> _SolverState:
        return None

    def step(
        self,
        terms: AbstractTerm,
        t0: RealScalarLike,
        t1: RealScalarLike,
        y0: Y,
        args: Args,
        solver_state: _SolverState,
        made_jump: BoolScalarLike,
        path_state: _PathState,
    ) -> tuple[Y, _ErrorEstimate, DenseInfo, _SolverState, _PathState, RESULTS]:
        del solver_state, made_jump
        control, path_state = terms.contr(t0, t1, path_state)
        y1 = (y0**ω + terms.vf_prod(t0, y0, args, control) ** ω).ω
        dense_info = dict(y0=y0, y1=y1)
        return y1, None, dense_info, None, path_state, RESULTS.successful

    def func(
        self,
        terms: AbstractTerm,
        t0: RealScalarLike,
        y0: Y,
        args: Args,
    ) -> VF:
        return terms.vf(t0, y0, args)
