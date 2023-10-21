from typing_extensions import TypeAlias

from equinox.internal import ω
from jaxtyping import ArrayLike, PyTree

from .._custom_types import BoolScalarLike, DenseInfo, RealScalarLike
from .._local_interpolation import LocalLinearInterpolation
from .._solution import RESULTS
from .._term import AbstractTerm
from .base import AbstractSolver


_ErrorEstimate: TypeAlias = None
_SolverState: TypeAlias = None


class SemiImplicitEuler(AbstractSolver):
    """Semi-implicit Euler's method.

    Symplectic method. Does not support adaptive step sizing. Uses 1st order local
    linear interpolation for dense/ts output.
    """

    term_structure = (AbstractTerm, AbstractTerm)
    interpolation_cls = LocalLinearInterpolation

    def order(self, terms):
        return 1

    def init(
        self,
        terms: tuple[AbstractTerm, AbstractTerm],
        t0: RealScalarLike,
        t1: RealScalarLike,
        y0: PyTree[ArrayLike],
        args: PyTree,
    ) -> _SolverState:
        return None

    def step(
        self,
        terms: tuple[AbstractTerm, AbstractTerm],
        t0: RealScalarLike,
        t1: RealScalarLike,
        y0: tuple[PyTree[ArrayLike], PyTree[ArrayLike]],
        args: PyTree,
        solver_state: _SolverState,
        made_jump: BoolScalarLike,
    ) -> tuple[
        tuple[PyTree[ArrayLike], PyTree[ArrayLike]],
        _ErrorEstimate,
        DenseInfo,
        _SolverState,
        RESULTS,
    ]:
        del solver_state, made_jump

        term_1, term_2 = terms
        y0_1, y0_2 = y0

        control1 = term_1.contr(t0, t1)
        control2 = term_2.contr(t0, t1)
        y1_1 = (y0_1**ω + term_1.vf_prod(t0, y0_2, args, control1) ** ω).ω
        y1_2 = (y0_2**ω + term_2.vf_prod(t0, y1_1, args, control2) ** ω).ω

        y1 = (y1_1, y1_2)
        dense_info = dict(y0=y0, y1=y1)
        return y1, None, dense_info, None, RESULTS.successful

    def func(
        self,
        terms: tuple[AbstractTerm, AbstractTerm],
        t0: RealScalarLike,
        y0: tuple[PyTree[ArrayLike], PyTree[ArrayLike]],
        args: PyTree,
    ) -> tuple[PyTree[ArrayLike], PyTree[ArrayLike]]:

        term_1, term_2 = terms
        y0_1, y0_2 = y0
        f1 = term_1.vf(t0, y0_2, args)
        f2 = term_2.vf(t0, y0_1, args)
        return f1, f2
