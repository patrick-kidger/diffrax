from typing import Tuple

import jax

from ..custom_types import Bool, DenseInfo, PyTree, Scalar
from ..local_interpolation import LocalLinearInterpolation
from ..misc import ω
from ..solution import RESULTS
from ..term import AbstractTerm
from .base import AbstractSolver


_ErrorEstimate = None
_SolverState = None


class SemiImplicitEuler(AbstractSolver):
    """Semi-implicit Euler's method.

    Symplectic method. Does not support adaptive step sizing.
    """

    term_structure = jax.tree_structure((0, 0))
    interpolation_cls = LocalLinearInterpolation

    def order(self, terms):
        return 1

    def step(
        self,
        terms: Tuple[AbstractTerm, AbstractTerm],
        t0: Scalar,
        t1: Scalar,
        y0: Tuple[PyTree, PyTree],
        args: PyTree,
        solver_state: _SolverState,
        made_jump: Bool,
    ) -> Tuple[Tuple[PyTree, PyTree], _ErrorEstimate, DenseInfo, _SolverState, RESULTS]:
        del made_jump

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
        terms: Tuple[AbstractTerm, AbstractTerm],
        t0: Scalar,
        y0: Tuple[PyTree, PyTree],
        args: PyTree,
    ) -> Tuple[PyTree, PyTree]:

        term_1, term_2 = terms
        y0_1, y0_2 = y0
        f1 = term_1.func(t0, y0_2, args)
        f2 = term_2.func(t0, y0_1, args)
        return (f1, f2)
