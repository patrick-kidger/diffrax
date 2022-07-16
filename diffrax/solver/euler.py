from typing import Tuple

import jax

from ..custom_types import Bool, DenseInfo, PyTree, Scalar
from ..local_interpolation import LocalLinearInterpolation
from ..misc import ω
from ..solution import RESULTS
from ..term import AbstractTerm
from .base import AbstractItoSolver


_ErrorEstimate = None
_SolverState = None


class Euler(AbstractItoSolver):
    """Euler's method.

    1st order explicit Runge--Kutta method. Does not support adaptive step sizing.

    When used to solve SDEs, converges to the Itô solution.
    """

    term_structure = jax.tree_structure(0)
    interpolation_cls = LocalLinearInterpolation

    def order(self, terms):
        return 1

    def strong_order(self, terms):
        return 0.5

    def step(
        self,
        terms: AbstractTerm,
        t0: Scalar,
        t1: Scalar,
        y0: PyTree,
        args: PyTree,
        solver_state: _SolverState,
        made_jump: Bool,
    ) -> Tuple[PyTree, _ErrorEstimate, DenseInfo, _SolverState, RESULTS]:
        del solver_state, made_jump
        control = terms.contr(t0, t1)
        y1 = (y0**ω + terms.vf_prod(t0, y0, args, control) ** ω).ω
        dense_info = dict(y0=y0, y1=y1)
        return y1, None, dense_info, None, RESULTS.successful

    def func(
        self,
        terms: AbstractTerm,
        t0: Scalar,
        y0: PyTree,
        args: PyTree,
    ) -> PyTree:
        return terms.vf(t0, y0, args)
