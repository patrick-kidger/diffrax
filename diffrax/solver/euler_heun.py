from typing import Tuple

import jax

from ..custom_types import Bool, DenseInfo, PyTree, Scalar
from ..local_interpolation import LocalLinearInterpolation
from ..misc import ω
from ..solution import RESULTS
from ..term import AbstractTerm
from .base import AbstractStratonovichSolver


_ErrorEstimate = None
_SolverState = None


class EulerHeun(AbstractStratonovichSolver):
    """Euler-Heun method.

    Used to solve SDEs, and converges to the Stratonovich solution.
    """

    term_structure = jax.tree_structure((0, 0))
    interpolation_cls = LocalLinearInterpolation

    def order(self, terms):
        return 1

    def strong_order(self, terms):
        return 0.5

    def step(
        self,
        terms: Tuple[AbstractTerm, AbstractTerm],
        t0: Scalar,
        t1: Scalar,
        y0: PyTree,
        args: PyTree,
        solver_state: _SolverState,
        made_jump: Bool,
    ) -> Tuple[PyTree, _ErrorEstimate, DenseInfo, _SolverState, RESULTS]:
        del solver_state, made_jump

        drift, diffusion = terms
        dt = drift.contr(t0, t1)
        dW = diffusion.contr(t0, t1)

        f0 = drift.vf_prod(t0, y0, args, dt)
        g0 = diffusion.vf_prod(t0, y0, args, dW)

        y_prime = (y0**ω + g0**ω).ω
        g_prime = diffusion.vf_prod(t0, y_prime, args, dW)

        y1 = (y0**ω + f0**ω + 0.5 * (g0**ω + g_prime**ω)).ω

        dense_info = dict(y0=y0, y1=y1)
        return y1, None, dense_info, None, RESULTS.successful

    def func(
        self,
        terms: Tuple[AbstractTerm, AbstractTerm],
        t0: Scalar,
        y0: PyTree,
        args: PyTree,
    ) -> PyTree:
        drift, diffusion = terms
        return drift.vf(t0, y0, args), diffusion.vf(t0, y0, args)
