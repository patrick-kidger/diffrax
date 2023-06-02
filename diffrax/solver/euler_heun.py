from typing import Tuple

from equinox.internal import ω

from ..custom_types import Bool, DenseInfo, PyTree, Scalar
from ..local_interpolation import LocalLinearInterpolation
from ..solution import RESULTS
from ..term import AbstractTerm, MultiTerm, ODETerm
from .base import AbstractStratonovichSolver


_ErrorEstimate = None
_SolverState = None


class EulerHeun(AbstractStratonovichSolver):
    """Euler-Heun method.

    Uses a 1st order local linear interpolation scheme for dense/ts output.

    This should be called with `terms=MultiTerm(drift_term, diffusion_term)`, where the
    drift is an `ODETerm`.

    Used to solve SDEs, and converges to the Stratonovich solution.
    """

    term_structure = MultiTerm[Tuple[ODETerm, AbstractTerm]]
    interpolation_cls = LocalLinearInterpolation

    def order(self, terms):
        return 1

    def strong_order(self, terms):
        return 0.5

    def init(
        self,
        terms: MultiTerm[Tuple[ODETerm, AbstractTerm]],
        t0: Scalar,
        t1: Scalar,
        y0: PyTree,
        args: PyTree,
    ) -> _SolverState:
        return None

    def step(
        self,
        terms: MultiTerm[Tuple[ODETerm, AbstractTerm]],
        t0: Scalar,
        t1: Scalar,
        y0: PyTree,
        args: PyTree,
        solver_state: _SolverState,
        made_jump: Bool,
    ) -> Tuple[PyTree, _ErrorEstimate, DenseInfo, _SolverState, RESULTS]:
        del solver_state, made_jump

        drift, diffusion = terms.terms
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
        terms: MultiTerm[Tuple[AbstractTerm, AbstractTerm]],
        t0: Scalar,
        y0: PyTree,
        args: PyTree,
    ) -> PyTree:
        drift, diffusion = terms.terms
        return drift.vf(t0, y0, args), diffusion.vf(t0, y0, args)
