from collections.abc import Callable
from typing import ClassVar
from typing_extensions import TypeAlias

from equinox.internal import ω

from .._custom_types import Args, BoolScalarLike, DenseInfo, RealScalarLike, VF, Y
from .._local_interpolation import LocalLinearInterpolation
from .._solution import RESULTS
from .._term import AbstractTerm, MultiTerm, ODETerm
from .base import AbstractStratonovichSolver


_ErrorEstimate: TypeAlias = None
_SolverState: TypeAlias = None


class EulerHeun(AbstractStratonovichSolver):
    """Euler-Heun method.

    Uses a 1st order local linear interpolation scheme for dense/ts output.

    This should be called with `terms=MultiTerm(drift_term, diffusion_term)`, where the
    drift is an `ODETerm`.

    Used to solve SDEs, and converges to the Stratonovich solution.
    """

    term_structure: ClassVar = MultiTerm[tuple[ODETerm, AbstractTerm]]
    interpolation_cls: ClassVar[
        Callable[..., LocalLinearInterpolation]
    ] = LocalLinearInterpolation

    def order(self, terms):
        return 1

    def strong_order(self, terms):
        return 0.5

    def init(
        self,
        terms: MultiTerm[tuple[ODETerm, AbstractTerm]],
        t0: RealScalarLike,
        t1: RealScalarLike,
        y0: Y,
        args: Args,
    ) -> _SolverState:
        return None

    def step(
        self,
        terms: MultiTerm[tuple[ODETerm, AbstractTerm]],
        t0: RealScalarLike,
        t1: RealScalarLike,
        y0: Y,
        args: Args,
        solver_state: _SolverState,
        made_jump: BoolScalarLike,
    ) -> tuple[Y, _ErrorEstimate, DenseInfo, _SolverState, RESULTS]:
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
        terms: MultiTerm[tuple[AbstractTerm, AbstractTerm]],
        t0: RealScalarLike,
        y0: Y,
        args: Args,
    ) -> VF:
        drift, diffusion = terms.terms
        return drift.vf(t0, y0, args), diffusion.vf(t0, y0, args)
