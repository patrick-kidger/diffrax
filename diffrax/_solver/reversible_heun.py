from collections.abc import Callable
from typing import ClassVar
from typing_extensions import TypeAlias

import jax.lax as lax
from equinox.internal import ω
from jaxtyping import PyTree

from .._custom_types import Args, BoolScalarLike, DenseInfo, RealScalarLike, VF, Y
from .._local_interpolation import LocalLinearInterpolation
from .._solution import RESULTS
from .._term import AbstractTerm
from .base import AbstractAdaptiveSolver, AbstractStratonovichSolver


_SolverState: TypeAlias = tuple[PyTree, PyTree]


class ReversibleHeun(AbstractAdaptiveSolver, AbstractStratonovichSolver):
    """Reversible Heun method.

    Algebraically reversible 2nd order method. Has an embedded 1st order method for
    adaptive step sizing. Uses 1st order local linear interpolation for dense/ts output.

    When used to solve SDEs, converges to the Stratonovich solution.

    ??? cite "Reference"

        ```bibtex
        @article{kidger2021efficient,
            author={Kidger, Patrick and Foster, James and Li, Xuechen and Lyons, Terry},
            title={{E}fficient and {A}ccurate {G}radients for {N}eural {SDE}s},
            year={2021},
            journal={Advances in Neural Information Processing Systems}
        }
        ```
    """

    term_structure: ClassVar = AbstractTerm
    interpolation_cls: ClassVar[
        Callable[..., LocalLinearInterpolation]
    ] = LocalLinearInterpolation  # TODO use something better than this?

    def order(self, terms):
        return 2

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
        del t1
        vf0 = terms.vf(t0, y0, args)
        return y0, vf0

    def step(
        self,
        terms: AbstractTerm,
        t0: RealScalarLike,
        t1: RealScalarLike,
        y0: Y,
        args: Args,
        solver_state: _SolverState,
        made_jump: BoolScalarLike,
    ) -> tuple[Y, Y, DenseInfo, _SolverState, RESULTS]:
        yhat0, vf0 = solver_state

        vf0 = lax.cond(made_jump, lambda _: terms.vf(t0, y0, args), lambda _: vf0, None)

        control = terms.contr(t0, t1)
        yhat1 = (2 * y0**ω - yhat0**ω + terms.prod(vf0, control) ** ω).ω
        vf1 = terms.vf(t1, yhat1, args)
        y1 = (y0**ω + 0.5 * terms.prod((vf0**ω + vf1**ω).ω, control) ** ω).ω
        y1_error = (0.5 * terms.prod((vf1**ω - vf0**ω).ω, control) ** ω).ω

        dense_info = dict(y0=y0, y1=y1)
        solver_state = (yhat1, vf1)
        return y1, y1_error, dense_info, solver_state, RESULTS.successful

    def func(self, terms: AbstractTerm, t0: RealScalarLike, y0: Y, args: Args) -> VF:
        return terms.vf(t0, y0, args)
