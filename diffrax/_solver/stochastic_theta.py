from collections.abc import Callable
from typing import ClassVar
from typing_extensions import TypeAlias

import optimistix as optx
from equinox.internal import ω

from .._custom_types import Args, BoolScalarLike, DenseInfo, RealScalarLike, VF, Y
from .._local_interpolation import LocalLinearInterpolation
from .._root_finder import with_stepsize_controller_tols
from .._solution import RESULTS
from .._term import AbstractTerm, MultiTerm, ODETerm
from .base import AbstractAdaptiveSolver, AbstractImplicitSolver, AbstractItoSolver


_SolverState: TypeAlias = None


def _implicit_relation(z1, nonlinear_solve_args):
    (
        vf_prod_drift,
        t1,
        y0,
        args,
        control,
        k0_drift,
        k0_diff,
        theta,
    ) = nonlinear_solve_args
    add_state = (y0**ω + z1**ω).ω
    implicit_drift = (vf_prod_drift(t1, add_state, args, control) ** ω * theta).ω
    euler_drift = ((1 - theta) * k0_drift**ω).ω
    diff = (z1**ω - (implicit_drift**ω + euler_drift**ω + k0_diff**ω).ω ** ω).ω
    return diff


class StochasticTheta(
    AbstractImplicitSolver,
    AbstractAdaptiveSolver,
    AbstractItoSolver,
):
    r"""Stochastic Theta method.

    Stochastic A stable 0.5 strong order (1.0 weak order) SDIRK method. Has an embedded
    1st order Euler method for adaptive step sizing. Uses 1 stage. Uses a 1st order local
    linear interpolation for dense/ts output.

    !!! warning

        If `theta` is 0, this results in an explicit Euler step, which is also how the error
        estimate is computed (which would result in estimated error being 0).

    ??? cite "Reference"

        ```bibtex
        @article{higham2000mean,
            title={Mean-square and asymptotic stability of the stochastic theta method},
            author={Higham, Desmond J},
            journal={SIAM journal on numerical analysis},
            volume={38},
            number={3},
            pages={753--769},
            year={2000},
            publisher={SIAM}
        }
        ```
    """

    theta: float
    term_structure: ClassVar = MultiTerm[tuple[ODETerm, AbstractTerm]]
    interpolation_cls: ClassVar[
        Callable[..., LocalLinearInterpolation]
    ] = LocalLinearInterpolation
    root_finder: optx.AbstractRootFinder = with_stepsize_controller_tols(optx.Chord)()
    root_find_max_steps: int = 10

    def order(self, terms):
        return 1

    def error_order(self, terms):
        return 1.0

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
        terms: MultiTerm[tuple[ODETerm, AbstractTerm]],
        t0: RealScalarLike,
        t1: RealScalarLike,
        y0: Y,
        args: Args,
        solver_state: _SolverState,
        made_jump: BoolScalarLike,
    ) -> tuple[Y, Y, DenseInfo, _SolverState, RESULTS]:
        del made_jump
        control = terms.contr(t0, t1)
        k0_drift = terms.terms[0].vf_prod(t0, y0, args, control[0])
        k0_diff = terms.terms[1].vf_prod(t0, y0, args, control[1])
        root_args = (
            terms.terms[0].vf_prod,
            t1,
            y0,
            args,
            control[0],
            k0_drift,
            k0_diff,
            self.theta,
        )
        nonlinear_sol = optx.root_find(
            _implicit_relation,
            self.root_finder,
            k0_drift,
            root_args,
            throw=False,
            max_steps=self.root_find_max_steps,
        )
        k1 = nonlinear_sol.value
        y1 = (y0**ω + k1**ω).ω
        # Use the trapezoidal rule for adaptive step sizing.
        k0 = (k0_drift**ω + k0_diff**ω).ω
        y_error = (0.5 * (k1**ω - k0**ω)).ω
        dense_info = dict(y0=y0, y1=y1)
        solver_state = None
        result = RESULTS.promote(nonlinear_sol.result)
        return y1, y_error, dense_info, solver_state, result

    def func(
        self,
        terms: AbstractTerm,
        t0: RealScalarLike,
        y0: Y,
        args: Args,
    ) -> VF:
        return terms.vf(t0, y0, args)
