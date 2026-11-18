from collections.abc import Callable
from typing import ClassVar, TypeAlias

import equinox.internal as eqxi
import jax.lax as lax
from equinox.internal import ω
from jaxtyping import ArrayLike, Float, PyTree

from .._custom_types import Args, BoolScalarLike, DenseInfo, RealScalarLike, VF
from .._local_interpolation import LocalLinearInterpolation
from .._solution import RESULTS
from .._term import AbstractTerm
from .base import AbstractSolver


_ErrorEstimate: TypeAlias = None
_SolverState: TypeAlias = VF

Ya: TypeAlias = PyTree[Float[ArrayLike, "?*y"], " Y"]  # pyright: ignore[reportUndefinedVariable]
Yb: TypeAlias = PyTree[Float[ArrayLike, "?*y"], " Y"]  # pyright: ignore[reportUndefinedVariable]


class StormerVerlet(AbstractSolver):
    """Störmer--Verlet method, in its velocity Verlet (kick-drift-kick leapfrog) form.

    2nd order symplectic method. Does not support adaptive step sizing. Uses 1st order
    local linear interpolation for dense/ts output.

    Letting `v, w = y0` and `f, g = terms`, each step is a half step of `w`, a full
    step of `v`, and another half step of `w`:

    $w_{1/2} = w_0 + \\frac{h}{2} g(t_0, v_0)$

    $v_1 = v_0 + h f(t_0, w_{1/2})$

    $w_1 = w_{1/2} + \\frac{h}{2} g(t_1, v_1)$

    The vector field `g(t_1, v_1)` from the end of one step is reused at the start of
    the next, so each step costs one evaluation of each of `f` and `g`. After a jump
    (e.g. from `jump_ts`) it is evaluated afresh.

    ??? cite "Reference"

        ```bibtex
        @article{hairer2003geometric,
            title={Geometric numerical integration illustrated by the
                   St{\\"o}rmer--Verlet method},
            author={Hairer, Ernst and Lubich, Christian and Wanner, Gerhard},
            journal={Acta Numerica},
            volume={12},
            pages={399--450},
            year={2003},
            publisher={Cambridge University Press}
        }
        ```
    """

    term_structure: ClassVar = (AbstractTerm, AbstractTerm)
    interpolation_cls: ClassVar[Callable[..., LocalLinearInterpolation]] = (
        LocalLinearInterpolation
    )

    def order(self, terms):
        return 2

    def init(
        self,
        terms: tuple[AbstractTerm, AbstractTerm],
        t0: RealScalarLike,
        t1: RealScalarLike,
        y0: tuple[Ya, Yb],
        args: Args,
    ) -> _SolverState:
        _, term_2 = terms
        y0_1, _ = y0
        return term_2.vf(t0, y0_1, args)

    def step(
        self,
        terms: tuple[AbstractTerm, AbstractTerm],
        t0: RealScalarLike,
        t1: RealScalarLike,
        y0: tuple[Ya, Yb],
        args: Args,
        solver_state: _SolverState,
        made_jump: BoolScalarLike,
    ) -> tuple[tuple[Ya, Yb], _ErrorEstimate, DenseInfo, _SolverState, RESULTS]:
        term_1, term_2 = terms
        y0_1, y0_2 = y0

        # `g(t0, v0)`, stored from the end of the previous step. After a jump it has to
        # be recomputed; `made_jump` is a static `False` when no jump is possible.
        vf0_2 = solver_state
        if made_jump is not False:
            vf0_2 = lax.cond(
                eqxi.unvmap_any(made_jump),
                lambda: term_2.vf(t0, y0_1, args),
                lambda: vf0_2,
            )

        control1 = term_1.contr(t0, t1)
        control2 = term_2.contr(t0, t1)
        half_control2 = (0.5 * control2**ω).ω

        yhalf_2 = (y0_2**ω + term_2.prod(vf0_2, half_control2) ** ω).ω
        y1_1 = (y0_1**ω + term_1.vf_prod(t0, yhalf_2, args, control1) ** ω).ω
        vf1_2 = term_2.vf(t1, y1_1, args)
        y1_2 = (yhalf_2**ω + term_2.prod(vf1_2, half_control2) ** ω).ω

        y1 = (y1_1, y1_2)
        dense_info = dict(y0=y0, y1=y1)
        return y1, None, dense_info, vf1_2, RESULTS.successful

    def func(
        self,
        terms: tuple[AbstractTerm, AbstractTerm],
        t0: RealScalarLike,
        y0: tuple[Ya, Yb],
        args: Args,
    ) -> VF:
        term_1, term_2 = terms
        y0_1, y0_2 = y0
        f1 = term_1.vf(t0, y0_2, args)
        f2 = term_2.vf(t0, y0_1, args)
        return f1, f2


StormerVerlet.__init__.__doc__ = """**Arguments:** None"""
