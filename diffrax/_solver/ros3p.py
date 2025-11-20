from collections.abc import Callable
from dataclasses import dataclass
from typing import ClassVar, TypeAlias

import jax
import jax.numpy as jnp
import lineax as lx
from equinox.internal import ω

from .._custom_types import Args, BoolScalarLike, DenseInfo, RealScalarLike, VF, Y
from .._local_interpolation import LocalLinearInterpolation
from .._solution import RESULTS
from .._term import AbstractTerm
from .base import AbstractAdaptiveSolver


_SolverState: TypeAlias = None


@dataclass(frozen=True)
class RosenbrockTableau:
    """The coefficient tableau for Rosenbrock methods"""

    m_sol: jnp.ndarray
    m_error: jnp.ndarray

    a_lower: tuple[jnp.ndarray, ...]
    c_lower: tuple[jnp.ndarray, ...]

    α: jnp.ndarray
    γ: jnp.ndarray

    # Example tableau
    #
    # α1 | a11 a12 a13 | c11 c12 c13 | γ1
    # α1 | a21 a22 a23 | c21 c22 c23 | γ2
    # α3 | a31 a32 a33 | c31 c32 c33 | γ3
    # ---+----------------
    #    | m1  m2  m3
    #    | me1 me2 me3


RosenbrockTableau.__init__.__doc__ = """**Arguments:**

- m_sol: the linear combination of stages to produce the increment of the solution.
- m_error: the linear combination of stages to produce the increment of lower order
    solution. It is used for error estimation.
- a_lower: the lower triangle of a[i][j] matrix. The first array represents the 
    should be of shape `(1,)`. Each subsequent array should be of shape `(2,)`, 
    `(3,)` etc. The final array should have shape `(k - 1,)`. It is linear combination
    of previous stage to calculate the current stage and used as increment for y.
- c_lower: the lower triangle of c[i][j] matrix. The first array represents the 
    should be of shape `(1,)`. Each subsequent array should be of shape `(2,)`, 
    `(3,)` etc. The final array should have shape `(k - 1,)`.It is linear combination
    of previous stage, used as stability increment for current stage.
- α: the time increment coefficient.
- γ: the stage multipler for time derivative.

"""

_tableau = RosenbrockTableau(
    m_sol=jnp.array([2.0, 0.5773502691896258, 0.4226497308103742]),
    m_error=jnp.array([2.113248654051871, 1.0, 0.4226497308103742]),
    a_lower=(jnp.array([1.267949192431123]), jnp.array([1.267949192431123, 0.0])),
    c_lower=(
        jnp.array([-1.607695154586736]),
        jnp.array([-3.464101615137755, -1.732050807568877]),
    ),
    α=jnp.array([0.0, 1.0, 1.0]),
    γ=jnp.array(
        [
            0.7886751345948129,
            -0.2113248654051871,
            -1.0773502691896260,
        ]
    ),
)


class Ros3p(AbstractAdaptiveSolver):
    r"""Ros3p method.

    3rd order Rosenbrock method for solving stiff equation. Uses a 1st order local linear
    interpolation for dense output.

    ??? cite "Reference"

        ```bibtex
        @article{LangVerwer2001ROS3P,
          author    = {Lang, J. and Verwer, J.},
          title     = {ROS3P---An Accurate Third-Order Rosenbrock Solver Designed
                       for Parabolic Problems},
          journal   = {BIT Numerical Mathematics},
          volume    = {41},
          number    = {4},
          pages     = {731--738},
          year      = {2001},
          doi       = {10.1023/A:1021900219772}
         }
         ```
    """

    term_structure: ClassVar = AbstractTerm
    interpolation_cls: ClassVar[Callable[..., LocalLinearInterpolation]] = (
        LocalLinearInterpolation
    )

    tableau: ClassVar[RosenbrockTableau] = _tableau

    def init(self, terms, t0, t1, y0, args) -> _SolverState:
        del terms, t0, t1, y0, args
        return None

    def order(self, terms):
        return 3

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
        del made_jump, solver_state

        time_derivative = jax.jacfwd(lambda t: terms.vf(t, y0, args))(t0)

        eye = jnp.eye(len(time_derivative))
        control = terms.contr(t0, t1)

        # common L.H.S
        A = (lx.MatrixLinearOperator(eye) / (control * self.tableau.γ[0])) - (
            lx.JacobianLinearOperator(
                lambda y, args: terms.vf(t0, y, args), y0, args=args
            )
        )

        # stage 1
        stage_1_b = (
            terms.vf(
                (t0**ω + (self.tableau.α[0] ** ω * control**ω)).ω,
                y0,
                args,
            )
            ** ω
            + (control**ω * self.tableau.γ[0] ** ω * time_derivative**ω)
        ).ω

        # solving Ax=b
        u1 = lx.linear_solve(A, stage_1_b).value

        # stage 2
        stage_2_b = (
            terms.vf(
                (t0**ω + (self.tableau.α[1] ** ω * control**ω)).ω,
                (y0**ω + (self.tableau.a_lower[0][0] ** ω * u1**ω)).ω,
                args,
            )
            ** ω
            + ((self.tableau.c_lower[0][0] ** ω / control**ω) * u1**ω)
            + (control**ω * self.tableau.γ[1] ** ω * time_derivative**ω)
        ).ω

        # solving Ax=b
        u2 = lx.linear_solve(A, stage_2_b).value

        # stage 3
        stage_3_b = (
            terms.vf(
                (t0**ω + self.tableau.α[2] ** ω * control**ω).ω,
                (
                    y0**ω
                    + (self.tableau.a_lower[1][0] ** ω * u1**ω)
                    + (self.tableau.a_lower[1][1] ** ω * u2**ω)
                ).ω,
                args,
            )
            ** ω
            + ((self.tableau.c_lower[1][0] ** ω / control**ω) * u1**ω)
            + ((self.tableau.c_lower[1][1] ** ω / control**ω) * u2**ω)
            + (control**ω * self.tableau.γ[2] ** ω * time_derivative**ω)
        ).ω

        # solving Ax=b
        u3 = lx.linear_solve(A, stage_3_b).value

        y1 = (
            y0**ω
            + self.tableau.m_sol[0] ** ω * u1**ω
            + self.tableau.m_sol[1] ** ω * u2**ω
            + self.tableau.m_sol[2] ** ω * u3**ω
        ).ω
        y1_lower = (
            y0**ω
            + self.tableau.m_error[0] ** ω * u1**ω
            + self.tableau.m_error[1] ** ω * u2**ω
            + self.tableau.m_error[2] ** ω * u3**ω
        ).ω

        y1_error = y1 - y1_lower
        dense_info = dict(y0=y0, y1=y1)
        return y1, y1_error, dense_info, None, RESULTS.successful

    def func(
        self,
        terms: AbstractTerm,
        t0: RealScalarLike,
        y0: Y,
        args: Args,
    ) -> VF:
        return terms.vf(t0, y0, args)


Ros3p.__init__.__doc__ = """**Arguments:** None"""
