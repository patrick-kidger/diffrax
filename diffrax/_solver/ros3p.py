from collections.abc import Callable
from dataclasses import dataclass
from typing import ClassVar, TypeAlias

import equinox.internal as eqxi
import jax
import jax.lax as lax
import jax.numpy as jnp
import jax.tree_util as jtu
import lineax as lx
import numpy as np
from equinox.internal import ω
from jaxtyping import ArrayLike

from .._custom_types import (
    Args,
    BoolScalarLike,
    DenseInfo,
    RealScalarLike,
    VF,
    Y,
)
from .._local_interpolation import ThirdOrderHermitePolynomialInterpolation
from .._solution import RESULTS
from .._term import AbstractTerm
from .base import AbstractAdaptiveSolver

_SolverState: TypeAlias = VF


@dataclass(frozen=True)
class _RosenbrockTableau:
    """The coefficient tableau for Rosenbrock methods"""

    m_sol: np.ndarray
    m_error: np.ndarray

    a_lower: tuple[np.ndarray, ...]
    c_lower: tuple[np.ndarray, ...]

    α: np.ndarray
    γ: np.ndarray

    num_stages: int

    # Example tableau
    #
    # α1 | a11 a12 a13 | c11 c12 c13 | γ1
    # α1 | a21 a22 a23 | c21 c22 c23 | γ2
    # α3 | a31 a32 a33 | c31 c32 c33 | γ3
    # ---+----------------
    #    | m1  m2  m3
    #    | me1 me2 me3


_tableau = _RosenbrockTableau(
    m_sol=np.array([2.0, 0.5773502691896258, 0.4226497308103742]),
    m_error=np.array([2.113248654051871, 1.0, 0.4226497308103742]),
    a_lower=(
        np.array([1.267949192431123]),
        np.array([1.267949192431123, 0.0]),
    ),
    c_lower=(
        np.array([-1.607695154586736]),
        np.array([-3.464101615137755, -1.732050807568877]),
    ),
    α=np.array([0.0, 1.0, 1.0]),
    γ=np.array(
        [
            0.7886751345948129,
            -0.2113248654051871,
            -1.0773502691896260,
        ]
    ),
    num_stages=3,
)


class Ros3p(AbstractAdaptiveSolver):
    r"""Ros3p method.

    3rd order Rosenbrock method for solving stiff equation. Uses third-order Hermite
    polynomial interpolation for dense output.

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

    term_structure: ClassVar = AbstractTerm[ArrayLike, ArrayLike]
    interpolation_cls: ClassVar[
        Callable[..., ThirdOrderHermitePolynomialInterpolation]
    ] = ThirdOrderHermitePolynomialInterpolation.from_k

    tableau: ClassVar[_RosenbrockTableau] = _tableau

    def init(self, terms, t0, t1, y0, args) -> _SolverState:
        del t1
        return terms.vf(t0, y0, args)

    def order(self, terms):
        return 3

    def step(
        self,
        terms: AbstractTerm[ArrayLike, ArrayLike],
        t0: RealScalarLike,
        t1: RealScalarLike,
        y0: Y,
        args: Args,
        solver_state: _SolverState,
        made_jump: BoolScalarLike,
    ) -> tuple[Y, Y, DenseInfo, _SolverState, RESULTS]:
        y0_leaves = jtu.tree_leaves(y0)
        sol_dtype = jnp.result_type(*y0_leaves)

        time_derivative = jax.jacfwd(lambda t: terms.vf(t, y0, args))(t0)
        control = terms.contr(t0, t1)

        γ = jnp.array(self.tableau.γ, dtype=sol_dtype)
        α = jnp.array(self.tableau.α, dtype=sol_dtype)

        def embed_lower(x):
            out = np.zeros(
                (self.tableau.num_stages, self.tableau.num_stages), dtype=x[0].dtype
            )
            for i, val in enumerate(x):
                out[i + 1, : i + 1] = val
            return jnp.array(out, dtype=sol_dtype)

        a_lower = embed_lower(self.tableau.a_lower)
        c_lower = embed_lower(self.tableau.c_lower)
        m_sol = jnp.array(self.tableau.m_sol, dtype=sol_dtype)
        m_error = jnp.array(self.tableau.m_error, dtype=sol_dtype)

        # common L.H.S
        eye_shape = jax.ShapeDtypeStruct(time_derivative.shape, dtype=sol_dtype)
        A = (lx.IdentityLinearOperator(eye_shape) / (control * γ[0])) - (
            lx.JacobianLinearOperator(
                lambda y, args: terms.vf(t0, y, args), y0, args=args
            )
        )

        u = jnp.zeros(
            (self.tableau.num_stages,) + time_derivative.shape, dtype=sol_dtype
        )

        def use_saved_vf(u):
            stage_0_vf = solver_state
            stage_0_b = (
                stage_0_vf**ω + (control**ω * γ[0] ** ω * time_derivative**ω)
            ).ω
            stage_0_u = lx.linear_solve(A, stage_0_b).value

            u = u.at[0].set(stage_0_u)
            start_stage = 1
            return u, start_stage

        if made_jump is False:
            u, start_stage = use_saved_vf(u)
        else:
            u, start_stage = lax.cond(
                eqxi.unvmap_any(made_jump), lambda u: (u, 0), use_saved_vf, u
            )

        def body(u, stage):
            vf = terms.vf(
                (t0**ω + α[stage] ** ω * control**ω).ω,
                (
                    y0**ω
                    + (a_lower[stage][0] ** ω * u[0] ** ω)
                    + (a_lower[stage][1] ** ω * u[1] ** ω)
                ).ω,
                args,
            )
            b = (
                vf**ω
                + ((c_lower[stage][0] ** ω / control**ω) * u[0] ** ω)
                + ((c_lower[stage][1] ** ω / control**ω) * u[1] ** ω)
                + (control**ω * γ[stage] ** ω * time_derivative**ω)
            ).ω
            stage_u = lx.linear_solve(A, b).value
            u = u.at[stage].set(stage_u)
            return u, vf

        u, stage_vf = lax.scan(
            f=body, init=u, xs=jnp.arange(start_stage, self.tableau.num_stages)
        )

        y1 = (
            y0**ω
            + m_sol[0] ** ω * u[0] ** ω
            + m_sol[1] ** ω * u[1] ** ω
            + m_sol[2] ** ω * u[2] ** ω
        ).ω
        y1_lower = (
            y0**ω
            + m_error[0] ** ω * u[0] ** ω
            + m_error[1] ** ω * u[1] ** ω
            + m_error[2] ** ω * u[2] ** ω
        ).ω
        y1_error = y1 - y1_lower

        if start_stage == 0:
            vf0 = stage_vf[0]  # type: ignore
        else:
            vf0 = solver_state
        vf1 = terms.vf(t1, y1, args)
        k = jnp.stack((terms.prod(vf0, control), terms.prod(vf1, control)))

        dense_info = dict(y0=y0, y1=y1, k=k)
        return y1, y1_error, dense_info, vf1, RESULTS.successful

    def func(
        self,
        terms: AbstractTerm[ArrayLike, ArrayLike],
        t0: RealScalarLike,
        y0: Y,
        args: Args,
    ) -> VF:
        return terms.vf(t0, y0, args)


Ros3p.__init__.__doc__ = """**Arguments:** None"""
