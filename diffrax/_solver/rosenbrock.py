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
class RosenbrockTableau:
    """The coefficient tableau for Rosenbrock methods"""

    m_sol: np.ndarray
    m_error: np.ndarray

    a_lower: tuple[np.ndarray, ...]
    c_lower: tuple[np.ndarray, ...]

    α: np.ndarray
    γ: np.ndarray

    num_stages: int

    def __post_init__(self):
        assert self.α.ndim == 1
        assert self.γ.ndim == 1
        assert self.m_sol.ndim == 1
        assert self.m_error.ndim == 1
        assert self.α.shape[0] - 1 == len(self.a_lower)
        assert self.α.shape[0] - 1 == len(self.c_lower)
        assert self.α.shape[0] == self.γ.shape[0]
        assert all(i + 1 == a_i.shape[0] for i, a_i in enumerate(self.a_lower))
        assert all(i + 1 == a_i.shape[0] for i, a_i in enumerate(self.c_lower))
        object.__setattr__(self, "num_stages", len(self.m_sol))


RosenbrockTableau.__init__.__doc__ = """**Arguments:**

Example tableau
α1 | a11 a12 a13 | c11 c12 c13 | γ1
α1 | a21 a22 a23 | c21 c22 c23 | γ2
α3 | a31 a32 a33 | c31 c32 c33 | γ3
---+----------------
   | m1  m2  m3
   | me1 me2 me3

Let `k` denote the number of stages of the solver.

- `a_lower`: the lower triangle (without the diagonal) of the tableau. Should
    be a tuple of NumPy arrays, corresponding to the rows of this lower triangle. The
    first array represents the should be of shape `(1,)`. Each subsequent array should
    be of shape `(2,)`, `(3,)` etc. The final array should have shape `(k - 1,)`.
- `c_lower`: the lower triangle (without the diagonal) of the tableau. Should
    be a tuple of NumPy arrays, corresponding to the rows of this lower triangle. The
    first array represents the should be of shape `(1,)`. Each subsequent array should
    be of shape `(2,)`, `(3,)` etc. The final array should have shape `(k - 1,)`.
- `m_sol`: the linear combination of stages to take to produce the output at each step.
    Should be a NumPy array of shape `(k,)`.
- `m_error`: the linear combination of stages to take to produce the error estimate at
    each step. Should be a NumPy array of shape `(k,)`. Note that this is *not*
    differenced against `b_sol` prior to evaluation. (i.e. `b_error` gives the linear
    combination for producing the error estimate directly, not for producing some
    alternate solution that is compared against the main solution).
- `α`: the time increment.
- `γ`: the vector field increment.
"""


class AbstractRosenbrock(AbstractAdaptiveSolver):
    r"""Abstract base class for Rosenbrock solvers for stiff equations.

    Uses third-order Hermite polynomial interpolation for dense output.

    Subclasses should define `tableau` as a class-level attribute that is an
    instance of `diffrax.RosenbrockTableau`.
    """

    term_structure: ClassVar = AbstractTerm[ArrayLike, ArrayLike]
    interpolation_cls: ClassVar[
        Callable[..., ThirdOrderHermitePolynomialInterpolation]
    ] = ThirdOrderHermitePolynomialInterpolation.from_k

    tableau: ClassVar[RosenbrockTableau]

    def init(self, terms, t0, t1, y0, args) -> _SolverState:
        del t1
        return terms.vf(t0, y0, args)

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
            stage_0_b = stage_0_vf + ((control * γ[0]) * time_derivative)
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
            # Σ_j a_{stage j} · u_j
            y0_increment = jnp.tensordot(a_lower[stage], u, axes=[[0], [0]])
            vf = terms.vf(
                t0 + (α[stage] * control),
                y0 + y0_increment,
                args,
            )

            # Σ_j (c_{stage j}/control) · u_j
            c_scaled_control = jax.vmap(lambda c: c / control)(c_lower[stage])
            vf_increment = jnp.tensordot(c_scaled_control, u, axes=[[0], [0]])

            b = vf + vf_increment + ((control * γ[stage]) * time_derivative)
            # solving Ax=b
            stage_u = lx.linear_solve(A, b).value
            u = u.at[stage].set(stage_u)
            return u, vf

        u, stage_vf = lax.scan(
            f=body, init=u, xs=jnp.arange(start_stage, self.tableau.num_stages)
        )

        y1 = y0 + jnp.tensordot(m_sol, u, axes=[[0], [0]])
        y1_lower = y0 + jnp.tensordot(m_error, u, axes=[[0], [0]])
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
