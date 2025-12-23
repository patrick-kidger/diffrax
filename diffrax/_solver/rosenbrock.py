from dataclasses import dataclass, field
from typing import ClassVar, TypeAlias

import equinox as eqx
import equinox.internal as eqxi
import jax
import jax.flatten_util as fu
import jax.numpy as jnp
import jax.tree_util as jtu
import lineax as lx
import numpy as np
from equinox.internal import ω

from .._custom_types import (
    Args,
    BoolScalarLike,
    DenseInfo,
    RealScalarLike,
    VF,
    Y,
)
from .._solution import RESULTS
from .._term import AbstractTerm
from .base import AbstractAdaptiveSolver


_SolverState: TypeAlias = None


@dataclass(frozen=True)
class RosenbrockTableau:
    """The coefficient tableau for Rosenbrock methods"""

    m_sol: np.ndarray
    m_error: np.ndarray

    a_lower: tuple[np.ndarray, ...]
    c_lower: tuple[np.ndarray, ...]

    α: np.ndarray
    γ: np.ndarray

    num_stages: int = field(init=False)

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
- `m_error`: the linear combination of stages to produce a lower-order solution
    for error estimation. Should be a NumPy array of shape `(k,)`. The error is
    calculated as the difference between the main solution (using `m_sol`) and
    this lower-order solution (using `m_error`), providing an estimate of the
    local truncation error for adaptive step size control.
- `α`: the time increment.
- `γ`: the vector field increment.
"""


class AbstractRosenbrock(AbstractAdaptiveSolver):
    r"""Abstract base class for Rosenbrock solvers for stiff equations.

    Subclasses should define `tableau` and `interpolation_cls` as class-level attributes
    `tableau` should be an instance of `diffrax.RosenbrockTableau`, and
    `interpolation_cls` should be an instance of `diffrax.AbstractLocalInterpolation`.
    """

    term_structure: ClassVar = AbstractTerm

    tableau: ClassVar[RosenbrockTableau]

    rodas: ClassVar[bool] = False

    linear_solver: lx.AbstractLinearSolver = lx.LU()

    def init(self, terms, t0, t1, y0, args) -> _SolverState:
        del t0, t1
        if any(
            eqx.is_array_like(xi) and jnp.iscomplexobj(xi)
            for xi in jtu.tree_leaves((terms, y0, args))
        ):
            # TODO: add complex dtype support.
            raise ValueError("rosenbrock does not support complex dtypes.")

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
        del solver_state, made_jump

        y0_leaves = jtu.tree_leaves(y0)
        sol_dtype = jnp.result_type(*y0_leaves)

        control = terms.contr(t0, t1)
        identity = jtu.tree_map(lambda leaf: jnp.ones_like(leaf), control)

        time_derivative = jax.jacfwd(lambda t: terms.vf_prod(t, y0, args, identity))(t0)
        time_derivative, unravel_t = fu.ravel_pytree(time_derivative)

        jacobian = jax.jacfwd(lambda y: terms.vf_prod(t0, y, args, identity))(y0)
        jacobian, _ = fu.ravel_pytree(jacobian)
        jacobian = jnp.reshape(jacobian, time_derivative.shape * 2)

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
        dt = jtu.tree_leaves(control)[0]
        eye = jnp.eye(len(time_derivative))
        if self.rodas:
            A = lx.MatrixLinearOperator(eye - dt * γ[0] * jacobian)
        else:
            A = lx.MatrixLinearOperator((eye / (dt * γ[0])) - jacobian)

        k = jnp.zeros(
            (self.tableau.num_stages,) + time_derivative.shape, dtype=sol_dtype
        )

        def body(buffer, stage):
            # Σ_j a_{stage j} · u_j
            u = buffer[...]
            y0_increment = jnp.tensordot(a_lower[stage], u, axes=[[0], [0]])

            if self.rodas:
                # control . Fy . Σ_j (c_{stage j}) · u_j
                vf_increment = jnp.tensordot(c_lower[stage], u, axes=[[0], [0]])
                vf_increment = dt * (jacobian @ vf_increment)
            else:
                # Σ_j (c_{stage j}/control) · u_j
                c_scaled_control = jax.vmap(lambda c: c / dt)(c_lower[stage])
                vf_increment = jnp.tensordot(c_scaled_control, u, axes=[[0], [0]])

            scaled_time_derivative = γ[stage] * time_derivative
            if self.rodas:
                # sqrt(control) * γ_i * Ft
                scaled_time_derivative = jnp.power(dt, 2) * scaled_time_derivative
            else:
                # control * γ_i * Ft
                scaled_time_derivative = dt * scaled_time_derivative

            vf = terms.vf_prod(
                (t0 + (α[stage] * dt)),
                (y0**ω + unravel_t(y0_increment) ** ω).ω,
                args,
                identity,
            )
            vf, unravel = fu.ravel_pytree(vf)
            if self.rodas:
                vf = dt * vf

            b = vf + vf_increment + scaled_time_derivative
            # solving Ax=b
            stage_k = lx.linear_solve(A, b).value

            buffer = buffer.at[stage].set(stage_k)
            return buffer, unravel(vf)

        k, stage_vf = eqxi.scan(
            f=body,
            init=k,
            xs=jnp.arange(0, self.tableau.num_stages),
            kind="checkpointed",
            buffers=lambda x: x,
            checkpoints="all",
        )

        y1_increment = jnp.tensordot(m_sol, k, axes=[[0], [0]])
        y1_lower_increment = jnp.tensordot(m_error, k, axes=[[0], [0]])
        y1_increment = unravel_t(y1_increment)
        y1_lower_increment = unravel_t(y1_lower_increment)

        y1 = (y0**ω + y1_increment**ω).ω
        y1_lower = (y0**ω + y1_lower_increment**ω).ω
        y1_error = (y1**ω - y1_lower**ω).ω

        if self.rodas:
            dense_info = dict(y0=y0, k=k)
        else:
            k1 = jtu.tree_map(lambda leaf: leaf[0] * dt, stage_vf)
            vf1 = terms.vf(t1, y1, args)
            k = jtu.tree_map(
                lambda k1, k2: jnp.stack([k1, k2]),
                k1,
                terms.prod(vf1, control),
            )
            dense_info = dict(y0=y0, y1=y1, k=k)

        return y1, y1_error, dense_info, None, RESULTS.successful

    def func(
        self,
        terms: AbstractTerm,
        t0: RealScalarLike,
        y0: Y,
        args: Args,
    ) -> VF:
        identity = jtu.tree_map(lambda leaf: jnp.ones_like(leaf), t0)
        return terms.vf_prod(t0, y0, args, identity)
