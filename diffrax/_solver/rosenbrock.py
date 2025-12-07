from collections.abc import Callable
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
from .._local_interpolation import ThirdOrderHermitePolynomialInterpolation
from .._solution import RESULTS
from .._term import AbstractTerm, ODETerm, WrapTerm
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

    Uses third-order Hermite polynomial interpolation for dense output.

    Subclasses should define `tableau` as a class-level attribute that is an
    instance of `diffrax.RosenbrockTableau`.
    """

    term_structure: ClassVar = AbstractTerm
    interpolation_cls: ClassVar[
        Callable[..., ThirdOrderHermitePolynomialInterpolation]
    ] = ThirdOrderHermitePolynomialInterpolation.from_k

    tableau: ClassVar[RosenbrockTableau]

    linear_solver: lx.AbstractLinearSolver = lx.AutoLinearSolver(well_posed=True)

    def init(self, terms, t0, t1, y0, args) -> _SolverState:
        del t0, t1
        if any(
            eqx.is_array_like(xi) and jnp.iscomplexobj(xi)
            for xi in jtu.tree_leaves((terms, y0, args))
        ):
            # TODO: add complex dtype support.
            raise ValueError("rosenbrock does not support complex dtypes.")

        if isinstance(terms, ODETerm):
            return

        if isinstance(terms, WrapTerm):
            inner_term = terms.term
            if isinstance(inner_term, ODETerm):
                return

        raise NotImplementedError(
            f"Cannot use `terms={type(terms).__name__}`."
            "Consider using terms=ODETerm(...)."
        )

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

        time_derivative = jax.jacfwd(lambda t: terms.vf(t, y0, args))(t0)
        time_derivative, unravel_t = fu.ravel_pytree(time_derivative)
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
        in_structure = jax.eval_shape(lambda: y0)
        A = (lx.IdentityLinearOperator(in_structure) / (control * γ[0])) - (
            lx.JacobianLinearOperator(
                lambda y, args: terms.vf(t0, y, args), y0, args=args
            )
        )

        u = jnp.zeros(
            (self.tableau.num_stages,) + time_derivative.shape, dtype=sol_dtype
        )

        def body(buffer, stage):
            # Σ_j a_{stage j} · u_j
            u = buffer[...]
            y0_increment = jnp.tensordot(a_lower[stage], u, axes=[[0], [0]])
            # Σ_j (c_{stage j}/control) · u_j
            c_scaled_control = jax.vmap(lambda c: c / control)(c_lower[stage])
            vf_increment = jnp.tensordot(c_scaled_control, u, axes=[[0], [0]])
            # control * γ_i * Ft
            scaled_time_derivative = control * γ[stage] * time_derivative

            y0_increment = unravel_t(y0_increment)
            vf_increment = unravel_t(vf_increment)
            scaled_time_derivative = unravel_t(scaled_time_derivative)

            vf = terms.vf(
                (t0**ω + (α[stage] ** ω * control**ω)).ω,
                (y0**ω + y0_increment**ω).ω,
                args,
            )
            b = (vf**ω + vf_increment**ω + scaled_time_derivative**ω).ω
            # solving Ax=b
            stage_u = lx.linear_solve(A, b, self.linear_solver).value
            stage_u, _ = fu.ravel_pytree(stage_u)
            buffer = buffer.at[stage].set(stage_u)
            return buffer, vf

        u, stage_vf = eqxi.scan(
            f=body,
            init=u,
            xs=jnp.arange(0, self.tableau.num_stages),
            kind="checkpointed",
            buffers=lambda x: x,
            checkpoints="all",
        )

        y1_increment = jnp.tensordot(m_sol, u, axes=[[0], [0]])
        y1_lower_increment = jnp.tensordot(m_error, u, axes=[[0], [0]])
        y1_increment = unravel_t(y1_increment)
        y1_lower_increment = unravel_t(y1_lower_increment)

        y1 = (y0**ω + y1_increment**ω).ω
        y1_lower = (y0**ω + y1_lower_increment**ω).ω
        y1_error = (y1**ω - y1_lower**ω).ω

        vf0 = jtu.tree_map(lambda stage_vf: stage_vf[0], stage_vf)
        vf1 = terms.vf(t1, y1, args)
        k = jnp.stack(
            (
                jnp.asarray(terms.prod(vf0, control)),
                jnp.asarray(terms.prod(vf1, control)),
            )
        )
        dense_info = dict(y0=jnp.asarray(y0), y1=jnp.asarray(y1), k=k)

        return y1, y1_error, dense_info, None, RESULTS.successful

    def func(
        self,
        terms: AbstractTerm,
        t0: RealScalarLike,
        y0: Y,
        args: Args,
    ) -> VF:
        return terms.vf(t0, y0, args)
