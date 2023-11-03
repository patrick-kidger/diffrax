from typing import Callable, NamedTuple, Tuple

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
from equinox.internal import ω

from ..custom_types import Array, Bool, DenseInfo, PyTree, Scalar
from ..local_interpolation import LocalLinearInterpolation
from ..solution import RESULTS
from ..term import AbstractTerm
from .base import AbstractImplicitSolver


_ErrorEstimate = None
_SolverState = None

ConstrainFn = Callable[[PyTree], Array]


class RattleVars(NamedTuple):
    p_1_2: PyTree  # Midpoint momentum
    q_1: PyTree  # Midpoint position
    p_1: PyTree  # final momentum
    lam: PyTree  # Midpoint Lagrange multiplier (state)
    mu: PyTree  # final Lagrange multiplier (momentum)


class Rattle(AbstractImplicitSolver):
    """Rattle method.

    2nd order symplectic method with constrains `constrain(x)=0`.

    ??? cite "Reference"

        ```bibtex
        @article{ANDERSEN198324,
            title = {Rattle: A “velocity” version of the shake
            algorithm for molecular dynamics calculations},
            journal = {Journal of Computational Physics},
            volume = {52},
            number = {1},
            pages = {24-34},
            year = {1983},
            author = {Hans C Andersen},
        }
        ```
    """

    term_structure = (AbstractTerm, AbstractTerm)
    interpolation_cls = LocalLinearInterpolation
    # Fix TypeError: non-default argument 'constrain' follows default argument
    constrain: ConstrainFn = None

    def order(self, terms):
        return 2

    def init(
        self,
        terms: Tuple[AbstractTerm, AbstractTerm],
        t0: Scalar,
        t1: Scalar,
        y0: PyTree,
        args: PyTree,
    ) -> _SolverState:
        return None

    def step(
        self,
        terms: Tuple[AbstractTerm, AbstractTerm],
        t0: Scalar,
        t1: Scalar,
        y0: Tuple[PyTree, PyTree],
        args: PyTree,
        solver_state: _SolverState,
        made_jump: Bool,
    ) -> Tuple[Tuple[PyTree, PyTree], _ErrorEstimate, DenseInfo, _SolverState, RESULTS]:
        del solver_state, made_jump

        term_1, term_2 = terms
        midpoint = (t1 + t0) / 2

        control1_half_1 = term_1.contr(t0, midpoint)
        control1_half_2 = term_1.contr(midpoint, t1)

        control2_half_1 = term_2.contr(t0, midpoint)
        control2_half_2 = term_2.contr(midpoint, t1)

        p0, q0 = y0

        def eq(x: RattleVars, args=None):
            _, vjp_fun = jax.vjp(self.constrain, q0)
            _, vjp_fun_mu = jax.vjp(self.constrain, x.q_1)

            zero = (
                (
                    p0**ω
                    - control1_half_1 * (vjp_fun(x.lam)[0]) ** ω
                    + term_1.vf_prod(t0, q0, args, control1_half_1) ** ω
                    - x.p_1_2**ω
                ).ω,
                (
                    q0**ω
                    + term_2.vf_prod(t0, x.p_1_2, args, control2_half_1) ** ω
                    + term_2.vf_prod(midpoint, x.p_1_2, args, control2_half_2) ** ω
                    - x.q_1**ω
                ).ω,
                self.constrain(x.q_1),
                (
                    x.p_1_2**ω
                    + term_1.vf_prod(midpoint, x.q_1, args, control1_half_2) ** ω
                    - (control1_half_2 * vjp_fun_mu(x.mu)[0] ** ω)
                    - x.p_1**ω
                ).ω,
                jax.jvp(self.constrain, (x.q_1,), (term_2.vf(t1, x.p_1, args),))[1],
            )
            return zero

        cs = jax.eval_shape(self.constrain, q0)

        init_vars = RattleVars(
            p_1_2=p0,
            q_1=(q0**ω * 2).ω,
            p_1=p0,
            lam=jtu.tree_map(jnp.zeros_like, cs),
            mu=jtu.tree_map(jnp.zeros_like, cs),
        )

        sol = self.nonlinear_solver(eq, init_vars, None)

        y1 = (sol.root.p_1, sol.root.q_1)
        dense_info = dict(y0=y0, y1=y1)
        return y1, None, dense_info, None, RESULTS.successful

    def func(
        self,
        terms: Tuple[AbstractTerm, AbstractTerm],
        t0: Scalar,
        y0: Tuple[PyTree, PyTree],
        args: PyTree,
    ) -> Tuple[PyTree, PyTree]:
        term_1, term_2 = terms
        y0_1, y0_2 = y0
        f1 = term_1.func(t0, y0_2, args)
        f2 = term_2.func(t0, y0_1, args)
        return (f1, f2)
