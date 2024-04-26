import operator
from typing import Optional, Tuple

import equinox as eqx
import jax.tree_util as jtu
import lineax as lx
from jax import numpy as jnp
import jax
from jaxtyping import PyTree

from .._custom_types import Args, BoolScalarLike, DenseInfo, RealScalarLike, VF, Y
from .._heuristics import is_sde
from .._solution import RESULTS
from .._term import (
    AbstractTerm,
    ControlTerm,
    ODETerm,
    WeaklyDiagonalControlTerm,
)
from .base import (
    _SolverState as _AbstractSolverState,
    AbstractSolver,
    AbstractWrappedSolver,
)


_SolverState = Tuple[_AbstractSolverState, RealScalarLike]


class KLSolver(AbstractWrappedSolver[_SolverState]):
    """
    SDE KL Divergence. Only works with SDEs.

    Terms must be (term1, term2), both SDEs, same diffusion, diff drift.
    """

    solver: AbstractSolver[_SolverState]
    linear_solver: lx.AbstractLinearSolver = lx.AutoLinearSolver(well_posed=None)

    def order(self, terms: PyTree[AbstractTerm]) -> Optional[int]:
        return self.solver.order(terms)

    def strong_order(self, terms: PyTree[AbstractTerm]) -> Optional[RealScalarLike]:
        return self.solver.strong_order(terms)

    def error_order(self, terms: PyTree[AbstractTerm]) -> Optional[RealScalarLike]:
        if is_sde(terms):
            order = self.strong_order(terms)
        else:
            order = self.order(terms)
        return order

    @property
    def term_structure(self):
        return self.solver.term_structure

    @property
    def interpolation_cls(self):  # pyright: ignore
        return self.solver.interpolation_cls

    def init(
        self,
        terms: PyTree[AbstractTerm],
        t0: RealScalarLike,
        t1: RealScalarLike,
        y0: Y,
        args: Args,
    ) -> _SolverState:
        return self.solver.init(terms, t0, t1, y0, args)

    def step(
        self,
        terms: PyTree[AbstractTerm],
        t0: RealScalarLike,
        t1: RealScalarLike,
        y0: Y,
        args: Args,
        solver_state: _SolverState,
        made_jump: BoolScalarLike,
    ) -> tuple[Y, Optional[Y], DenseInfo, _SolverState, RESULTS]:
        terms1, terms2 = terms.terms
        # print(terms1, terms2)
        y1, y_error, dense_info, solver_state, result = self.solver.step(
            terms1, t0, t1, y0, args, solver_state, made_jump
        )
        return y1, y_error, dense_info, solver_state, result

    def func(
        self, terms: PyTree[AbstractTerm], t0: RealScalarLike, y0: Y, args: Args
    ) -> VF:
        return self.solver.func(terms, t0, y0, args)


def evaluate_kl(terms: PyTree[AbstractTerm], t0: RealScalarLike, y0: Y, args: Args, linear_solver: lx.AbstractLinearSolver) -> RealScalarLike:
    terms1, terms2 = terms.terms
    drift_term1 = jtu.tree_map(
        lambda x: x if isinstance(x, ODETerm) else None,
        terms1,
        is_leaf=lambda x: isinstance(x, ODETerm),
    )
    drift_term1 = jtu.tree_leaves(
        drift_term1, is_leaf=lambda x: isinstance(x, ODETerm)
    )
    drift_term2 = jtu.tree_map(
        lambda x: x if isinstance(x, ODETerm) else None,
        terms2,
        is_leaf=lambda x: isinstance(x, ODETerm),
    )
    drift_term2 = jtu.tree_leaves(
        drift_term2, is_leaf=lambda x: isinstance(x, ODETerm)
    )

    drift_term1 = eqx.error_if(
        drift_term1, len(drift_term1) != 1, "First SDE doesn't have one ODETerm!"
    )
    drift_term2 = eqx.error_if(
        drift_term2, len(drift_term2) != 1, "Second SDE doesn't have one ODETerm!"
    )
    # print(drift_term1, drift_term2)
    drift_term1, drift_term2 = drift_term1[0], drift_term2[0]

    drift1 = drift_term1.vf(t0, y0, args)
    drift2 = drift_term2.vf(t0, y0, args)
    # print('d1d2', drift1, drift2)
    drift = jtu.tree_map(operator.sub, drift1, drift2)

    diffusion_term = jtu.tree_map(
        lambda x: x
        if isinstance(x, ControlTerm) or isinstance(x, WeaklyDiagonalControlTerm)
        else None,
        terms1,
        is_leaf=lambda x: isinstance(x, ControlTerm)
        or isinstance(x, WeaklyDiagonalControlTerm),
    )
    diffusion_term = jtu.tree_leaves(
        diffusion_term,
        is_leaf=lambda x: isinstance(x, ControlTerm)
        or isinstance(x, WeaklyDiagonalControlTerm),
    )

    diffusion_term = eqx.error_if(
        diffusion_term, len(diffusion_term) != 1, "SDE has multiple control terms!"
    )
    diffusion_term = diffusion_term[0]

    diffusion = diffusion_term.vf(t0, y0, args)  # assumes same diffusion

    diffusion = jtu.tree_map(lambda x: jnp.where(
        jax.lax.stop_gradient(x) > 1e-7,
        x,
        jnp.full_like(x, fill_value=1e-7) * jnp.sign(x),
    ), diffusion)

    drift_tree_structure = jtu.tree_structure(drift)
    diffusion_tree_structure = jtu.tree_structure(diffusion)
    # print(drift, diffusion)
    if drift_tree_structure == diffusion_tree_structure:
        if isinstance(diffusion_term, WeaklyDiagonalControlTerm):
            diffusion_linear_operator = jtu.tree_map(
                lx.DiagonalLinearOperator, diffusion
            )
        else:
            diffusion_linear_operator = jtu.tree_map(
                lx.MatrixLinearOperator, diffusion
            )

        divergences = jtu.tree_map(
            lambda a, b: lx.linear_solve(a, b, solver=linear_solver).value,
            diffusion_linear_operator,
            drift,
            is_leaf=lambda x: eqx.is_array(x)
            or isinstance(x, lx.AbstractLinearOperator),
        )

        kl_divergence = jtu.tree_map(lambda x: 0.5 * jnp.sum(x) ** 2, divergences)
        kl_divergence = jtu.tree_reduce(operator.add, kl_divergence)

    else:
        raise ValueError(
            "drift and diffusion should have the same PyTree structure"
            + f" \n {drift_tree_structure} != {diffusion_tree_structure}"
        )
    
    return jnp.squeeze(kl_divergence)