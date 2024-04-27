import operator
from typing import Optional, Tuple, Union

import equinox as eqx
import jax
import jax.tree_util as jtu
import lineax as lx
from jax import numpy as jnp
from jaxtyping import PyTree

from .._custom_types import (
    Args,
    BoolScalarLike,
    Control,
    DenseInfo,
    RealScalarLike,
    VF,
    Y,
)
from .._heuristics import is_sde
from .._solution import RESULTS
from .._term import (
    AbstractTerm,
    ControlTerm,
    MultiTerm,
    ODETerm,
    WeaklyDiagonalControlTerm,
)
from .base import (
    _SolverState,
    AbstractSolver,
    AbstractWrappedSolver,
)


_ControlTerm = Union[ControlTerm, WeaklyDiagonalControlTerm]


def _compute_kl_integral(
    drift_term1: ODETerm,
    drift_term2: ODETerm,
    diffusion_term: _ControlTerm,
    t0: RealScalarLike,
    y0: Y,
    args: Args,
    linear_solver: lx.AbstractLinearSolver,
) -> Tuple[VF, RealScalarLike]:
    """
    Compute the KL divergence.
    """
    drift1 = drift_term1.vf(t0, y0, args)
    drift2 = drift_term2.vf(t0, y0, args)
    drift = jtu.tree_map(operator.sub, drift1, drift2)

    diffusion = diffusion_term.vf(t0, y0, args)  # assumes same diffusion

    diffusion = jtu.tree_map(
        lambda x: jnp.where(
            jax.lax.stop_gradient(x) > 1e-7,
            x,
            jnp.full_like(x, fill_value=1e-7) * jnp.sign(x),
        ),
        diffusion,
    )

    drift_tree_structure = jtu.tree_structure(drift)
    diffusion_tree_structure = jtu.tree_structure(diffusion)

    if drift_tree_structure == diffusion_tree_structure:
        if isinstance(diffusion_term, WeaklyDiagonalControlTerm):
            diffusion_linear_operator = jtu.tree_map(
                lx.DiagonalLinearOperator, diffusion
            )
        else:
            diffusion_linear_operator = jtu.tree_map(lx.MatrixLinearOperator, diffusion)

        divergences = jtu.tree_map(
            lambda a, b: lx.linear_solve(a, b, solver=linear_solver).value,
            diffusion_linear_operator,
            drift,
            is_leaf=lambda x: eqx.is_array(x)
            or isinstance(x, lx.AbstractLinearOperator),
        )
        kl_divergence = jtu.tree_map(lambda x: 0.5 * jnp.sum(x**2), divergences)
        kl_divergence = jtu.tree_reduce(operator.add, kl_divergence)

    else:
        raise ValueError(
            "drift and diffusion should have the same PyTree structure"
            + f" \n {drift_tree_structure} != {diffusion_tree_structure}"
        )
    return drift1, jnp.squeeze(kl_divergence)


class _KLDrift(AbstractTerm):
    drift1: ODETerm
    drift2: ODETerm
    diffusion: _ControlTerm
    linear_solver: lx.AbstractLinearSolver

    def vf(self, t: RealScalarLike, y: Y, args: Args) -> Tuple[VF, RealScalarLike]:
        y, _ = y
        return _compute_kl_integral(
            self.drift1, self.drift2, self.diffusion, t, y, args, self.linear_solver
        )

    def contr(self, t0: RealScalarLike, t1: RealScalarLike, **kwargs) -> Control:
        return t1 - t0

    def prod(self, vf: VF, control: RealScalarLike) -> Y:
        return jtu.tree_map(lambda v: control * v, vf)


class _KLControlTerm(AbstractTerm):
    control_term: _ControlTerm

    def vf(self, t: RealScalarLike, y: Y, args: Args) -> Tuple[VF, RealScalarLike]:
        y, _ = y
        vf = self.control_term.vf(t, y, args)
        return vf, 0.0

    def contr(
        self, t0: RealScalarLike, t1: RealScalarLike, **kwargs
    ) -> Tuple[Control, RealScalarLike]:
        return self.control_term.contr(t0, t1), 0.0

    def vf_prod(self, t: RealScalarLike, y: Y, args: Args, control: Control) -> Y:
        y, _ = y
        control, _ = control
        return self.control_term.vf_prod(t, y, args, control), 0.0

    def prod(self, vf: VF, control: Control) -> Y:
        vf, _ = vf
        control, _ = control
        return self.control_term.prod(vf, control), 0.0


class KLSolver(AbstractWrappedSolver[_SolverState]):
    r"""Given an SDE of the form

    $$
    \mathrm{d}y(t) = f_\theta (t, y(t)) dt + g_\phi (t, y(t)) dW(t) \qquad \zeta_\theta (ts[0]) = y_0
    $$

    $$
    \mathrm{d}z(t) = h_\psi (t, z(t)) dt + g_\phi (t, z(t)) dW(t) \qquad \nu_\psi (ts[0]) = z_0
    $$

    compute:

    $$
    \int_{ts[i-1]}^{ts[i]} g_\phi (t, y(t))^{-1} (f_\theta (t, y(y)) - h_\psi (t, y(t))) dt
    $$

    for every time interval. This is useful for KL based latent SDEs. The output
    of the solution.ys will be a tuple containing (ys, kls) where kls is the KL
    divergence integration at that time. Unless the noise is diagonal, this
    inverse can be extremely costly for higher dimenions.

    The input must be a `MultiTerm` composed of the first SDE with drift `f`
    and diffusion `g` and the second either a SDE or just the drift term
    (since the diffusion is assumed to be the same). For example, a type
    of: `MuliTerm(MultiTerm(ODETerm, _ControlTerm), ODETerm)`.

    ??? cite "References"

        See section 5 of:

        ```bibtex
        @inproceedings{li2020scalable,
            title={Scalable gradients for stochastic differential equations},
            author={Li, Xuechen and Wong, Ting-Kam Leonard and Chen, Ricky TQ and Duvenaud, David},
            booktitle={International Conference on Artificial Intelligence and Statistics},
            pages={3870--3882},
            year={2020},
            organization={PMLR}
        }
        ```

        Or section 4.3.2 of:

        ```bibtex
        @article{kidger2022neural,
            title={On neural differential equations},
            author={Kidger, Patrick},
            journal={arXiv preprint arXiv:2202.02435},
            year={2022}
        }
        ```
    """  # noqa: E501

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
        drift_term1, drift_term2 = drift_term1[0], drift_term2[0]

        diffusion_term = jtu.tree_map(
            lambda x: x if isinstance(x, _ControlTerm) else None,
            terms1,
            is_leaf=lambda x: isinstance(x, _ControlTerm),
        )
        diffusion_term = jtu.tree_leaves(
            diffusion_term, is_leaf=lambda x: isinstance(x, _ControlTerm)
        )

        diffusion_term = eqx.error_if(
            diffusion_term, len(diffusion_term) != 1, "SDE has multiple control terms!"
        )
        diffusion_term = diffusion_term[0]
        kl_terms = MultiTerm(
            _KLDrift(drift_term1, drift_term2, diffusion_term, self.linear_solver),
            _KLControlTerm(diffusion_term),
        )
        y1, y_error, dense_info, solver_state, result = self.solver.step(
            kl_terms, t0, t1, y0, args, solver_state, made_jump
        )
        return y1, y_error, dense_info, solver_state, result

    def func(
        self, terms: PyTree[AbstractTerm], t0: RealScalarLike, y0: Y, args: Args
    ) -> VF:
        return self.solver.func(terms, t0, y0, args)


KLSolver.__init__.__doc__ = """**Arguments:**

- `solver`: The solver to wrap.
- `linear_solver`: The lineax solver to use when computing $g^{-1}f$.
"""
