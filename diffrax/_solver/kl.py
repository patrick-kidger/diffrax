import operator
from typing import Optional

import equinox as eqx
import jax.tree_util as jtu
import lineax as lx
from jax import numpy as jnp
from jaxtyping import Array, PyTree

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
)
from .base import (
    _SolverState,
    AbstractSolver,
    AbstractWrappedSolver,
)


class KLState(eqx.Module):
    """
    The state of the SDE and the KL divergence.
    """

    y: Y
    kl_metric: Array


def _compute_kl_integral(
    drift_term1: ODETerm,
    drift_term2: ODETerm,
    diffusion_term: ControlTerm,
    t0: RealScalarLike,
    y0: Y,
    args: Args,
    linear_solver: lx.AbstractLinearSolver,
) -> KLState:
    """
    Compute the KL divergence.
    """
    drift1 = drift_term1.vf(t0, y0, args)
    drift2 = drift_term2.vf(t0, y0, args)
    drift = jtu.tree_map(operator.sub, drift1, drift2)

    diffusion = diffusion_term.vf(t0, y0, args)  # assumes same diffusion

    if not isinstance(diffusion, lx.AbstractLinearOperator):
        diffusion = lx.MatrixLinearOperator(diffusion)

    divergences = lx.linear_solve(diffusion, drift, solver=linear_solver).value

    kl_divergence = jtu.tree_map(lambda x: 0.5 * jnp.sum(x**2), divergences)
    kl_divergence = jtu.tree_reduce(operator.add, kl_divergence)

    return KLState(drift1, kl_divergence)


class _KLDrift(AbstractTerm):
    drift1: ODETerm
    drift2: ODETerm
    diffusion: ControlTerm
    linear_solver: lx.AbstractLinearSolver

    def vf(self, t: RealScalarLike, y: KLState, args: Args) -> KLState:
        y = y.y
        return _compute_kl_integral(
            self.drift1, self.drift2, self.diffusion, t, y, args, self.linear_solver
        )

    def contr(self, t0: RealScalarLike, t1: RealScalarLike, **kwargs) -> Control:
        return t1 - t0

    def prod(self, vf: VF, control: RealScalarLike) -> Y:
        return jtu.tree_map(lambda v: control * v, vf)


class _KLControlTerm(AbstractTerm):
    control_term: ControlTerm

    def vf(self, t: RealScalarLike, y: Y, args: Args) -> KLState:
        y = y.y
        vf = self.control_term.vf(t, y, args)
        return KLState(vf, jnp.array(0.0))

    def contr(self, t0: RealScalarLike, t1: RealScalarLike, **kwargs) -> Control:
        return self.control_term.contr(t0, t1), jnp.array(0.0)

    def vf_prod(
        self, t: RealScalarLike, y: KLState, args: Args, control: Control
    ) -> KLState:
        y = y.y
        return KLState(self.control_term.vf_prod(t, y, args, control), jnp.array(0.0))

    def prod(self, vf: KLState, control: Control) -> KLState:
        vf = vf.y
        return KLState(self.control_term.prod(vf, control), jnp.array(0.0))


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
    of: `MuliTerm(MultiTerm(ODETerm, _DiffusionTerm), ODETerm)`.

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
        y0: KLState,
        args: Args,
        solver_state: _SolverState,
        made_jump: BoolScalarLike,
    ) -> tuple[KLState, Optional[Y], DenseInfo, _SolverState, RESULTS]:
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

        drift_term1, drift_term2 = drift_term1[0], drift_term2[0]

        diffusion_term = jtu.tree_map(
            lambda x: x if isinstance(x, ControlTerm) else None,
            terms1,
            is_leaf=lambda x: isinstance(x, ControlTerm),
        )
        diffusion_term = jtu.tree_leaves(
            diffusion_term,
            is_leaf=lambda x: isinstance(x, ControlTerm),
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


def initialize_kl(
    solver: AbstractSolver,
    y0: Y,
    linear_solver: lx.AbstractLinearSolver = lx.AutoLinearSolver(well_posed=None),
) -> tuple[KLSolver, KLState]:
    """
    Initialize the KL solver and state.


    **Arguments**

    - `solver`: the method for solving the SDE.
    - `y0`: the initial state
    - `linear_solver`: the method for computing $g^{-1}f$.

    **Returns**

    A `KLState` containing the `KLSolver` and the new initial state. Both of
    these can be directly fed into `diffeqsolve`.

    """
    return KLSolver(solver, linear_solver), KLState(y=y0, kl_metric=jnp.array(0.0))
