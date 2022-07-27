import operator

import jax
import jax.numpy as jnp

from ..custom_types import PyTree, Scalar
from ..term import ControlTerm, MultiTerm, ODETerm, WeaklyDiagonalControlTerm


def _kl(drift1, drift2, diffusion):
    inv_diffusion = jnp.linalg.pinv(diffusion)
    scale = inv_diffusion @ (drift1 - drift2)
    return 0.5 * jnp.sum(scale**2)


def _kl_diagonal(drift1, drift2, diffusion):
    # stable division
    diffusion = jnp.where(
        jax.lax.stop_gradient(diffusion) > 1e-7,
        diffusion,
        jnp.full_like(diffusion, fill_value=1e-7) * jnp.sign(diffusion),
    )
    scale = (drift1 - drift2) / diffusion
    return 0.5 * jnp.sum(scale**2)


class _AugControlTerm(ControlTerm):

    control_term: ControlTerm

    def __init__(self, control_term: ControlTerm) -> None:
        super().__init__(control_term.vector_field, control_term.control)
        self.control_term = control_term

    def vf(self, t: Scalar, y: PyTree, args: PyTree) -> PyTree:
        y, _ = y
        vf = self.control_term.vf(t, y, args)
        return vf, 0.0

    def contr(self, t0: Scalar, t1: Scalar) -> PyTree:
        return self.control_term.contr(t0, t1), 0.0

    def vf_prod(self, t: Scalar, y: PyTree, args: PyTree, control: PyTree) -> PyTree:
        y, _ = y
        return self.control_term.vf_prod(t, y, args, control), 0.0


class _AugVectorField(ODETerm):

    sde1: MultiTerm
    sde2: MultiTerm
    context: callable
    kl: callable

    def __init__(self, sde1, sde2, context) -> None:
        super().__init__(sde1.terms[0].vector_field)
        if sde1.terms[1] is not sde2.terms[1]:
            raise ValueError("Two SDEs should share the same control terms")
        self.sde1 = sde1
        self.sde2 = sde2
        if isinstance(self.sde1.terms[1], WeaklyDiagonalControlTerm):
            self.kl = _kl_diagonal
        else:
            self.kl = _kl
        self.context = context

    def vf(self, t: Scalar, y: PyTree, args: PyTree) -> PyTree:
        y, _ = y
        context = self.context(t)
        aug_y = y if context is None else jnp.concatenate([y, context], axis=-1)
        drift1 = self.sde1.terms[0].vf(t, aug_y, args)
        drift2 = self.sde2.terms[0].vf(t, y, args)
        diffusion = self.sde1.terms[1].vf(t, y, args)
        kl_divergence = jax.tree_map(self.kl, drift1, drift2, diffusion)
        kl_divergence = jax.tree_util.tree_reduce(operator.add, kl_divergence)
        return drift1, kl_divergence


def sde_kl_divergence(
    *, sde1: MultiTerm, sde2: MultiTerm, context: callable, y0: PyTree
):
    if context is None:
        context = lambda t: None
    aug_y0 = (y0, 0.0)
    aug_drift = _AugVectorField(sde1, sde2, context=context)
    aug_control = _AugControlTerm(sde1.terms[1])
    aug_sde = MultiTerm(aug_drift, aug_control)

    return aug_sde, aug_y0
