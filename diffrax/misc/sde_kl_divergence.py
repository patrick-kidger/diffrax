import operator

import equinox as eqx
import jax
import jax.numpy as jnp

from ..brownian import AbstractBrownianPath
from ..custom_types import PyTree


def _kl(drift1, drift2, diffusion):
    inv_diffusion = jnp.linalg.pinv(diffusion)
    scale = inv_diffusion @ (drift1 - drift2)
    return 0.5 * jnp.sum(scale**2)


class _AugDrift(eqx.Module):
    drift1: callable
    drift2: callable
    diffusion: callable
    context: callable

    def __call__(self, t, y, args):
        y, _ = y
        context = self.context(t)
        aug_y = jnp.concatenate([y, context], axis=-1)
        drift1 = self.drift1(t, aug_y, args)
        drift2 = self.drift2(t, y, args)
        diffusion = self.diffusion(t, y, args)
        kl_divergence = jax.tree_map(_kl, drift1, drift2, diffusion)
        kl_divergence = jax.tree_util.tree_reduce(operator.add, kl_divergence)
        return drift1, kl_divergence


class _AugDiffusion(eqx.Module):
    diffusion: callable

    def __call__(self, t, y, args):
        y, _ = y
        diffusion = self.diffusion(t, y, args)
        return diffusion, 0.0


class _AugBrownianPath(eqx.Module):
    bm: AbstractBrownianPath

    @property
    def t0(self):
        return self.bm.t0

    @property
    def t1(self):
        return self.bm.t1

    def evaluate(self, t0, t1):
        return self.bm.evaluate(t0, t1), 0.0


def sde_kl_divergence(
    *,
    drift1: callable,
    drift2: callable,
    diffusion: callable,
    context: callable,
    y0: PyTree,
    bm: AbstractBrownianPath,
):
    aug_y0 = (y0, 0.0)
    return (
        _AugDrift(drift1, drift2, diffusion, context),
        _AugDiffusion(diffusion),
        aug_y0,
        _AugBrownianPath(bm),
    )
