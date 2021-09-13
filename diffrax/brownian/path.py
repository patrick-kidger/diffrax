from typing import Tuple

import equinox as eqx
import jax.lax as lax
import jax.numpy as jnp
import jax.random as jrandom

from .base import AbstractBrownianPath


class UnsafeBrownianPath(AbstractBrownianPath):
    key: jrandom.PRNGKey
    shape: Tuple[int] = eqx.static_field()

    @property
    def t0(self):
        return None

    @property
    def t1(self):
        return None

    def evaluate(self, t0, t1):
        t0_ = lax.bitcast_convert_type(t0, jnp.int32)
        t1_ = lax.bitcast_convert_type(t1, jnp.int32)
        key = jrandom.fold_in(self.key, t0_)
        key = jrandom.fold_in(key, t1_)
        return jrandom.normal(key, self.shape) * jnp.sqrt(t1 - t0)
