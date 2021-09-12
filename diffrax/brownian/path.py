from typing import Tuple

import jax.lax as lax
import jax.numpy as jnp
import jax.random as jrandom

from .base import AbstractBrownianPath


class UnsafeBrownianPath(AbstractBrownianPath):
    key: jrandom.PRNGKey
    shape: Tuple[int]

    def evaluate(self, t0, t1):
        t0_ = lax.bitcast_convert_type(t0, jnp.int32)
        t1_ = lax.bitcast_convert_type(t1, jnp.int32)
        key = jrandom.fold_in(self.key, t0_)
        key = jrandom.fold_in(key, t1_)
        return jrandom.normal(key, self.shape) * (t1 - t0)
