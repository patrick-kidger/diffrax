import jax
import jax.numpy as jnp


def vmap_any(x):
    while hasattr(x, '_trace') and isinstance(x._trace, jax.interpreters.batching.BatchTrace):
        x = x.val
    return jnp.any(x)
