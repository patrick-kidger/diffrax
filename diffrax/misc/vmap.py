import jax
import jax.numpy as jnp


@jax.jit
def _jit_all(x):
    return jnp.all(x)


@jax.jit
def _jit_any(x):
    return jnp.any(x)


def vmap_all(x):
    while hasattr(x, "_trace") and isinstance(
        x._trace, jax.interpreters.batching.BatchTrace
    ):
        x = x.val
    return _jit_all(x)


def vmap_any(x):
    while hasattr(x, "_trace") and isinstance(
        x._trace, jax.interpreters.batching.BatchTrace
    ):
        x = x.val
    return _jit_any(x)
