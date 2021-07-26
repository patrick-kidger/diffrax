import jax
import jax.numpy as jnp

from .custom_types import PyTree


def _stack_pytrees(*arrays):
    return jnp.stack(arrays)


def stack_pytrees(pytrees: list[PyTree]) -> PyTree:
    return jax.tree_map(_stack_pytrees, *pytrees)
