import jax
import jax.numpy as jnp
import numpy as np

from .custom_types import Array, PyTree


def _stack_pytrees(*arrays):
    return jnp.stack(arrays)


def stack_pytrees(pytrees: list[PyTree]) -> PyTree:
    return jax.tree_map(_stack_pytrees, *pytrees)


def safe_concatenate(arrays: list[Array]) -> Array:
    if len(arrays) == 0:
        return jnp.array([])
    return jnp.concatenate(arrays)


class frozenndarray:
    def __init__(self, *, array, **kwargs):
        super().__init__(**kwargs)
        array.flags.writeable = False
        _hash = hash(array.data.tobytes())
        self._array = array
        self._hash = _hash

    def __repr__(self):
        return f"{type(self).__name__}(array={self._array})"

    def __array__(self):
        return self._array

    def __hash__(self):
        return self._hash


def frozenarray(*args, **kwargs):
    return frozenndarray(array=np.array(*args, **kwargs))
