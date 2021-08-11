from typing import List

import jax
import jax.numpy as jnp
import numpy as np

from .custom_types import Array, PyTree


def _stack_pytrees(*arrays):
    return jnp.stack(arrays)


def stack_pytrees(pytrees: List[PyTree]) -> PyTree:
    return jax.tree_map(_stack_pytrees, *pytrees)


def safe_concatenate(arrays: List[Array]) -> Array:
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


class ContainerMeta(type):
    def __new__(cls, name, bases, dict):
        assert "_reverse_lookup" not in dict
        dict["_reverse_lookup"] = {value: key for key, value in dict.items()}
        # Check that all values are unique
        assert (
            len(dict) == len(dict["_reverse_lookup"]) + 1
        )  # +1 for dict['_reverse_lookup'] itself.
        return super().__new__(cls, name, bases, dict)

    def __getitem__(cls, item):
        return cls._reverse_lookup[item]


def vmap_all(x):
    while hasattr(x, "_trace") and isinstance(
        x._trace, jax.interpreters.batching.BatchTrace
    ):
        x = x.val
    return jnp.all(x)


def vmap_any(x):
    while hasattr(x, "_trace") and isinstance(
        x._trace, jax.interpreters.batching.BatchTrace
    ):
        x = x.val
    return jnp.any(x)
