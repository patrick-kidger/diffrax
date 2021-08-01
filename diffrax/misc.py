import jax
import jax.numpy as jnp
import math
import numpy as np
from typing import Tuple

from .custom_types import Array, PyTree, SquashTreeDef


def _stack_pytrees(*arrays):
    return jnp.stack(arrays)


def stack_pytrees(pytrees: list[PyTree]) -> PyTree:
    return jax.tree_map(_stack_pytrees, *pytrees)


def safe_concatenate(arrays: list[Array]) -> Array:
    if len(arrays) == 0:
        return jnp.array([])
    return jnp.concatenate(arrays)


def tree_squash(tree: PyTree) -> Tuple[Array, SquashTreeDef]:
    flat, treedef = jax.tree_flatten(tree)
    if len(flat) == 1:
        # Optimised no-copying case
        shapes = None
        splits = None
        treedef = (treedef, shapes, splits)
        flat = jnp.asarray(flat[0])
    else:
        shapes = tuple(flat_i.shape for flat_i in flat)
        splits = tuple(np.array([math.prod(shape) for shape in shapes[:-1]]).cumsum())
        flat = [flat_i.flatten() for flat_i in flat]
        flat = safe_concatenate(flat)
        treedef = SquashTreeDef(treedef, shapes, splits)
    return flat, treedef


def tree_unsquash(treedef: SquashTreeDef, flat: Array) -> PyTree:
    treedef, shapes, splits = treedef
    if shapes is None:
        flat = [flat]
    else:
        flat = jnp.split(flat, splits)
        flat = [flat_i.reshape(shape) for flat_i, shape in zip(flat, shapes)]
    return jax.tree_unflatten(treedef, flat)


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
        assert '_reverse_lookup' not in dict
        dict['_reverse_lookup'] = {value: key for key, value in dict.items()}
        # Check that all values are unique
        assert len(dict) == len(dict['_reverse_lookup']) + 1  # +1 for dict['_reverse_lookup'] itself.
        return super().__new__(cls, name, bases, dict)

    def __getitem__(cls, item):
        return cls._reverse_lookup[item]


def vmap_all(x):
    while hasattr(x, '_trace') and isinstance(x._trace, jax.interpreters.batching.BatchTrace):
        x = x.val
    return jnp.all(x)


def vmap_any(x):
    while hasattr(x, '_trace') and isinstance(x._trace, jax.interpreters.batching.BatchTrace):
        x = x.val
    return jnp.any(x)


def vmap_max(x):
    while hasattr(x, '_trace') and isinstance(x._trace, jax.interpreters.batching.BatchTrace):
        x = x.val
    return jnp.max(x)
