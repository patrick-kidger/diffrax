import warnings
from typing import List, Tuple

import jax
import jax.lax as lax
import jax.numpy as jnp
import numpy as np

from ..custom_types import Array, PyTree
from .frozenarray import frozenarray, frozenndarray
from .refholder import RefHolder


def _empty_unravel_list(_):
    return []


def _unravel_list(
    arr: Array["flat"],  # noqa: F821
    indices: RefHolder[frozenndarray],
    shapes: RefHolder[List[Tuple[int, ...]]],
    from_dtypes: RefHolder[List[jnp.dtype]],
) -> List[Array]:
    indices = np.asarray(indices.value)
    shapes = shapes.value
    from_dtypes = from_dtypes.value
    chunks = jnp.split(arr, indices[:-1])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # ignore complex-to-real cast warning
        return [
            lax.convert_element_type(chunk.reshape(shape), dtype)
            for chunk, shape, dtype in zip(chunks, shapes, from_dtypes)
        ]


def _ravel_list(
    leaves: List[Array],
) -> Tuple[Array["flat"], jax.tree_util.Partial]:  # noqa: F821
    if not leaves:
        return jnp.array([]), jax.tree_util.Partial(_empty_unravel_list)
    from_dtypes = [jnp.result_type(leaf) for leaf in leaves]
    to_dtype = jnp.result_type(*from_dtypes)
    sizes = [jnp.size(leaf) for leaf in leaves]
    ravel = lambda leaf: jnp.ravel(lax.convert_element_type(leaf, to_dtype))
    raveled = jnp.concatenate([ravel(leaf) for leaf in leaves])
    indices = RefHolder(frozenarray(np.cumsum(sizes)))
    shapes = RefHolder([jnp.shape(leaf) for leaf in leaves])
    from_dtypes = RefHolder(from_dtypes)
    return raveled, jax.tree_util.Partial(
        _unravel_list, indices=indices, shapes=shapes, from_dtypes=from_dtypes
    )


@jax.jit
def _unravel_pytree(
    flat: Array["flat"],  # noqa: F821
    treedef: RefHolder,
    unravel_list: jax.tree_util.Partial,
) -> PyTree:
    return jax.tree_unflatten(treedef.value, unravel_list(flat))


@jax.jit
def ravel_pytree(
    pytree: PyTree,
) -> Tuple[Array["flat"], jax.tree_util.Partial]:  # noqa: F821
    """Like jax.flatten_util.ravel_pytree, but doesn't create a new unravel function
    each time. This means the unravel function can be passed to JIT-compiled functions
    without triggering recompilation, if pytree has the same structure.
    """

    leaves, treedef = jax.tree_flatten(pytree)
    flat, unravel_list = _ravel_list(leaves)
    treedef = RefHolder(treedef)
    return flat, jax.tree_util.Partial(
        _unravel_pytree, treedef=treedef, unravel_list=unravel_list
    )
