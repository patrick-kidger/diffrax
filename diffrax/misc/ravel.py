import warnings
from typing import List, Tuple

import jax
import jax.lax as lax
import jax.numpy as jnp
import numpy as np

from ..custom_types import Array, PyTree
from .refholder import RefHolder


def _empty_unravel_list(_):
    return []


def _unravel_list(
    arr: Array["flat"],  # noqa: F821
    indices: RefHolder[np.ndarray],
    shapes: RefHolder[List[Tuple[int, ...]]],
    from_dtypes: List[jnp.dtype],
) -> List[Array]:
    indices = indices.value
    shapes = shapes.value
    chunks = jnp.split(arr, indices[:-1])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # ignore complex-to-real cast warning
        return [
            lax.convert_element_type(chunk.reshape(shape), dtype)
            for chunk, shape, dtype in zip(chunks, shapes, from_dtypes)
        ]


def _ravel_list(leaves: List[Array]) -> Tuple[Array["flat"], callable]:  # noqa: F821
    if not leaves:
        return jnp.array([], jnp.float32), _empty_unravel_list
    from_dtypes = [jnp.result_type(leaf) for leaf in leaves]
    to_dtype = jnp.result_type(*from_dtypes)
    sizes = [x.size for x in leaves]
    shapes = RefHolder([x.shape for x in leaves])
    indices = RefHolder(np.cumsum(sizes))
    ravel = lambda e: jnp.ravel(lax.convert_element_type(e, to_dtype))
    raveled = jnp.concatenate([ravel(e) for e in leaves])
    return raveled, jax.tree_util.Partial(
        _unravel_list, indices=indices, shapes=shapes, from_dtypes=from_dtypes
    )


def _unravel_pytree(
    flat: Array["flat"], treedef, unravel_list: callable  # noqa: F821
) -> PyTree:
    return jax.tree_unflatten(treedef, unravel_list(flat))


def ravel_pytree(pytree: PyTree) -> Tuple[Array["flat"], callable]:  # noqa: F821
    """Like jax.flatten_util.ravel_pytree, but doesn't create a new unravel function
    each time. This means the unravel function can be passed to JIT-compiled functions
    without triggering recompilation, if pytree has the same structure.
    """

    leaves, treedef = jax.tree_flatten(pytree)
    flat, unravel_list = _ravel_list(leaves)
    return flat, jax.tree_util.Partial(
        _unravel_pytree, treedef=treedef, unravel_list=unravel_list
    )
