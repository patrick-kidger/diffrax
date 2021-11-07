import warnings
from typing import Generic, List, Tuple, TypeVar

import equinox as eqx
import jax
import jax.lax as lax
import jax.numpy as jnp
import numpy as np

from ..custom_types import Array, PyTree
from .frozenarray import frozenarray, frozenndarray


_T = TypeVar("_T")


class _Static(eqx.Module, Generic[_T]):
    value: _T = eqx.static_field()


def _empty_unravel_list(_):
    return []


def _unravel_list(
    arr: Array["flat"],  # noqa: F821
    indices: _Static[frozenndarray],
    shapes: _Static[List[Tuple[int, ...]]],
    from_dtypes: _Static[List[jnp.dtype]],
) -> List[Array]:
    indices = np.asarray(indices.value)
    shapes = shapes.value
    from_dtypes = from_dtypes.value
    chunks = jnp.split(arr, indices[:-1])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # ignore complex-to-real cast warning
        return [
            lax.convert_element_type(
                chunk.reshape(shape), jnp.result_type(dtype, arr.dtype)
            )
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
    indices = _Static(frozenarray(np.cumsum(sizes)))
    shapes = _Static([jnp.shape(leaf) for leaf in leaves])
    from_dtypes = _Static(from_dtypes)
    return raveled, jax.tree_util.Partial(
        _unravel_list, indices=indices, shapes=shapes, from_dtypes=from_dtypes
    )


@jax.jit
def _unravel_pytree(
    flat: Array["flat"],  # noqa: F821
    treedef: _Static,
    unravel_list: jax.tree_util.Partial,
) -> PyTree:
    return jax.tree_unflatten(treedef.value, unravel_list(flat))


@jax.jit
def ravel_pytree(
    pytree: PyTree,
) -> Tuple[Array["flat"], jax.tree_util.Partial]:  # noqa: F821
    """Like `jax.flatten_util.ravel_pytree`, but doesn't create a new unravel function
    each time. This means the unravel function can be passed to JIT-compiled functions
    without triggering recompilation, if `pytree` has the same structure.

    In addition, unravelling will consider the dtype of the argment to be unravelled.
    In particular this means that if the object to be raveled consists of all-integers,
    and the object to unraveled has a floating-point dtype, then it will be unraveled
    with floating point dtype.

    **Arguments:**

    - `pytree`: Some PyTree with JAX arrays on the leaves.

    **Returns:**

    A 2-tuple. The first element is a single-dimensional JAX array, featuring all of
    the leaves of the input `pytree` flattened and concatenated. The second element is
    a function that can be used to reconstitute an array of the same shape (but
    possibly different dtype) into a PyTree with the same structure, and leaves of the
    same shape, as the original `pytree` input.
    """

    leaves, treedef = jax.tree_flatten(pytree)
    flat, unravel_list = _ravel_list(leaves)
    treedef = _Static(treedef)
    return flat, jax.tree_util.Partial(
        _unravel_pytree, treedef=treedef, unravel_list=unravel_list
    )
