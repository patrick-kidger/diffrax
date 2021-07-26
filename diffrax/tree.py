import jax
import jax.numpy as jnp
import math
import numpy as np
from typing import Tuple

from .custom_types import Array, PyTree, SquashTreeDef
from .misc import safe_concatenate


def tree_squash(tree: PyTree) -> Tuple[Array, SquashTreeDef]:
    flat, treedef = jax.tree_flatten(tree)
    if len(flat) == 1:
        # Optimised no-copying case
        shapes = None
        splits = None
        treedef = (treedef, shapes, splits)
        flat = flat[0]
    else:
        shapes = [flat_i.shape for flat_i in flat]
        splits = np.array([math.prod(shape) for shape in shapes[:-1]]).cumsum()
        flat = [flat_i.flatten() for flat_i in flat]
        flat = safe_concatenate(flat)
        treedef = (treedef, shapes, splits)
    return flat, treedef


def tree_unsquash(treedef: SquashTreeDef, flat: Array) -> PyTree:
    treedef, shapes, splits = treedef
    if shapes is None:
        flat = [flat]
    else:
        flat = jnp.split(flat, splits)
        flat = [flat_i.reshape(shape) for flat_i, shape in zip(flat, shapes)]
    return jax.tree_unflatten(treedef, flat)
