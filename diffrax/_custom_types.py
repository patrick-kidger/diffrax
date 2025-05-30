from typing import Any, TYPE_CHECKING

import equinox as eqx
import equinox.internal as eqxi
import jax.tree_util as jtu
import numpy as np
from jaxtyping import (
    AbstractDtype,
    Array,
    ArrayLike,
    Bool,
    Float,
    Int,
    PyTree,
    Real,
    Shaped,
)


if TYPE_CHECKING:
    BoolScalarLike = bool | Array | np.ndarray
    FloatScalarLike = float | Array | np.ndarray
    IntScalarLike = int | Array | np.ndarray
    RealScalarLike = bool | int | float | Array | np.ndarray
else:
    BoolScalarLike = Bool[ArrayLike, ""]
    FloatScalarLike = Float[ArrayLike, ""]
    IntScalarLike = Int[ArrayLike, ""]
    RealScalarLike = Real[ArrayLike, ""]


Y = PyTree[Shaped[ArrayLike, "?*y"], "Y"]
VF = PyTree[Shaped[ArrayLike, "?*vf"], "VF"]
Control = PyTree[Shaped[ArrayLike, "?*control"], "C"]
Args = PyTree[Any]

BM = PyTree[Shaped[ArrayLike, "?*bm"], "BM"]

DenseInfo = dict[str, PyTree[Array]]
DenseInfos = dict[str, PyTree[Shaped[Array, "times-1 ..."]]]
BufferDenseInfos = dict[str, PyTree[eqxi.MaybeBuffer[Shaped[Array, "times ..."]]]]
sentinel: Any = eqxi.doc_repr(object(), "sentinel")


class AbstractBrownianIncrement(eqx.Module):
    """
    Abstract base class for all Brownian increments.
    """

    dt: eqx.AbstractVar[PyTree[FloatScalarLike, "BM"]]
    W: eqx.AbstractVar[BM]


class AbstractSpaceTimeLevyArea(AbstractBrownianIncrement):
    """
    Abstract base class for all Space Time Levy Areas.
    """

    H: eqx.AbstractVar[BM]


class AbstractSpaceTimeTimeLevyArea(AbstractSpaceTimeLevyArea):
    """
    Abstract base class for all Space Time Time Levy Areas.
    """

    K: eqx.AbstractVar[BM]


class BrownianIncrement(AbstractBrownianIncrement):
    """
    Pytree containing the `dt` time increment and `W` the Brownian motion.
    """

    dt: PyTree[FloatScalarLike, "BM"]
    W: BM


class SpaceTimeLevyArea(AbstractSpaceTimeLevyArea):
    """
    Pytree containing the `dt` time increment, `W` the Brownian motion, and `H`
    the Space Time Levy Area.
    """

    dt: PyTree[FloatScalarLike, "BM"]
    W: BM
    H: BM


class SpaceTimeTimeLevyArea(AbstractSpaceTimeTimeLevyArea):
    """
    Pytree containing the `dt` time increment, `W` the Brownian motion, `H`
    the Space Time Levy Area, and `K` the Space Time Time Levy Area.
    """

    dt: PyTree[FloatScalarLike, "BM"]
    W: BM
    H: BM
    K: BM


AbstractBrownianIncrement.__module__ = "diffrax"
AbstractSpaceTimeLevyArea.__module__ = "diffrax"
AbstractSpaceTimeTimeLevyArea.__module__ = "diffrax"
BrownianIncrement.__module__ = "diffrax"
SpaceTimeLevyArea.__module__ = "diffrax"
SpaceTimeTimeLevyArea.__module__ = "diffrax"


def levy_tree_transpose(
    tree_shape, tree: PyTree[AbstractBrownianIncrement]
) -> AbstractBrownianIncrement:
    """Helper that takes a `PyTree `of `AbstractBrownianIncrement`s and transposes
    into an `AbstractBrownianIncrement` of `PyTree`s.

    **Arguments:**

    - `tree_shape`: Corresponds to `outer_treedef` in `jax.tree_transpose`.
    - `tree`: the `PyTree` of `AbstractBrownianIncrement`s to transpose.

    **Returns:**

    An `AbstractBrownianIncrement` of `PyTree`s.
    """
    inner_tree = jtu.tree_leaves(
        tree, is_leaf=lambda x: isinstance(x, AbstractBrownianIncrement)
    )[0]
    inner_tree_shape = jtu.tree_structure(inner_tree)
    return jtu.tree_transpose(
        outer_treedef=jtu.tree_structure(tree_shape),
        inner_treedef=inner_tree_shape,
        pytree_to_transpose=tree,
    )


del Array, ArrayLike, PyTree, Bool, Int, Shaped, Float, AbstractDtype
