import typing
from typing import Any, Optional, TYPE_CHECKING, Union

import equinox as eqx
import equinox.internal as eqxi
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np
from jaxtyping import AbstractDtype, Array, ArrayLike, Bool, Float, Int, PyTree, Shaped


if TYPE_CHECKING:
    from typing import Annotated as Real
else:

    class Real(AbstractDtype):
        dtypes = Float.dtypes + Int.dtypes  # pyright: ignore


if TYPE_CHECKING:
    BoolScalarLike = Union[bool, Array, np.ndarray]
    FloatScalarLike = Union[float, Array, np.ndarray]
    IntScalarLike = Union[int, Array, np.ndarray]
elif getattr(typing, "GENERATING_DOCUMENTATION", False):
    # Skip the union with Array in docs.
    BoolScalarLike = bool
    FloatScalarLike = float
    IntScalarLike = int

    #
    # Because they appear in our docstrings, we also monkey-patch some non-Diffrax
    # types that have similar defined-in-one-place, exported-in-another behaviour.
    #

    jtu.Partial.__module__ = "jax.tree_util"

else:
    FloatScalarLike = Float[ArrayLike, ""]
    IntScalarLike = Int[ArrayLike, ""]
    BoolScalarLike = Bool[ArrayLike, ""]


RealScalarLike = Union[FloatScalarLike, IntScalarLike]

Y = PyTree[Shaped[ArrayLike, "?*y"], "Y"]
VF = PyTree[Shaped[ArrayLike, "?*vf"], "VF"]
Control = PyTree[Shaped[ArrayLike, "?*control"], "C"]
Args = PyTree[Any]

DenseInfo = dict[str, PyTree[Array]]
DenseInfos = dict[str, PyTree[Shaped[Array, "times ..."]]]
BufferDenseInfos = dict[str, PyTree[eqxi.MaybeBuffer[Shaped[Array, "times ..."]]]]
sentinel: Any = eqxi.doc_repr(object(), "sentinel")


class LevyVal(eqx.Module):
    dt: PyTree
    W: PyTree
    H: Optional[PyTree]
    bar_H: Optional[PyTree]
    K: Optional[PyTree]
    bar_K: Optional[PyTree]


def levy_tree_transpose(tree_shape, levy_area, tree):
    """Helper that takes a PyTree of LevyVals and transposes
    into a LevyVal of PyTrees.

    **Arguments:**

    - `tree_shape`: Corresponds to `outer_treedef` in `jax.tree_transpose`.
    - `levy_area`: can be `""` or `"space-time"`, which indicates
    which fields of the LevyVal will have values.
    - `tree`: the PyTree of LevyVals to transpose.

    **Returns:**

    A `LevyVal` of PyTrees.
    """
    if levy_area == "space-time":
        hh_default_val = jnp.zeros(())
        kk_default_val = None
    elif levy_area == "":
        hh_default_val = None
        kk_default_val = None
    else:
        assert False
    return jtu.tree_transpose(
        outer_treedef=jtu.tree_structure(tree_shape),
        inner_treedef=jtu.tree_structure(
            LevyVal(
                dt=0.0,
                W=jnp.zeros(()),
                H=hh_default_val,
                bar_H=None,
                K=kk_default_val,
                bar_K=None,
            )
        ),
        pytree_to_transpose=tree,
    )


del Array, ArrayLike, PyTree, Bool, Int, Shaped, Float, AbstractDtype
