import typing
from typing import Any, TYPE_CHECKING, Union

import equinox.internal as eqxi
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

Y = PyTree[Float[ArrayLike, "?*y"], "Y"]
VF = PyTree[Float[ArrayLike, "?*vf"], "VF"]
Control = PyTree[Float[ArrayLike, "?*control"], "C"]
Args = PyTree[Any]

DenseInfo = dict[str, PyTree[Array]]
DenseInfos = dict[str, PyTree[Shaped[Array, "times ..."]]]
BufferDenseInfos = dict[str, PyTree[eqxi.MaybeBuffer[Shaped[Array, "times ..."]]]]
sentinel: Any = eqxi.doc_repr(object(), "sentinel")

del Array, ArrayLike, PyTree, Bool, Int, Shaped, Float, AbstractDtype
