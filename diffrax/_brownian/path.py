import math
from typing import cast, Optional, Union

import equinox as eqx
import equinox.internal as eqxi
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
import lineax.internal as lxi
from jaxtyping import Array, PRNGKeyArray, PyTree

from .._custom_types import levy_tree_transpose, LevyArea, LevyVal, RealScalarLike
from .._misc import (
    force_bitcast_convert_type,
    is_tuple_of_ints,
    split_by_tree,
)
from .base import AbstractBrownianPath


class UnsafeBrownianPath(AbstractBrownianPath):
    """Brownian simulation that is only suitable for certain cases.

    This is a very quick way to simulate Brownian motion, but can only be used when all
    of the following are true:

    1. You are using a fixed step size controller. (Not an adaptive one.)

    2. You do not need to backpropagate through the differential equation.

    3. You do not need deterministic solutions with respect to `key`. (This
       implementation will produce different results based on fluctuations in
       floating-point arithmetic.)

    Internally this operates by just sampling a fresh normal random variable over every
    interval, ignoring the correlation between samples exhibited in true Brownian
    motion. Hence the restrictions above. (They describe the general case for which the
    correlation structure isn't needed.)

    Depending on the `levy_area` argument, this can also be used to generate Levy area.
    """

    shape: PyTree[jax.ShapeDtypeStruct] = eqx.field(static=True)
    levy_area: LevyArea = eqx.field(static=True)
    key: PRNGKeyArray

    def __init__(
        self,
        shape: Union[tuple[int, ...], PyTree[jax.ShapeDtypeStruct]],
        key: PRNGKeyArray,
        levy_area: LevyArea = "",
    ):
        self.shape = (
            jax.ShapeDtypeStruct(shape, lxi.default_floating_dtype())
            if is_tuple_of_ints(shape)
            else shape
        )
        self.key = key
        if levy_area not in ["", "space-time"]:
            raise ValueError(
                f"levy_area must be one of '', 'space-time', but got {levy_area}."
            )
        self.levy_area = levy_area

        if any(
            not jnp.issubdtype(x.dtype, jnp.inexact)
            for x in jtu.tree_leaves(self.shape)
        ):
            raise ValueError("UnsafeBrownianPath dtypes all have to be floating-point.")

    @property
    def t0(self):
        return -jnp.inf

    @property
    def t1(self):
        return jnp.inf

    @eqx.filter_jit
    def evaluate(
        self,
        t0: RealScalarLike,
        t1: Optional[RealScalarLike] = None,
        left: bool = True,
        use_levy: bool = False,
    ) -> Union[PyTree[Array], LevyVal]:
        del left
        if t1 is None:
            dtype = jnp.result_type(t0)
            t1 = t0
            t0 = jnp.array(0, dtype)
        else:
            with jax.numpy_dtype_promotion("standard"):
                dtype = jnp.result_type(t0, t1)
            t0 = jnp.astype(t0, dtype)
            t1 = jnp.astype(t1, dtype)
        t0 = eqxi.nondifferentiable(t0, name="t0")
        t1 = eqxi.nondifferentiable(t1, name="t1")
        t1 = cast(RealScalarLike, t1)
        t0_ = force_bitcast_convert_type(t0, jnp.int32)
        t1_ = force_bitcast_convert_type(t1, jnp.int32)
        key = jr.fold_in(self.key, t0_)
        key = jr.fold_in(key, t1_)
        key = split_by_tree(key, self.shape)
        out = jtu.tree_map(
            lambda key, shape: self._evaluate_leaf(
                t0, t1, key, shape, self.levy_area, use_levy
            ),
            key,
            self.shape,
        )
        if use_levy:
            out = levy_tree_transpose(self.shape, self.levy_area, out)
            assert isinstance(out, LevyVal)
        return out

    @staticmethod
    def _evaluate_leaf(
        t0: RealScalarLike,
        t1: RealScalarLike,
        key,
        shape: jax.ShapeDtypeStruct,
        levy_area: str,
        use_levy: bool,
    ):
        w_std = jnp.sqrt(t1 - t0).astype(shape.dtype)

        if levy_area == "space-time":
            key, key_hh = jr.split(key, 2)
            hh_std = w_std / math.sqrt(12)
            hh = jr.normal(key_hh, shape.shape, shape.dtype) * hh_std
        elif levy_area == "":
            hh = None
        else:
            assert False
        w = jr.normal(key, shape.shape, shape.dtype) * w_std

        if use_levy:
            return LevyVal(dt=t1 - t0, W=w, H=hh, bar_H=None, K=None, bar_K=None)
        else:
            return w


UnsafeBrownianPath.__init__.__doc__ = """
**Arguments:**

- `shape`: Should be a PyTree of `jax.ShapeDtypeStruct`s, representing the shape, 
    dtype, and PyTree structure of the output. For simplicity, `shape` can also just 
    be a tuple of integers, describing the shape of a single JAX array. In that case
    the dtype is chosen to be the default floating-point dtype.
- `key`: A random key.
- `levy_area`: Whether to additionally generate Levy area. This is required by some SDE
    solvers.
"""
