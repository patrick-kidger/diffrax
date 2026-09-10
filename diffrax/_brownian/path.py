import math
from typing import cast

import equinox as eqx
import equinox.internal as eqxi
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
import lineax.internal as lxi
from jaxtyping import Array, PRNGKeyArray, PyTree, Int, Float
from lineax.internal import complex_to_real_dtype

from .._custom_types import (
    AbstractBrownianIncrement,
    BrownianIncrement,
    levy_tree_transpose,
    RealScalarLike,
    SpaceTimeLevyArea,
    SpaceTimeTimeLevyArea,
)
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

    !!! info "Lévy Area"

        Can be initialised with `levy_area` set to `diffrax.BrownianIncrement`, or
        `diffrax.SpaceTimeLevyArea`. If `levy_area=diffrax.SpaceTimeLevyArea`, then it
        also computes space-time Lévy area `H`. This is an additional source of
        randomness required for certain stochastic Runge--Kutta solvers; see
        [`diffrax.AbstractSRK`][] for more information.

        An error will be thrown during tracing if Lévy area is required but is not
        available.

        The choice here will impact the Brownian path, so even with the same key, the
        trajectory will be different depending on the value of `levy_area`.
    """

    shape: PyTree[jax.ShapeDtypeStruct] = eqx.field(static=True)
    levy_area: type[BrownianIncrement | SpaceTimeLevyArea | SpaceTimeTimeLevyArea] = (
        eqx.field(static=True)
    )
    key: PRNGKeyArray
    arr: PyTree[Float[Array, " steps"]] | None

    def __init__(
        self,
        shape: tuple[int, ...] | PyTree[jax.ShapeDtypeStruct],
        key: PRNGKeyArray,
        levy_area: type[
            BrownianIncrement | SpaceTimeLevyArea | SpaceTimeTimeLevyArea
        ] = BrownianIncrement,
        precompute: int | None = None
    ):
        """**Arguments:**

        - `shape`: Should be a PyTree of `jax.ShapeDtypeStruct`s, representing the
            shape, dtype, and PyTree structure of the output. For simplicity, `shape`
            can also just be a tuple of integers, describing the shape of a single JAX
            array. In that case the dtype is chosen to be the default floating-point
            dtype.
        - `key`: A JAX random key, as from `jax.random.key(seed)`.
        - `levy_area`: Whether to additionally generate Lévy area. This is required by
            some SDE solvers.
        """
        self.shape = (
            jax.ShapeDtypeStruct(shape, lxi.default_floating_dtype())
            if is_tuple_of_ints(shape)
            else shape
        )
        key, subkey = jax.random.split(key)
        self.key = key
        self.levy_area = levy_area
        # seems bad to define precompute and max_steps, should be the same I imagine? make eqx.tree_at in integrate? Idk
        if precompute is None:
            self.arr = None
        else:
            subkeys = split_by_tree(subkey, self.shape)
            self.arr = jax.tree.map(
                lambda subkey, shape: self._generate_noise(subkey, shape, precompute),
                subkeys,
                self.shape,
            )

        if any(
            not jnp.issubdtype(x.dtype, jnp.inexact)
            for x in jtu.tree_leaves(self.shape)
        ):
            raise ValueError("UnsafeBrownianPath dtypes all have to be floating-point.")

    def _generate_noise(
        self,
        key: PRNGKeyArray,
        shape: jax.ShapeDtypeStruct,
        max_steps: int,
    ) -> Float[Array, "..."]:
        if self.levy_area is SpaceTimeTimeLevyArea:
            noise = jr.normal(key, (max_steps, 3, *shape.shape), shape.dtype)
        elif self.levy_area is SpaceTimeLevyArea:
            noise = jr.normal(key, (max_steps, 2, *shape.shape), shape.dtype)
        elif self.levy_area is BrownianIncrement:
            noise = jr.normal(key, (max_steps, *shape.shape), shape.dtype)
        else:
            assert False

        return noise

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
        t1: RealScalarLike | None = None,
        left: bool = True,
        use_levy: bool = False,
        index: None | Int[Array, ""] = None
    ) -> PyTree[Array] | AbstractBrownianIncrement:
        """Implements [`diffrax.AbstractBrownianPath.evaluate`][]."""
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

        if self.arr is not None:
            out = jax.tree.map(
                lambda shape, noise: self._evaluate_leaf_precomputed(
                    t0, t1, shape, self.levy_area, use_levy, noise
                ),
                self.shape,
                jax.tree.map(lambda x: x[index], self.arr),
            )
            if use_levy:
                out = levy_tree_transpose(self.shape, out)
                assert isinstance(out, self.levy_area)
            return out

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
            out = levy_tree_transpose(self.shape, out)
            assert isinstance(out, self.levy_area)
        return out

    @staticmethod
    def _evaluate_leaf(
        t0: RealScalarLike,
        t1: RealScalarLike,
        key,
        shape: jax.ShapeDtypeStruct,
        levy_area: type[BrownianIncrement | SpaceTimeLevyArea | SpaceTimeTimeLevyArea],
        use_levy: bool,
    ):
        w_std = jnp.sqrt(t1 - t0).astype(shape.dtype)
        dt = jnp.asarray(t1 - t0, dtype=complex_to_real_dtype(shape.dtype))

        if levy_area is SpaceTimeTimeLevyArea:
            key_w, key_hh, key_kk = jr.split(key, 3)
            w = jr.normal(key_w, shape.shape, shape.dtype) * w_std
            hh_std = w_std / math.sqrt(12)
            hh = jr.normal(key_hh, shape.shape, shape.dtype) * hh_std
            kk_std = w_std / math.sqrt(720)
            kk = jr.normal(key_kk, shape.shape, shape.dtype) * kk_std
            levy_val = SpaceTimeTimeLevyArea(dt=dt, W=w, H=hh, K=kk)

        elif levy_area is SpaceTimeLevyArea:
            key_w, key_hh = jr.split(key, 2)
            w = jr.normal(key_w, shape.shape, shape.dtype) * w_std
            hh_std = w_std / math.sqrt(12)
            hh = jr.normal(key_hh, shape.shape, shape.dtype) * hh_std
            levy_val = SpaceTimeLevyArea(dt=dt, W=w, H=hh)
        elif levy_area is BrownianIncrement:
            w = jr.normal(key, shape.shape, shape.dtype) * w_std
            levy_val = BrownianIncrement(dt=dt, W=w)
        else:
            assert False

        if use_levy:
            return levy_val
        return w

    @staticmethod
    def _evaluate_leaf_precomputed(
        t0: RealScalarLike,
        t1: RealScalarLike,
        shape: jax.ShapeDtypeStruct,
        levy_area: type[BrownianIncrement | SpaceTimeLevyArea | SpaceTimeTimeLevyArea],
        use_levy: bool,
        noises: Float[Array, "..."],
    ):
        w_std = jnp.sqrt(t1 - t0).astype(shape.dtype)
        dt = jnp.asarray(t1 - t0, dtype=complex_to_real_dtype(shape.dtype))

        if levy_area is SpaceTimeTimeLevyArea:
            w = noises[0] * w_std
            hh_std = w_std / math.sqrt(12)
            hh = noises[1] * hh_std
            kk_std = w_std / math.sqrt(720)
            kk = noises[2] * kk_std
            levy_val = SpaceTimeTimeLevyArea(dt=dt, W=w, H=hh, K=kk)

        elif levy_area is SpaceTimeLevyArea:
            w = noises[0] * w_std
            hh_std = w_std / math.sqrt(12)
            hh = noises[1] * hh_std
            levy_val = SpaceTimeLevyArea(dt=dt, W=w, H=hh)
        elif levy_area is BrownianIncrement:
            w = noises * w_std
            levy_val = BrownianIncrement(dt=dt, W=w)
        else:
            assert False

        if use_levy:
            return levy_val
        return w