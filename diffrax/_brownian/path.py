import math
from typing import cast, Optional, TypeAlias, Union

import equinox as eqx
import equinox.internal as eqxi
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
import lineax.internal as lxi
from jaxtyping import Array, Float, PRNGKeyArray, PyTree
from lineax.internal import complex_to_real_dtype

from .._custom_types import (
    AbstractBrownianIncrement,
    Args,
    BrownianIncrement,
    IntScalarLike,
    levy_tree_transpose,
    RealScalarLike,
    SpaceTimeLevyArea,
    SpaceTimeTimeLevyArea,
    Y,
)
from .._misc import (
    force_bitcast_convert_type,
    is_tuple_of_ints,
    split_by_tree,
)
from .base import AbstractBrownianPath


_Control = Union[PyTree[Array], AbstractBrownianIncrement]
_BrownianState: TypeAlias = Union[
    tuple[None, PyTree[Array], IntScalarLike], tuple[PRNGKeyArray, None, None]
]


class DirectBrownianPath(AbstractBrownianPath[_Control, _BrownianState]):
    """Brownian simulation that is only suitable for certain cases.

    This is a very quick way to simulate Brownian motion (faster than VBT), but can
    only beused if you are not using an adaptive scheme that rejects steps
    (pre-visible adaptive methods are valid).

    If using the stateless `evaluate` method, stricter requirements are imposed, namely:

    1. You are not using an adaptive solver that rejects steps.

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
    key: PRNGKeyArray
    levy_area: type[
        Union[BrownianIncrement, SpaceTimeLevyArea, SpaceTimeTimeLevyArea]
    ] = eqx.field(static=True)
    precompute: Optional[int] = eqx.field(static=True)

    def __init__(
        self,
        shape: Union[tuple[int, ...], PyTree[jax.ShapeDtypeStruct]],
        key: PRNGKeyArray,
        levy_area: type[
            Union[BrownianIncrement, SpaceTimeLevyArea, SpaceTimeTimeLevyArea]
        ] = BrownianIncrement,
        precompute: Optional[int] = None,
    ):
        self.shape = (
            jax.ShapeDtypeStruct(shape, lxi.default_floating_dtype())
            if is_tuple_of_ints(shape)
            else shape
        )
        self.key = key
        self.levy_area = levy_area
        self.precompute = precompute

        if any(
            not jnp.issubdtype(x.dtype, jnp.inexact)
            for x in jtu.tree_leaves(self.shape)
        ):
            raise ValueError("DirectBrownianPath dtypes all have to be floating-point.")

    @property
    def t0(self):
        return -jnp.inf

    @property
    def t1(self):
        return jnp.inf

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

    def init(
        self,
        t0: RealScalarLike,
        t1: RealScalarLike,
        y0: Y,
        args: Args,
    ) -> _BrownianState:
        if self.precompute is not None:
            max_steps = self.precompute
            subkey = split_by_tree(self.key, self.shape)
            noise = jtu.tree_map(
                lambda subkey, shape: self._generate_noise(subkey, shape, max_steps),
                subkey,
                self.shape,
            )
            counter = 0
            key = None
            return key, noise, counter
        else:
            noise = None
            counter = None
            _, key = jr.split(self.key)
            return key, noise, counter

    def __call__(
        self,
        t0: RealScalarLike,
        brownian_state: _BrownianState,
        t1: Optional[RealScalarLike] = None,
        left: bool = True,
        use_levy: bool = False,
    ) -> tuple[_Control, _BrownianState]:
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

        key, noises, counter = brownian_state
        if self.precompute:  # precomputed noise
            assert noises is not None and counter is not None
            out = jtu.tree_map(
                lambda shape, noise: self._evaluate_leaf_precomputed(
                    t0, t1, shape, self.levy_area, use_levy, noise
                ),
                self.shape,
                jax.tree.map(lambda x: x[counter], noises),
            )
            if use_levy:
                out = levy_tree_transpose(self.shape, out)
                assert isinstance(out, self.levy_area)
            # if a solver needs to call .evaluate twice, but wants access to the same
            # brownian motion, the solver could just use the same original state
            return out, (None, noises, counter + 1)
        else:
            assert noises is None and counter is None and key is not None
            new_key, subkey = jr.split(key)
            subkeys = split_by_tree(subkey, self.shape)
            out = jtu.tree_map(
                lambda k, shape: self._evaluate_leaf(
                    t0, t1, k, shape, self.levy_area, use_levy
                ),
                subkeys,
                self.shape,
            )
            if use_levy:
                out = levy_tree_transpose(self.shape, out)
                assert isinstance(out, self.levy_area)
            return out, (new_key, None, None)

    @eqx.filter_jit
    def evaluate(
        self,
        t0: RealScalarLike,
        t1: Optional[RealScalarLike] = None,
        left: bool = True,
        use_levy: bool = False,
    ) -> Union[PyTree[Array], AbstractBrownianIncrement]:
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
            out = levy_tree_transpose(self.shape, out)
            assert isinstance(out, self.levy_area)
        return out

    @staticmethod
    def _evaluate_leaf_precomputed(
        t0: RealScalarLike,
        t1: RealScalarLike,
        shape: jax.ShapeDtypeStruct,
        levy_area: type[
            Union[BrownianIncrement, SpaceTimeLevyArea, SpaceTimeTimeLevyArea]
        ],
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

    @staticmethod
    def _evaluate_leaf(
        t0: RealScalarLike,
        t1: RealScalarLike,
        key: PRNGKeyArray,
        shape: jax.ShapeDtypeStruct,
        levy_area: type[
            Union[BrownianIncrement, SpaceTimeLevyArea, SpaceTimeTimeLevyArea]
        ],
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


DirectBrownianPath.__init__.__doc__ = """
**Arguments:**

- `shape`: Should be a PyTree of `jax.ShapeDtypeStruct`s, representing the shape, 
    dtype, and PyTree structure of the output. For simplicity, `shape` can also just 
    be a tuple of integers, describing the shape of a single JAX array. In that case
    the dtype is chosen to be the default floating-point dtype.
- `key`: A random key.
- `levy_area`: Whether to additionally generate Lévy area. This is required by some SDE
    solvers.
- `precompute`: Size of array to precompute the brownian motion (if possible). 
    Precomputing requires additional memory at initialization time, but can result in 
    faster integrations. Some thought may be required before enabling this, as solvers 
    which require multiple brownian increments may result in index out of bounds 
    causing silent errors as the size of the precomputed brownian motion is derived 
    from the maximum steps.
"""

UnsafeBrownianPath = DirectBrownianPath
