"""
v0.5.0 introduced a new implementation for `diffrax.VirtualBrownianTree` that is
additionally capable of computing Levy area.

Here we check the speed of the new implementation against the old implementation, to be
sure that it is still fast.
"""

import timeit
from typing import cast, Optional, Union
from typing_extensions import TypeAlias

import equinox as eqx
import equinox.internal as eqxi
import jax
import jax.lax as lax
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
import lineax.internal as lxi
import numpy as np
from diffrax import AbstractBrownianPath, VirtualBrownianTree
from jaxtyping import Array, Float, PRNGKeyArray, PyTree, Real


RealScalarLike: TypeAlias = Real[Union[int, float, Array, np.ndarray], ""]


class _State(eqx.Module):
    s: RealScalarLike
    t: RealScalarLike
    u: RealScalarLike
    w_s: Float[Array, " *shape"]
    w_t: Float[Array, " *shape"]
    w_u: Float[Array, " *shape"]
    key: PRNGKeyArray


class OldVBT(AbstractBrownianPath):
    t0: RealScalarLike
    t1: RealScalarLike
    tol: RealScalarLike
    shape: PyTree[jax.ShapeDtypeStruct] = eqx.field(static=True)
    key: PRNGKeyArray

    def __init__(
        self,
        t0: RealScalarLike,
        t1: RealScalarLike,
        tol: RealScalarLike,
        shape: tuple[int, ...],
        key: PRNGKeyArray,
        levy_area: str,
    ):
        assert levy_area == ""
        self.t0 = t0
        self.t1 = t1
        self.tol = tol
        self.shape = jax.ShapeDtypeStruct(shape, lxi.default_floating_dtype())
        self.key = key

    @property
    def levy_area(self):
        assert False

    @eqx.filter_jit
    def evaluate(
        self,
        t0: RealScalarLike,
        t1: Optional[RealScalarLike] = None,
        left: bool = True,
        use_levy: bool = False,
    ) -> PyTree[Array]:
        del left, use_levy
        t0 = eqxi.nondifferentiable(t0, name="t0")
        if t1 is None:
            return self._evaluate(t0)
        else:
            t1 = cast(RealScalarLike, eqxi.nondifferentiable(t1, name="t1"))
            return jtu.tree_map(
                lambda x, y: x - y,
                self._evaluate(t1),
                self._evaluate(t0),
            )

    def _evaluate(self, τ: RealScalarLike) -> PyTree[Array]:
        map_func = lambda key, struct: self._evaluate_leaf(key, τ, struct)
        return jtu.tree_map(map_func, self.key, self.shape)

    def _brownian_bridge(self, s, t, u, w_s, w_u, key, shape, dtype):
        mean = w_s + (w_u - w_s) * ((t - s) / (u - s))
        var = (u - t) * (t - s) / (u - s)
        std = jnp.sqrt(var)
        return mean + std * jr.normal(key, shape, dtype)

    def _evaluate_leaf(
        self,
        key,
        τ: RealScalarLike,
        struct: jax.ShapeDtypeStruct,
    ) -> Array:
        shape, dtype = struct.shape, struct.dtype

        cond = self.t0 < self.t1
        t0 = jnp.where(cond, self.t0, self.t1).astype(dtype)
        t1 = jnp.where(cond, self.t1, self.t0).astype(dtype)

        t0 = eqxi.error_if(
            t0,
            τ < t0,
            "Cannot evaluate VirtualBrownianTree outside of its range [t0, t1].",
        )
        t1 = eqxi.error_if(
            t1,
            τ > t1,
            "Cannot evaluate VirtualBrownianTree outside of its range [t0, t1].",
        )
        τ = jnp.clip(τ, t0, t1).astype(dtype)

        key, init_key = jr.split(key, 2)
        thalf = t0 + 0.5 * (t1 - t0)
        w_t1 = jr.normal(init_key, shape, dtype) * jnp.sqrt(t1 - t0)
        w_thalf = self._brownian_bridge(t0, thalf, t1, 0, w_t1, key, shape, dtype)
        init_state = _State(
            s=t0,
            t=thalf,
            u=t1,
            w_s=jnp.zeros_like(w_t1),
            w_t=w_thalf,
            w_u=w_t1,
            key=key,
        )

        def _cond_fun(_state):
            return (_state.u - _state.s) > self.tol

        def _body_fun(_state):
            _key1, _key2 = jr.split(_state.key, 2)
            _cond = τ > _state.t
            _s = jnp.where(_cond, _state.t, _state.s)
            _u = jnp.where(_cond, _state.u, _state.t)
            _w_s = jnp.where(_cond, _state.w_t, _state.w_s)
            _w_u = jnp.where(_cond, _state.w_u, _state.w_t)
            _key = jnp.where(_cond, _key1, _key2)
            _t = _s + 0.5 * (_u - _s)
            _w_t = self._brownian_bridge(_s, _t, _u, _w_s, _w_u, _key, shape, dtype)
            return _State(s=_s, t=_t, u=_u, w_s=_w_s, w_t=_w_t, w_u=_w_u, key=_key)

        final_state = lax.while_loop(_cond_fun, _body_fun, init_state)

        s = final_state.s
        u = final_state.u
        w_s = final_state.w_s
        w_t = final_state.w_t
        w_u = final_state.w_u
        rescaled_τ = (τ - s) / (u - s)

        A = jnp.array([[2, -4, 2], [-3, 4, -1], [1, 0, 0]])
        coeffs = jnp.tensordot(A, jnp.stack([w_s, w_t, w_u]), axes=1)
        return jnp.polyval(coeffs, rescaled_τ)


key = jr.PRNGKey(0)
t0, t1 = 0.3, 20.3


def time_tree(tree_cls, num_ts, tol, levy_area):
    tree = tree_cls(t0=t0, t1=t1, tol=tol, shape=(10,), key=key, levy_area=levy_area)

    if num_ts == 1:
        ts = 11.2

        @jax.jit
        @eqx.debug.assert_max_traces(max_traces=1)
        def run(_ts):
            return tree.evaluate(_ts, use_levy=True)
    else:
        ts = jnp.linspace(t0, t1, num_ts)

        @jax.jit
        @eqx.debug.assert_max_traces(max_traces=1)
        def run(_ts):
            return jax.vmap(lambda _t: tree.evaluate(_t, use_levy=True))(_ts)

    return min(
        timeit.repeat(lambda: jax.block_until_ready(run(ts)), number=1, repeat=100)
    )


for levy_area in ("", "space-time"):
    print(f"-   {levy_area=}")
    for tol in (2**-3, 2**-12):
        print(f"--  {tol=}")
        for num_ts in (1, 100):
            print(f"--- {num_ts=}")
            if levy_area == "":
                print(f"Old: {time_tree(OldVBT, num_ts, tol, levy_area):.5f}")
            print(f"new: {time_tree(VirtualBrownianTree, num_ts, tol, levy_area):.5f}")
    print("")
