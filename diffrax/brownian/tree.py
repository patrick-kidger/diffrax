from dataclasses import field
from typing import Tuple

import equinox as eqx
import jax
import jax.lax as lax
import jax.numpy as jnp
import jax.random as jrandom

from ..custom_types import Array, Scalar
from ..misc import error_if
from .base import AbstractBrownianPath


#
# The notation here comes from section 5.5.2 of
#
# @phdthesis{kidger2021on,
#     title={{O}n {N}eural {D}ifferential {E}quations},
#     author={Patrick Kidger},
#     year={2021},
#     school={University of Oxford},
# }
#


class _State(eqx.Module):
    s: Scalar
    t: Scalar
    u: Scalar
    w_s: Scalar
    w_t: Scalar
    w_u: Scalar
    key: "jax.random.PRNGKey"


class VirtualBrownianTree(AbstractBrownianPath):
    """Brownian simulation that discretises the interval `[t0, t1]` to tolerance `tol`,
    and is piecewise linear at that discretisation.

    ??? cite "Reference"

        ```bibtex
        @article{li2020scalable,
          title={Scalable gradients for stochastic differential equations},
          author={Li, Xuechen and Wong, Ting-Kam Leonard and Chen, Ricky T. Q. and
                  Duvenaud, David},
          journal={International Conference on Artificial Intelligence and Statistics},
          year={2020}
        }
        ```
    """

    t0: Scalar = field(init=True)
    t1: Scalar = field(init=True)  # override init=False in AbstractPath
    tol: Scalar
    shape: Tuple[int] = eqx.static_field()
    key: "jax.random.PRNGKey"  # noqa: F821

    @eqx.filter_jit
    def evaluate(self, t0: Scalar, t1: Scalar, left: bool = True) -> Array:
        del left
        return self._evaluate(t1) - self._evaluate(t0)

    def _brownian_bridge(self, s, t, u, w_s, w_u, key):
        mean = w_s + (w_u - w_s) * ((t - s) / (u - s))
        var = (u - t) * (t - s) / (u - s)
        std = jnp.sqrt(var)
        return mean + std * jrandom.normal(key, self.shape)

    def _evaluate(self, τ: Scalar) -> Array:
        cond = self.t0 < self.t1
        t0 = jnp.where(cond, self.t0, self.t1)
        t1 = jnp.where(cond, self.t1, self.t0)

        error_if(
            τ < t0, "Cannot evaluate VirtualBrownianTree outside of its range [t0, t1]."
        )
        error_if(
            τ > t1, "Cannot evaluate VirtualBrownianTree outside of its range [t0, t1]."
        )
        # Clip because otherwise the while loop below won't terminate, and the above
        # errors are only raised after everything has finished executing.
        τ = jnp.clip(τ, t0, t1)

        key, init_key = jrandom.split(self.key, 2)
        thalf = t0 + 0.5 * (t1 - t0)
        w_t1 = jrandom.normal(init_key, self.shape) * jnp.sqrt(t1 - t0)
        w_thalf = self._brownian_bridge(t0, thalf, t1, 0, w_t1, key)
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
            return jnp.abs(τ - _state.t) > self.tol

        def _body_fun(_state):
            _key1, _key2 = jrandom.split(_state.key, 2)
            _cond = τ > _state.t
            _s = jnp.where(_cond, _state.t, _state.s)
            _u = jnp.where(_cond, _state.u, _state.t)
            _w_s = jnp.where(_cond, _state.w_t, _state.w_s)
            _w_u = jnp.where(_cond, _state.w_u, _state.w_t)
            _key = jnp.where(_cond, _key1, _key2)
            _t = _s + 0.5 * (_u - _s)
            _w_t = self._brownian_bridge(_s, _t, _u, _w_s, _w_u, _key)
            return _State(s=_s, t=_t, u=_u, w_s=_w_s, w_t=_w_t, w_u=_w_u, key=_key)

        final_state = lax.while_loop(_cond_fun, _body_fun, init_state)
        cond = τ > final_state.t
        s = jnp.where(cond, final_state.t, final_state.s)
        u = jnp.where(cond, final_state.u, final_state.t)
        w_s = jnp.where(cond, final_state.w_t, final_state.w_s)
        w_u = jnp.where(cond, final_state.w_u, final_state.w_t)
        return w_s + (w_u - w_s) * ((τ - s) / (u - s))


VirtualBrownianTree.__init__.__doc__ = """
**Arguments:**

- `t0`: The start of the interval the Brownian motion is defined over.
- `t1`: The start of the interval the Brownian motion is defined over.
- `tol`: The discretisation that `[t0, t1]` is discretised to.
- `shape`: What shape each individual Brownian sample should be.
- `key`: A random key.
"""
