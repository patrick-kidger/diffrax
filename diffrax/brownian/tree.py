from dataclasses import field
from typing import Literal, Optional, Tuple, Union

import equinox as eqx
import equinox.internal as eqxi
import jax
import jax.lax as lax
import jax.numpy as jnp
import jax.random as jrandom
import jax.tree_util as jtu

from ..custom_types import levy_tree_transpose, LevyVal, PyTree, Scalar
from ..misc import is_tuple_of_ints, linear_rescale, split_by_tree
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

# We define
# H_{s,t} = 1/(t-s) ( \int_s^t ( W_u - (u-s)/(t-s) W_{s,t} ) du ).
# bhh_t = t * H_{0,t}
# For more details see Definition 4.2.1 and Theorem 6.1.4 of
#
# Foster, J. M. (2020). Numerical approximations for stochastic
# differential equations [PhD thesis]. University of Oxford.


class _State(eqx.Module):
    level: int
    stu: tuple[Scalar, Scalar, Scalar]  # s, t, u
    w_stu: tuple[Scalar, Scalar, Scalar]  # W at times s, t, u
    w_st_tu: tuple[Scalar, Scalar]  # W_{s,t} and W_{t,u}
    key: "jax.random.PRNGKey"
    bhh_stu: Optional[Tuple[Scalar, Scalar, Scalar]]  # \bar{H} at times s, t, u
    bhh_st_tu: Optional[Tuple[Scalar, Scalar]]  # \bar{H}_{s,t} and \bar{H}_{t,u}
    bkk_stu: Optional[Tuple[Scalar, Scalar, Scalar]]  # \bar{K} at times s, t, u
    bkk_st_tu: Optional[Tuple[Scalar, Scalar]]  # \bar{K}_{s,t} and \bar{K}_{t,u}


def _levy_diff(x0: LevyVal, x1: LevyVal) -> LevyVal:
    r"""Computes $(W_{s,u}, H_{s,u})$ from $(W_s, \bar{H}_{s,u})$ and
    $(W_u, \bar{H}_u)$, where $\bar{H}_u = u * H_u$.

    **Arguments:**

    - `x0`: `LevyVal` at time `s`

    - `x1`: `LevyVal` at time `u`

    **Returns:**

    `LevyVal(W_su, H_su)`
    """

    su = (x1.dt - x0.dt).astype(x0.W.dtype)
    w_su = x1.W - x0.W
    if x0.H is None or x1.H is None:  # BM only case
        return LevyVal(dt=su, W=w_su, H=None, bar_H=None, K=None, bar_K=None)

    # levy_area == "space-time"
    _su = jnp.where(jnp.abs(su) < jnp.finfo(su).eps, jnp.inf, su)
    inverse_su = 1 / _su
    u_bb_s = x1.dt * x0.W - x0.dt * x1.W
    bhh_su = x1.bar_H - x0.bar_H - 0.5 * u_bb_s  # bhh_su = H_{s,u} * (u-s)
    hh_su = inverse_su * bhh_su

    return LevyVal(dt=su, W=w_su, H=hh_su, bar_H=None, K=None, bar_K=None)


def split_interval(_cond, x_stu, x_st_tu):
    x_s, x_t, x_u = x_stu
    x_st, x_tu = x_st_tu
    x_s = jnp.where(_cond, x_t, x_s)
    x_u = jnp.where(_cond, x_u, x_t)
    x_su = jnp.where(_cond, x_tu, x_st)
    return x_s, x_u, x_su


class VirtualBrownianTree(AbstractBrownianPath):
    """Brownian simulation that discretises the interval `[t0, t1]` to tolerance `tol`,
    and is piecewise quadratic at that discretisation.

    Can be initialised with `levy_area` set to `""`, or `"space-time"`.
    If `levy_area=="space_time"`, then it also computes space-time Lévy area `H`.
    This will impact the Brownian path, so even with the same key, the trajectory will
    be different depending on the value of `levy_area`.

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

        (The implementation here is a slight improvement on the reference implementation
        by using an interpolation method which ensures all the 2nd moments are correct.)
    """

    t0: Scalar = field(init=True)
    t1: Scalar = field(init=True)  # override init=False in AbstractPath
    tol: Scalar
    shape: PyTree[jax.ShapeDtypeStruct] = eqx.field(static=True)
    levy_area: Literal["", "space-time"] = eqx.field(static=True)
    key: "jax.random.PRNGKey"  # noqa: F821

    def __init__(
        self,
        t0: Scalar,
        t1: Scalar,
        tol: Scalar,
        shape: Union[Tuple[int, ...], PyTree[jax.ShapeDtypeStruct]],
        key: "jax.random.PRNGKey",
        levy_area: Literal["", "space-time"] = "",
    ):
        (t0, t1) = eqx.error_if((t0, t1), t0 >= t1, "t0 must be strictly less than t1")
        self.t0 = t0
        self.t1 = t1
        # Since we rescale the interval to [0,1],
        # we need to rescale the tolerance too.
        self.tol = tol / (self.t1 - self.t0)

        if levy_area not in ["", "space-time"]:
            raise ValueError(
                f"levy_area must be one of '', 'space-time', but got {levy_area}."
            )
        self.levy_area = levy_area
        self.shape = (
            jax.ShapeDtypeStruct(shape, jax.dtypes.canonicalize_dtype(None))
            if is_tuple_of_ints(shape)
            else shape
        )
        if any(
            not jnp.issubdtype(x.dtype, jnp.inexact)
            for x in jtu.tree_leaves(self.shape)
        ):
            raise ValueError(
                "VirtualBrownianTree dtypes all have to be floating-point."
            )
        self.key = split_by_tree(key, self.shape)

    def _denormalise_bm_inc(self, x: LevyVal) -> LevyVal:
        # TODO: demonstrate rescaling actually helps

        # Rescaling back from [0, 1] to the original interval [t0, t1].
        interval_len = self.t1 - self.t0  # can be any dtype
        sqrt_len = jnp.sqrt(interval_len)

        def sqrt_mult(z):
            # need to cast to dtype of each leaf in PyTree
            dtype = jnp.dtype(z)
            return (z * sqrt_len).astype(dtype)

        def mult(z):
            dtype = jnp.dtype(z)
            return (interval_len * z).astype(dtype)

        return LevyVal(
            dt=jtu.tree_map(mult, x.dt),
            W=jtu.tree_map(sqrt_mult, x.W),
            H=jtu.tree_map(sqrt_mult, x.H),
            bar_H=None,
            K=jtu.tree_map(sqrt_mult, x.K),
            bar_K=None,
        )

    @eqx.filter_jit
    def evaluate(
        self,
        t0: Scalar,
        t1: Optional[Scalar] = None,
        left: bool = True,
        use_levy: bool = False,
    ) -> LevyVal:
        def _is_levy_val(obj):
            return isinstance(obj, LevyVal)

        t0 = eqxi.nondifferentiable(t0, name="t0")
        # map the interval [self.t0, self.t1] onto [0,1]
        t0 = linear_rescale(self.t0, t0, self.t1)
        levy_0 = self._evaluate(t0)
        if t1 is None:
            levy_out = levy_0

        else:
            t1 = eqxi.nondifferentiable(t1, name="t1")
            # map the interval [self.t0, self.t1] onto [0,1]
            t1 = linear_rescale(self.t0, t1, self.t1)
            levy_1 = self._evaluate(t1)

            levy_out = jtu.tree_map(_levy_diff, levy_0, levy_1, is_leaf=_is_levy_val)

        levy_out = levy_tree_transpose(self.shape, self.levy_area, levy_out)
        # now map [0,1] back onto [self.t0, self.t1]
        levy_out = self._denormalise_bm_inc(levy_out)
        assert isinstance(levy_out, LevyVal)
        return levy_out if use_levy else levy_out.W

    def _evaluate(self, r: Scalar) -> PyTree[LevyVal]:
        """Maps the _evaluate_leaf function at time τ using self.key onto self.shape"""
        r = eqxi.error_if(
            r,
            r < 0,
            "Cannot evaluate VirtualBrownianTree outside of its range [t0, t1].",
        )
        r = eqxi.error_if(
            r,
            r > 1,
            "Cannot evaluate VirtualBrownianTree outside of its range [t0, t1].",
        )
        # Clip because otherwise the while loop below won't terminate, and the above
        # errors are only raised after everything has finished executing.
        r = jnp.clip(r, 0, 1)
        map_func = lambda key, shape: self._evaluate_leaf(key, r, shape)
        return jtu.tree_map(map_func, self.key, self.shape)

    def _brownian_arch(
        self,
        level: int,
        s: Scalar,
        u: Scalar,
        w: Tuple[Scalar, Scalar, Scalar],
        key,
        shape,
        dtype,
        bhh: Optional[Tuple[Scalar, Scalar, Scalar]],
        bkk: Optional[Tuple[Scalar, Scalar, Scalar]],
    ):
        r"""For `t = (s+u)/2` evaluates `w_t` and (optionally) `bhh_t`
         conditioned on `w_s`, `w_u`, `bhh_s`, `bhh_u`, where
         `bhh_st` represents $\bar{H}_{s,t} \coloneqq (t-s) H_{s,t}$.
         To avoid cancellation errors, requires an input of `w_su`, `bhh_su`
         and also returns `w_st` and `w_tu` in addition to just `w_t`. Same for `bhh`
         if it is not None.
         Note that the inputs and outputs already contain `bkk`. These values are
         there for the sake of a future extension with "space-time-time" Levy area
         and should be None for now.

        **Arguments:**
            - `s`: start time
            - `u`: end time
            - `w_s`: value of BM at s
            - `w_u`: value of BM at u
            - `w_su`: $W_{s,u}$
            - `key`:
            - `shape`:
            - `dtype`:
            - `bhh`: (optional) $(\bar{H}_s, \bar{H}_u, \bar{H}_{s,u})$
            - `bkk`: (optional) $(\bar{K}_s, \bar{K}_u, \bar{K}_{s,u})$

        **Returns:**
            - `t`: midpoint time
            - `w_t`: value of BM at t
            - `w_st_tu`: $(W_{s,t}, W_{t,u})$
            - `bhh`: (optional) $(\bar{H}_s, \bar{H}_t, \bar{H}_u)$
            - `bhh_st_tu`: (optional) $(\bar{H}_{s,t}, \bar{H}_{t,u})$
            - `bkk_t`: (optional) $(\bar{K}_s, \bar{K}_t, \bar{K}_u)$
            - `bkk_st_tu`: (optional) $(\bar{K}_{s,t}, \bar{K}_{t,u})$

        """

        su = jnp.power(jnp.asarray(2.0, dtype), -level)
        st = su / 2
        t = s + st
        u_minus_s = u - s
        su = eqxi.error_if(
            su,
            jnp.abs(u_minus_s - su) > 1e-17,
            "VirtualBrownianTree: u-s is not 2^(-tree_level)",
        )
        root_su = jnp.sqrt(su)

        w_s, w_u, w_su = w

        if self.levy_area == "space-time":
            assert bhh is not None
            assert bkk is None
            bhh_s, bhh_u, bhh_su = bhh

            z1_key, z2_key = jrandom.split(key, 2)
            z1 = jrandom.normal(z1_key, shape, dtype)
            z2 = jrandom.normal(z2_key, shape, dtype)
            z = z1 * (root_su / 4)
            n = z2 * jnp.sqrt(su / 12)

            w_term1 = w_su / 2
            w_term2 = 3 / (2 * su) * bhh_su + z
            w_st = w_term1 + w_term2
            w_tu = w_term1 - w_term2
            w_st_tu = (w_st, w_tu)

            bhh_term1 = bhh_su / 8 - su / 4 * z
            bhh_term2 = su / 4 * n
            bhh_st = bhh_term1 + bhh_term2
            bhh_tu = bhh_term1 - bhh_term2
            bhh_st_tu = (bhh_st, bhh_tu)

            w_t = w_s + w_st
            bhh_t = bhh_s + bhh_st + 0.5 * (t * w_s - s * w_t)
            bhh = (bhh_s, bhh_t, bhh_u)
            bkk = None
            bkk_st_tu = None

        else:
            assert bhh is None
            assert bkk is None
            mean = 0.5 * w_su
            w_term2 = root_su / 2 * jrandom.normal(key, shape, dtype)
            w_st = mean + w_term2
            w_tu = mean - w_term2
            w_st_tu = (w_st, w_tu)
            w_t = w_s + w_st
            bhh, bhh_st_tu, bkk, bkk_st_tu = None, None, None, None
        return t, w_t, w_st_tu, bhh, bhh_st_tu, bkk, bkk_st_tu

    def _evaluate_leaf(
        self,
        key,
        r: Scalar,
        shape: jax.ShapeDtypeStruct,
    ) -> LevyVal:
        shape, dtype = shape.shape, shape.dtype

        t0 = jnp.zeros((), dtype)
        t1 = jnp.ones((), dtype)
        r = jnp.asarray(r, dtype)

        w_0 = jnp.zeros(shape, dtype)

        if self.levy_area == "space-time":
            key, init_key_w, init_key_la, midpoint_key = jrandom.split(key, 4)
            w_1 = jrandom.normal(init_key_w, shape, dtype)

            bhh_1 = jnp.sqrt(1 / 12) * jrandom.normal(init_key_la, shape, dtype)
            bhh_0 = jnp.zeros_like(bhh_1)
            bhh = (bhh_0, bhh_1, bhh_1)
            bkk = None

        else:
            key, init_key_w, midpoint_key = jrandom.split(key, 3)
            w_1 = jrandom.normal(init_key_w, shape, dtype)
            bhh = None
            bkk = None

        w = (w_0, w_1, w_1)

        half, w_half, w_inc, bhh, bhh_inc, bkk, bkk_inc = self._brownian_arch(
            0, t0, t1, w, key, shape, dtype, bhh, bkk
        )
        init_state = _State(
            level=0,
            stu=(t0, half, t1),
            w_stu=(w_0, w_half, w_1),
            w_st_tu=w_inc,
            key=key,
            bhh_stu=bhh,
            bhh_st_tu=bhh_inc,
            bkk_stu=bkk,
            bkk_st_tu=bkk_inc,
        )

        def _cond_fun(_state):
            # Slight adaptation on the version of the algorithm given in the
            # above-referenced thesis. There the returned value is snapped to one of
            # the dyadic grid points, so they just stop once
            # jnp.abs(τ - state.s) > self.tol
            # Here, because we use quadratic splines to get better samples, we always
            # iterate down to the level of the spline.
            _s, _t, _u = _state.stu
            return (_u - _s) > 2 * self.tol

        def _body_fun(_state: _State):
            """Single-step of binary search for τ."""
            _level = _state.level + 1
            _key1, _key2 = jrandom.split(_state.key, 2)
            _s, _t, _u = _state.stu
            _cond = r > _t
            _s = jnp.where(_cond, _t, _s)
            _u = jnp.where(_cond, _u, _t)

            _w = split_interval(_cond, _state.w_stu, _state.w_st_tu)
            if self.levy_area in ["space-time", "space-time-time"]:
                _bhh = split_interval(_cond, _state.bhh_stu, _state.bhh_st_tu)
                _bkk = None
            else:
                _bhh = None
                _bkk = None

            _key = jnp.where(_cond, _key1, _key2)
            _key, _midpoint_key = jrandom.split(_key, 2)

            _t, _w_t, _w_inc, _bhh, _bhh_inc, _bkk, _bkk_inc = self._brownian_arch(
                _level, _s, _u, _w, _midpoint_key, shape, dtype, _bhh, _bkk
            )

            return _State(
                level=_level,
                stu=(_s, _t, _u),
                w_stu=(_w[0], _w_t, _w[2]),
                w_st_tu=_w_inc,
                key=_key,
                bhh_stu=_bhh,
                bhh_st_tu=_bhh_inc,
                bkk_stu=_bkk,
                bkk_st_tu=_bkk_inc,
            )

        final_state = lax.while_loop(_cond_fun, _body_fun, init_state)

        level = final_state.level + 1
        s, t, u = final_state.stu

        # Split the interval in half one last time depending on whether r < t or r > t
        # but this time complete the step with the general interpolation, rather
        # than the midpoint rule (as given by _brownian_arch).

        cond = r > t
        s = jnp.where(cond, t, s)
        u = jnp.where(cond, u, t)
        su = jnp.power(jnp.asarray(2.0, dtype), -level)
        su = eqxi.error_if(
            su,
            jnp.abs(u - s - su) > 1e-17,
            "VirtualBrownianTree: u-s is not 2^(-tree_level) in final step.",
        )

        sr = r - s
        ru = u - r  # make sure su = sr + ru regardless of cancellation error

        w_s, w_u, w_su = split_interval(cond, final_state.w_stu, final_state.w_st_tu)
        key1, key2 = jrandom.split(final_state.key, 2)
        key = jnp.where(cond, key1, key2)

        # BM only case
        if self.levy_area == "":
            z = jrandom.normal(key, shape, dtype)
            w_sr = sr / su * w_su + jnp.sqrt(sr * ru / su) * z
            w_r = w_s + w_sr
            return LevyVal(dt=r, W=w_r, H=None, bar_H=None, K=None, bar_K=None)

        elif self.levy_area == "space-time":
            bhh_s, bhh_u, bhh_su = split_interval(
                cond, final_state.bhh_stu, final_state.bhh_st_tu
            )
            sr3 = jnp.power(sr, 3)
            ru3 = jnp.power(ru, 3)
            su3 = jnp.power(su, 3)
            key1, key2 = jrandom.split(key, 2)
            x1 = jrandom.normal(key1, shape, dtype)
            x2 = jrandom.normal(key2, shape, dtype)

            sr_ru_half = jnp.sqrt(sr * ru)
            d = jnp.sqrt(sr3 + ru3)
            d_prime = 1 / (2 * su * d)
            a = d_prime * sr3 * sr_ru_half
            b = d_prime * ru3 * sr_ru_half

            w_sr = sr / su * w_su + 6 * sr * ru / su3 * bhh_su + 2 * (a + b) / su * x1
            w_r = w_s + w_sr
            c = jnp.sqrt(3 * sr3 * ru3) / (6 * d)
            bhh_sr = sr3 / su3 * bhh_su - a * x1 + c * x2
            bhh_r = bhh_s + bhh_sr + 0.5 * (r * w_s - s * w_r)

            inverse_r = 1 / jnp.where(jnp.abs(r) < jnp.finfo(r).eps, jnp.inf, r)
            hh_r = inverse_r * bhh_r

        else:
            assert False

        return LevyVal(dt=r, W=w_r, H=hh_r, bar_H=bhh_r, K=None, bar_K=None)
