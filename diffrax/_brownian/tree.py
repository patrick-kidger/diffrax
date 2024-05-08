import math
from typing import Literal, Optional, TypeVar, Union
from typing_extensions import TypeAlias

import equinox as eqx
import equinox.internal as eqxi
import jax
import jax.lax as lax
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
import lineax.internal as lxi
from jaxtyping import Array, Inexact, PRNGKeyArray, PyTree
from lineax.internal import complex_to_real_dtype

from .._custom_types import (
    AbstractBrownianIncrement,
    BoolScalarLike,
    BrownianIncrement,
    IntScalarLike,
    levy_tree_transpose,
    RealScalarLike,
    SpaceTimeLevyArea,
    SpaceTimeTimeLevyArea,
)
from .._misc import (
    is_tuple_of_ints,
    linear_rescale,
    split_by_tree,
)
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

# We define
# H_{s,t} = 1/(t-s) ( \int_s^t ( W_u - (u-s)/(t-s) W_{s,t} ) du ).
# bhh_t = t * H_{0,t}
# For more details see section 6.1 of
# @phdthesis{foster2020a,
#   publisher = {University of Oxford},
#   school = {University of Oxford},
#   title = {Numerical approximations for stochastic differential equations},
#   author = {Foster, James M.},
#   year = {2020}
# }
# For more about space-time Lévy area see Definition 4.2.1.
# For the midpoint rule for generating space-time Lévy area see Theorem 6.1.6.
# For the general interpolation rule for space-time Lévy area see Theorem 6.1.4.

FloatDouble: TypeAlias = tuple[Inexact[Array, " *shape"], Inexact[Array, " *shape"]]
FloatTriple: TypeAlias = tuple[
    Inexact[Array, " *shape"], Inexact[Array, " *shape"], Inexact[Array, " *shape"]
]
_Spline: TypeAlias = Literal["sqrt", "quad", "zero"]
_BrownianReturn = TypeVar("_BrownianReturn", bound=AbstractBrownianIncrement)


# An internal dataclass that holds the rescaled Lévy areas
# in addition to the non-rescaled ones, for the purposes of
# taking the difference between two Lévy areas.
class _LevyVal(eqx.Module):
    dt: RealScalarLike
    W: Array
    H: Optional[Array]
    bar_H: Optional[Array]
    K: Optional[Array]
    bar_K: Optional[Array]

    def __check_init__(self):
        if self.H is None:
            assert self.bar_H is None
            assert self.K is None
        if self.K is None:
            assert self.bar_K is None


class _State(eqx.Module):
    level: IntScalarLike  # level of the tree
    s: RealScalarLike  # starting time of the interval
    w_s_u_su: FloatTriple  # W_s, W_u, W_{s,u}
    key: PRNGKeyArray
    bhh_s_u_su: Optional[FloatTriple]  # \bar{H}_s, _u, _{s,u}
    bkk_s_u_su: Optional[FloatTriple]  # \bar{K}_s, _u, _{s,u}


def _levy_diff(_, x0: _LevyVal, x1: _LevyVal) -> AbstractBrownianIncrement:
    r"""Computes $(W_{s,u}, H_{s,u})$ from $(W_s, \bar{H}_{s,u})$ and
    $(W_u, \bar{H}_u)$, where $\bar{H}_u = u * H_u$.

    **Arguments:**

    - `_`: unused, for the purposes of aligning the `jtu.tree_map`.
    - `x0`: `_LevyVal` at time `s`.
    - `x1`: `_LevyVal` at time `u`.

    **Returns:**

    `AbstractBrownianIncrement(W_su, H_su, K_su)`
    """
    dtype = jnp.result_type(x0.W)
    tdtype = complex_to_real_dtype(dtype)
    su = jnp.asarray(x1.dt - x0.dt, dtype=tdtype)
    if x0.H is None:  # BM only case
        assert x1.H is None
        return BrownianIncrement(dt=su, W=x1.W - x0.W)

    # the following computation is common to the space-time
    # and the space-time-time Lévy area case
    assert x0.H is not None
    assert x1.H is not None
    assert x0.bar_H is not None
    assert x1.bar_H is not None
    w_su = x1.W - x0.W
    inverse_su = 1 / jnp.where(jnp.abs(su) < jnp.finfo(su).eps, jnp.inf, su)
    with jax.numpy_dtype_promotion("standard"):
        u_bb_s = x1.dt * x0.W - x0.dt * x1.W
        bhh_su = x1.bar_H - x0.bar_H - 0.5 * u_bb_s  # bhh_su = H_{s,u} * (u-s)
        hh_su = inverse_su * bhh_su

    if x0.K is None:  # space-time Lévy area case
        return SpaceTimeLevyArea(dt=su, W=w_su, H=hh_su)

    elif x0.K is not None:  # space-time-time Lévy area case
        assert x1.K is not None
        assert x0.bar_K is not None
        assert x1.bar_K is not None
        with jax.numpy_dtype_promotion("standard"):
            bkk_su = (
                x1.bar_K
                - x0.bar_K
                - (su / 2) * x0.bar_H
                + (x0.dt / 2) * bhh_su
                - ((x1.dt - 2 * x0.dt) / 12) * u_bb_s
            )
            su2 = jnp.square(su)
            inverse_su2 = 1 / jnp.where(jnp.abs(su2) < jnp.finfo(su2).eps, jnp.inf, su2)
            kk_su = inverse_su2 * bkk_su
        return SpaceTimeTimeLevyArea(dt=su, W=w_su, H=hh_su, K=kk_su)
    else:
        assert False


def _make_levy_val(_, x: _LevyVal) -> AbstractBrownianIncrement:
    tdtype = complex_to_real_dtype(x.W)
    dt = jnp.asarray(x.dt, dtype=tdtype)
    if x.H is None:
        return BrownianIncrement(dt=dt, W=x.W)
    elif x.K is None:
        return SpaceTimeLevyArea(dt=dt, W=x.W, H=x.H)
    else:
        return SpaceTimeTimeLevyArea(dt=dt, W=x.W, H=x.H, K=x.K)


def _split_interval(
    pred: BoolScalarLike, x_stu: Optional[FloatTriple], x_st_tu: Optional[FloatDouble]
) -> Optional[FloatTriple]:
    if x_stu is None:
        assert x_st_tu is None
        return None
    assert x_st_tu is not None
    x_s, x_t, x_u = x_stu
    x_st, x_tu = x_st_tu
    x_s = jnp.where(pred, x_t, x_s)
    x_u = jnp.where(pred, x_u, x_t)
    x_su = jnp.where(pred, x_tu, x_st)
    return x_s, x_u, x_su


class VirtualBrownianTree(AbstractBrownianPath):
    """Brownian simulation that discretises the interval `[t0, t1]` to tolerance `tol`.

    !!! info "Lévy Area"
        The parameter `levy_area` can be set to one of:

        - [`diffrax.BrownianIncrement`][] (default, generates the increment of W)
        - [`diffrax.SpaceTimeLevyArea`][] (generates W and the space-time Lévy area H)
        - [`diffrax.SpaceTimeTimeLevyArea`][] (generates W, H and the space-time-time
                                                Lévy area K)

        The choice of `levy_area` will impact the Brownian path, so even with the same
        key, the trajectory will be different depending on the value of `levy_area`.

    ??? cite "Reference"

        Virtual Brownian trees were proposed in
        ```bibtex
        @article{li2020scalable,
          title={Scalable gradients for stochastic differential equations},
          author={Li, Xuechen and Wong, Ting-Kam Leonard and Chen, Ricky T. Q. and
                  Duvenaud, David},
          journal={International Conference on Artificial Intelligence and Statistics},
          year={2020}
        }
        ```

        The implementation here is an improvement on the above, in that it additionally
        simulates space-time and space-time-time Lévy areas, and exactly matches the
        distribution of the Brownian motion and its Lévy areas at all query times.
        This is due to the paper

        ```bibtex
        @misc{jelinčič2024singleseed,
          title={Single-seed generation of Brownian paths and integrals
          for adaptive and high order SDE solvers},
          author={Andraž Jelinčič and James Foster and Patrick Kidger},
          year={2024},
          eprint={2405.06464},
          archivePrefix={arXiv},
          primaryClass={math.NA}
        }
        ```

        and Theorem 6.1.6 of

        ```bibtex
        @phdthesis{foster2020a,
          publisher = {University of Oxford},
          school = {University of Oxford},
          title = {Numerical approximations for stochastic differential equations},
          author = {Foster, James M.},
          year = {2020}
        }
        ```
    """

    t0: RealScalarLike
    t1: RealScalarLike
    tol: RealScalarLike
    shape: PyTree[jax.ShapeDtypeStruct] = eqx.field(static=True)
    levy_area: type[
        Union[BrownianIncrement, SpaceTimeLevyArea, SpaceTimeTimeLevyArea]
    ] = eqx.field(static=True)
    key: PyTree[PRNGKeyArray]
    _spline: _Spline = eqx.field(static=True)

    @eqxi.doc_remove_args("_spline")
    def __init__(
        self,
        t0: RealScalarLike,
        t1: RealScalarLike,
        tol: RealScalarLike,
        shape: Union[tuple[int, ...], PyTree[jax.ShapeDtypeStruct]],
        key: PRNGKeyArray,
        levy_area: type[
            Union[BrownianIncrement, SpaceTimeLevyArea, SpaceTimeTimeLevyArea]
        ] = BrownianIncrement,
        _spline: _Spline = "sqrt",
    ):
        (t0, t1) = eqx.error_if((t0, t1), t0 >= t1, "t0 must be strictly less than t1")
        self.t0 = t0
        self.t1 = t1
        # Since we rescale the interval to [0,1],
        # we need to rescale the tolerance too.
        self.tol = tol / (self.t1 - self.t0)
        self.levy_area = levy_area
        self._spline = _spline
        self.shape = (
            jax.ShapeDtypeStruct(shape, lxi.default_floating_dtype())
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

    def _denormalise_bm_inc(self, x: _BrownianReturn) -> _BrownianReturn:
        # Rescaling back from [0, 1] to the original interval [t0, t1].
        interval_len = self.t1 - self.t0  # can be any dtype
        sqrt_len = jnp.sqrt(interval_len)

        def mult(z):
            dtype = jnp.result_type(z)
            return jnp.astype(interval_len, dtype) * z

        def sqrt_mult(z):
            # need to cast to dtype of each leaf in PyTree
            dtype = jnp.result_type(z)
            return jnp.astype(sqrt_len, dtype) * z

        def is_dt(z):
            return z is x.dt

        dt, other = eqx.partition(x, is_dt)
        dt_normalized = jtu.tree_map(mult, dt)
        other_normalized = jtu.tree_map(sqrt_mult, other)
        return eqx.combine(dt_normalized, other_normalized)

    @eqx.filter_jit
    def evaluate(
        self,
        t0: RealScalarLike,
        t1: Optional[RealScalarLike] = None,
        left: bool = True,
        use_levy: bool = False,
    ) -> Union[PyTree[Array], AbstractBrownianIncrement]:
        t0 = eqxi.nondifferentiable(t0, name="t0")
        # map the interval [self.t0, self.t1] onto [0,1]
        t0 = linear_rescale(self.t0, t0, self.t1)
        levy_0 = self._evaluate(t0)
        if t1 is None:
            levy_out = levy_0
            levy_out = jtu.tree_map(_make_levy_val, self.shape, levy_out)
        else:
            t1 = eqxi.nondifferentiable(t1, name="t1")
            # map the interval [self.t0, self.t1] onto [0,1]
            t1 = linear_rescale(self.t0, t1, self.t1)
            levy_1 = self._evaluate(t1)
            # take the difference between the output for t0 and t1 via Chen's relation
            levy_out = jtu.tree_map(_levy_diff, self.shape, levy_0, levy_1)

        levy_out = levy_tree_transpose(self.shape, levy_out)
        # now map [0,1] back onto [self.t0, self.t1]
        levy_out = self._denormalise_bm_inc(levy_out)
        assert isinstance(levy_out, self.levy_area)
        return levy_out if use_levy else levy_out.W

    def _evaluate(self, r: RealScalarLike) -> PyTree:
        """Maps the _evaluate_leaf function at time r using self.key onto self.shape"""
        r = eqxi.error_if(
            r,
            (r < 0) | (r > 1),
            "Cannot evaluate VirtualBrownianTree outside of its range [t0, t1].",
        )
        map_func = lambda key, shape: self._evaluate_leaf(key, r, shape)
        return jtu.tree_map(map_func, self.key, self.shape)

    def _evaluate_leaf(
        self,
        key,
        r: RealScalarLike,
        struct: jax.ShapeDtypeStruct,
    ) -> _LevyVal:
        shape, dtype = struct.shape, struct.dtype
        tdtype = complex_to_real_dtype(dtype)
        t0 = jnp.zeros((), tdtype)
        r = jnp.asarray(r, tdtype)

        if self.levy_area is SpaceTimeTimeLevyArea:
            state_key, init_key_w, init_key_hh, init_key_kk = jr.split(key, 4)
            bhh_1 = jr.normal(init_key_hh, shape, dtype) / math.sqrt(12)
            bhh_0 = jnp.zeros_like(bhh_1)
            bhh = (bhh_0, bhh_1, bhh_1)
            bkk_1 = jr.normal(init_key_kk, shape, dtype) / math.sqrt(720)
            bkk_0 = jnp.zeros_like(bkk_1)
            bkk = (bkk_0, bkk_1, bkk_1)

        elif self.levy_area is SpaceTimeLevyArea:
            state_key, init_key_w, init_key_hh = jr.split(key, 3)
            bhh_1 = jr.normal(init_key_hh, shape, dtype) / math.sqrt(12)
            bhh_0 = jnp.zeros_like(bhh_1)
            bhh = (bhh_0, bhh_1, bhh_1)
            bkk = None

        elif self.levy_area is BrownianIncrement:
            state_key, init_key_w = jr.split(key, 2)
            bhh = None
            bkk = None

        else:
            assert False

        w_0 = jnp.zeros(shape, dtype)
        w_1 = jr.normal(init_key_w, shape, dtype)
        w = (w_0, w_1, w_1)

        init_state = _State(
            level=0, s=t0, w_s_u_su=w, key=state_key, bhh_s_u_su=bhh, bkk_s_u_su=bkk
        )

        def _cond_fun(_state):
            """Condition for the binary search for r."""
            # If true, continue splitting the interval and descending the tree.
            return 2.0 ** (-_state.level) > self.tol

        def _body_fun(_state: _State):
            """Single-step of the binary search for r."""

            (
                _t,
                _w_stu,
                _w_st_tu,
                _keys,
                _bhh_stu,
                _bhh_st_tu,
                _bkk_stu,
                _bkk_st_tu,
            ) = self._brownian_arch(_state, shape, dtype)

            _level = _state.level + 1
            _cond = r > _t
            _s = jnp.where(_cond, _t, _state.s)
            _key_st, _key_tu = _keys
            _key = jnp.where(_cond, _key_st, _key_tu)

            _w = _split_interval(_cond, _w_stu, _w_st_tu)
            assert _w is not None
            _bhh = _split_interval(_cond, _bhh_stu, _bhh_st_tu)
            _bkk = _split_interval(_cond, _bkk_stu, _bkk_st_tu)

            return _State(
                level=_level,
                s=_s,
                w_s_u_su=_w,
                key=_key,
                bhh_s_u_su=_bhh,
                bkk_s_u_su=_bkk,
            )

        final_state = lax.while_loop(_cond_fun, _body_fun, init_state)

        s = final_state.s
        su = jnp.asarray(2.0**-final_state.level, dtype=tdtype)

        sr = jax.nn.relu(r - s)
        # make sure su = sr + ru regardless of cancellation error
        ru = jax.nn.relu(su - sr)

        w_s, w_u, w_su = final_state.w_s_u_su

        if self.levy_area is SpaceTimeTimeLevyArea:
            # Based on Theorem 3.7 from the paper on
            # Single-seed generation of Brownian paths and integrals
            assert final_state.bhh_s_u_su is not None
            assert final_state.bkk_s_u_su is not None
            bhh_s, bhh_u, bhh_su = final_state.bhh_s_u_su
            bkk_s, bkk_u, bkk_su = final_state.bkk_s_u_su

            su3 = jnp.power(su, 3)
            sr_by_su = sr / su
            sr_by_su_3 = jnp.power(sr_by_su, 3)
            sr_by_su_5 = jnp.power(sr_by_su, 5)
            ru_by_su = ru / su
            sr_ru_by_su2 = sr_by_su * ru_by_su
            sr2 = jnp.square(sr)
            ru2 = jnp.square(ru)
            su2 = jnp.square(su)

            # compute the mean of (W_sr, H_sr, K_sr) conditioned on
            # (W_s, H_s, K_s, W_u, H_u, K_u)
            with jax.numpy_dtype_promotion("standard"):
                bb_mean = (6 * sr_ru_by_su2 / su) * bhh_su + (
                    120 * sr_ru_by_su2 * (0.5 - sr_by_su) / su2
                ) * bkk_su
                w_mean = sr_by_su * w_su + bb_mean
                h_mean = (sr_by_su**2 / su) * (bhh_su + (30 * ru_by_su / su) * bkk_su)
                k_mean = (sr_by_su_3 / su2) * bkk_su

                # compute the covariance matrix of (W_sr, H_sr, K_sr) conditioned on
                # (W_s, H_s, K_s, W_u, H_u, K_u)
                ww_cov = (
                    sr_by_su * ru_by_su * ((sr - ru) ** 4 + 4 * (sr2 * ru2))
                ) / su3
                wh_cov = -(sr_by_su_3 * ru_by_su * (sr2 - 3 * sr * ru + 6 * ru2)) / (
                    2 * su
                )
                wk_cov = (sr_by_su**4) * ru_by_su * (sr - ru) / 12
                hh_cov = (sr / 12) * (
                    1 - sr_by_su_3 * (sr2 + 2 * sr * ru + 16 * ru2) / su2
                )
                hk_cov = -(ru / 24) * sr_by_su_5
                kk_cov = (sr / 720) * (1.0 - sr_by_su_5)

            cov = jnp.array(
                [
                    [ww_cov, wh_cov, wk_cov],
                    [wh_cov, hh_cov, hk_cov],
                    [wk_cov, hk_cov, kk_cov],
                ]
            )

            if self._spline == "sqrt":
                # NOTE: jr.multivariate_normal is not compatible with jnp.float16,
                # so we need to cast to jnp.float32 before calling it.
                with jax.numpy_dtype_promotion("standard"):
                    dtype_atleast32 = jnp.result_type(dtype, jnp.float32)
                cov = jnp.asarray(cov, dtype_atleast32)
                hat_y = jr.multivariate_normal(
                    final_state.key,
                    jnp.zeros((3,), dtype_atleast32),
                    cov,
                    shape=shape,
                    dtype=dtype_atleast32,
                    method="svd",
                )
                hat_y = jnp.asarray(hat_y, dtype)

            elif self._spline == "zero":
                hat_y = jnp.zeros(shape=shape + (3,), dtype=dtype)
            else:
                raise ValueError(
                    f"When levy_area='space-time-time', only 'sqrt' and"
                    f" 'zero' splines are permitted, got {self._spline}."
                )

            hat_w_sr, hat_hh_sr, hat_kk_sr = [
                x.squeeze(axis=-1) for x in jnp.split(hat_y, 3, axis=-1)
            ]
            assert hat_w_sr.shape == hat_hh_sr.shape == hat_kk_sr.shape == shape

            w_sr = w_mean + hat_w_sr
            w_r = w_s + w_sr

            with jax.numpy_dtype_promotion("standard"):
                r_bb_s = r * w_s - s * w_r

                bhh_sr = sr * (h_mean + hat_hh_sr)
                bhh_r = bhh_s + bhh_sr + 0.5 * r_bb_s

                bkk_sr = sr2 * (k_mean + hat_kk_sr)
                bkk_r = (
                    bkk_s
                    + bkk_sr
                    + (sr / 2) * bhh_s
                    - (s / 2) * bhh_sr
                    + ((r - 2 * s) / 12) * r_bb_s
                )

                inverse_r = 1 / jnp.where(jnp.square(r) < jnp.finfo(r).eps, jnp.inf, r)
                hh_r = inverse_r * bhh_r
                kk_r = inverse_r**2 * bkk_r

            return _LevyVal(dt=r, W=w_r, H=hh_r, bar_H=bhh_r, K=kk_r, bar_K=bkk_r)

        elif self.levy_area is SpaceTimeLevyArea:
            # This is based on Theorem 6.1.4 of Foster's thesis (see above).

            assert final_state.bhh_s_u_su is not None
            bhh_s, bhh_u, bhh_su = final_state.bhh_s_u_su
            sr3 = jnp.power(sr, 3)
            ru3 = jnp.power(ru, 3)
            su3 = jnp.power(su, 3)

            # Here "quad" spline doesn't really exist, but we can still
            # compare "sqrt" and "zero" splines.
            if self._spline == "sqrt":
                key1, key2 = jr.split(final_state.key, 2)
                x1 = jr.normal(key1, shape, dtype)
                x2 = jr.normal(key2, shape, dtype)
            elif self._spline == "zero":
                x1 = jnp.zeros(shape, dtype)
                x2 = jnp.zeros(shape, dtype)
            else:
                raise ValueError(
                    f"When levy_area='SpaceTimeLevyArea', only 'sqrt' and"
                    f" 'zero' splines are permitted, got {self._spline}."
                )

            sr_ru_half = jnp.sqrt(sr * ru)
            d = jnp.sqrt(sr3 + ru3)
            d_prime = 1 / (2 * su * d)
            a = d_prime * sr3 * sr_ru_half
            b = d_prime * ru3 * sr_ru_half

            with jax.numpy_dtype_promotion("standard"):
                w_sr = (
                    sr / su * w_su + 6 * sr * ru / su3 * bhh_su + 2 * (a + b) / su * x1
                )
                w_r = w_s + w_sr
                c = jnp.sqrt(3 * sr3 * ru3) / (6 * d)
                bhh_sr = sr3 / su3 * bhh_su - a * x1 + c * x2
                bhh_r = bhh_s + bhh_sr + 0.5 * (r * w_s - s * w_r)

                inverse_r = 1 / jnp.where(jnp.abs(r) < jnp.finfo(r).eps, jnp.inf, r)
                hh_r = inverse_r * bhh_r

            return _LevyVal(dt=r, W=w_r, H=hh_r, bar_H=bhh_r, K=None, bar_K=None)

        elif self.levy_area is BrownianIncrement:
            with jax.numpy_dtype_promotion("standard"):
                w_mean = w_s + sr / su * w_su
                if self._spline == "sqrt":
                    z = jr.normal(final_state.key, shape, dtype)
                    bb = jnp.sqrt(sr * ru / su) * z
                elif self._spline == "quad":
                    z = jr.normal(final_state.key, shape, dtype)
                    bb = (sr * ru / su) * z
                elif self._spline == "zero":
                    bb = jnp.zeros(shape, dtype)
                else:
                    assert False
            w_r = w_mean + bb
            return _LevyVal(dt=r, W=w_r, H=None, bar_H=None, K=None, bar_K=None)

        else:
            assert False

    def _brownian_arch(
        self, _state: _State, shape, dtype
    ) -> tuple[
        RealScalarLike,
        FloatTriple,
        FloatDouble,
        tuple[PRNGKeyArray, PRNGKeyArray],
        Optional[FloatTriple],
        Optional[FloatDouble],
        Optional[FloatTriple],
        Optional[FloatDouble],
    ]:
        r"""For `t = (s+u)/2` evaluates `w_t` and (optionally) `bhh_t`
         conditioned on `w_s`, `w_u`, `bhh_s`, `bhh_u`, where
         `bhh_st` represents $\bar{H}_{s,t} \coloneqq (t-s) H_{s,t}$.
         To avoid cancellation errors, requires an input of `w_su`, `bhh_su`
         and also returns `w_st` and `w_tu` in addition to just `w_t`. Same for `bhh`
         if it is not None.
         Note that the inputs and outputs already contain `bkk`. These values are
         there for the sake of a future extension with "space-time-time" Lévy area
         and should be None for now.

        **Arguments:**

        - `_state`: The state of the Brownian tree
        - `shape`:
        - `dtype`:

        **Returns:**

        - `t`: midpoint time
        - `w_stu`: $(W_s, W_t, W_u)$
        - `w_st_tu`: $(W_{s,t}, W_{t,u})$
        - `keys`: a tuple of subinterval keys `(key_st, key_tu)`
        - `bhh_stu`: (optional) $(\bar{H}_s, \bar{H}_t, \bar{H}_u)$
        - `bhh_st_tu`: (optional) $(\bar{H}_{s,t}, \bar{H}_{t,u})$
        - `bkk_stu`: (optional) $(\bar{K}_s, \bar{K}_t, \bar{K}_u)$
        - `bkk_st_tu`: (optional) $(\bar{K}_{s,t}, \bar{K}_{t,u})$

        """
        key_st, midpoint_key, key_tu = jr.split(_state.key, 3)
        keys = (key_st, key_tu)
        tdtype = complex_to_real_dtype(dtype)
        su = jnp.asarray(2.0**-_state.level, dtype=tdtype)
        st = su / 2
        s = jnp.asarray(_state.s, dtype=tdtype)
        t = s + st
        root_su = jnp.sqrt(su)

        w_s, w_u, w_su = _state.w_s_u_su

        with jax.numpy_dtype_promotion("standard"):
            if self.levy_area is SpaceTimeTimeLevyArea:
                assert _state.bhh_s_u_su is not None
                assert _state.bkk_s_u_su is not None

                bhh_s, bhh_u, bhh_su = _state.bhh_s_u_su
                bkk_s, bkk_u, bkk_su = _state.bkk_s_u_su

                z1_key, z2_key, z3_key = jr.split(midpoint_key, 3)
                z1 = jr.normal(z1_key, shape, dtype)
                z2 = jr.normal(z2_key, shape, dtype)
                z3 = jr.normal(z3_key, shape, dtype)

                z = z1 * jnp.sqrt(su / 16)
                x1 = z2 * jnp.sqrt(su / 768)
                x2 = z3 * jnp.sqrt(su / 2880)

                su2 = su**2

                w_term1 = w_su / 2
                w_term2 = (3 / (2 * su)) * bhh_su + z
                w_st = w_term1 + w_term2
                w_tu = w_term1 - w_term2
                bhh_term1 = bhh_su / 8 - (st / 2) * z
                bhh_term2 = (15 / (8 * su)) * bkk_su + st * x1
                bhh_st = bhh_term1 + bhh_term2
                bhh_tu = bhh_term1 - bhh_term2
                bkk_term1 = bkk_su / 32 - (su2 / 8) * x1
                bkk_term2 = (su2 / 4) * x2
                bkk_st = bkk_term1 + bkk_term2
                bkk_tu = bkk_term1 - bkk_term2
                w_st_tu = (w_st, w_tu)
                bhh_st_tu = (bhh_st, bhh_tu)
                bkk_st_tu = (bkk_st, bkk_tu)

                w_t = w_s + w_st
                t_bb_s = t * w_s - s * w_t
                bhh_t = bhh_s + bhh_st + t_bb_s / 2
                bkk_t = (
                    bkk_s
                    + bkk_st
                    + (st / 2) * bhh_s
                    - (s / 2) * bhh_st
                    + ((t - 2 * s) / 12) * t_bb_s
                )

                w_stu = (w_s, w_t, w_u)
                bhh_stu = (bhh_s, bhh_t, bhh_u)
                bkk_stu = (bkk_s, bkk_t, bkk_u)

            elif self.levy_area is SpaceTimeLevyArea:
                assert _state.bhh_s_u_su is not None
                assert _state.bkk_s_u_su is None
                bhh_s, bhh_u, bhh_su = _state.bhh_s_u_su

                z1_key, z2_key = jr.split(midpoint_key, 2)
                z1 = jr.normal(z1_key, shape, dtype)
                z2 = jr.normal(z2_key, shape, dtype)
                z = z1 * (root_su / 4)
                n = z2 * jnp.sqrt(su / 12)

                w_term1 = w_su / 2
                w_term2 = (3 / (2 * su)) * bhh_su + z
                w_st = w_term1 + w_term2
                w_tu = w_term1 - w_term2
                w_st_tu = (w_st, w_tu)

                bhh_term1 = bhh_su / 8 - su / 4 * z
                bhh_term2 = (su / 4) * n
                bhh_st = bhh_term1 + bhh_term2
                bhh_tu = bhh_term1 - bhh_term2
                bhh_st_tu = (bhh_st, bhh_tu)

                w_t = w_s + w_st
                w_stu = (w_s, w_t, w_u)

                bhh_t = bhh_s + bhh_st + 0.5 * (t * w_s - s * w_t)
                bhh_stu = (bhh_s, bhh_t, bhh_u)
                bkk_stu = None
                bkk_st_tu = None

            elif self.levy_area is BrownianIncrement:
                assert _state.bhh_s_u_su is None
                assert _state.bkk_s_u_su is None
                mean = 0.5 * w_su
                w_term2 = (root_su / 2) * jr.normal(midpoint_key, shape, dtype)
                w_st = mean + w_term2
                w_tu = mean - w_term2
                w_st_tu = (w_st, w_tu)
                w_t = w_s + w_st
                w_stu = (w_s, w_t, w_u)
                bhh_stu, bhh_st_tu, bkk_stu, bkk_st_tu = None, None, None, None

            else:
                assert False

        return t, w_stu, w_st_tu, keys, bhh_stu, bhh_st_tu, bkk_stu, bkk_st_tu
