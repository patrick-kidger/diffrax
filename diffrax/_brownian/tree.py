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
# For more about space-time Levy area see Definition 4.2.1.
# For the midpoint rule for generating space-time Levy area see Theorem 6.1.6.
# For the general interpolation rule for space-time Levy area see Theorem 6.1.4.

FloatDouble: TypeAlias = tuple[Inexact[Array, " *shape"], Inexact[Array, " *shape"]]
FloatTriple: TypeAlias = tuple[
    Inexact[Array, " *shape"], Inexact[Array, " *shape"], Inexact[Array, " *shape"]
]
_Spline: TypeAlias = Literal["sqrt", "quad", "zero"]
_BrownianReturn = TypeVar("_BrownianReturn", bound=AbstractBrownianIncrement)


class _State(eqx.Module):
    level: IntScalarLike  # level of the tree
    s: RealScalarLike  # starting time of the interval
    w_s_u_su: FloatTriple  # W_s, W_u, W_{s,u}
    key: PRNGKeyArray
    bhh_s_u_su: Optional[FloatTriple]  # \bar{H}_s, _u, _{s,u}
    bkk_s_u_su: Optional[FloatTriple]  # \bar{K}_s, _u, _{s,u}


def _levy_diff(_, x0: tuple, x1: tuple) -> AbstractBrownianIncrement:
    r"""Computes $(W_{s,u}, H_{s,u})$ from $(W_s, \bar{H}_{s,u})$ and
    $(W_u, \bar{H}_u)$, where $\bar{H}_u = u * H_u$.

    **Arguments:**

    - `_`: unused, for the purposes of aligning the `jtu.tree_map`.
    - `x0`: `LevyVal` at time `s`.
    - `x1`: `LevyVal` at time `u`.

    **Returns:**

    `LevyVal(W_su, H_su)`
    """

    if len(x0) == 2:  # BM only case
        assert len(x1) == 2
        dt0, w0 = x0
        dt1, w1 = x1
        su = jnp.asarray(dt1 - dt0, dtype=complex_to_real_dtype(w0.dtype))
        return BrownianIncrement(dt=su, W=w1 - w0)

    elif len(x0) == 4:  # space-time levy area case
        assert len(x1) == 4
        dt0, w0, hh0, bhh0 = x0
        dt1, w1, hh1, bhh1 = x1

        w_su = w1 - w0
        su = jnp.asarray(dt1 - dt0, dtype=complex_to_real_dtype(w0.dtype))
        _su = jnp.where(jnp.abs(su) < jnp.finfo(su).eps, jnp.inf, su)
        inverse_su = 1 / _su
        with jax.numpy_dtype_promotion("standard"):
            u_bb_s = dt1 * w0 - dt0 * w1
            bhh_su = bhh1 - bhh0 - 0.5 * u_bb_s  # bhh_su = H_{s,u} * (u-s)
            hh_su = inverse_su * bhh_su
        return SpaceTimeLevyArea(dt=su, W=w_su, H=hh_su)
    else:
        assert False


def _make_levy_val(_, x: tuple) -> AbstractBrownianIncrement:
    if len(x) == 2:
        dt, w = x
        return BrownianIncrement(dt=dt, W=w)
    elif len(x) == 4:
        dt, w, hh, bhh = x
        return SpaceTimeLevyArea(dt=dt, W=w, H=hh)
    else:
        assert False


def _split_interval(
    pred: BoolScalarLike, x_stu: FloatTriple, x_st_tu: FloatDouble
) -> FloatTriple:
    x_s, x_t, x_u = x_stu
    x_st, x_tu = x_st_tu
    x_s = jnp.where(pred, x_t, x_s)
    x_u = jnp.where(pred, x_u, x_t)
    x_su = jnp.where(pred, x_tu, x_st)
    return x_s, x_u, x_su


class VirtualBrownianTree(AbstractBrownianPath):
    """Brownian simulation that discretises the interval `[t0, t1]` to tolerance `tol`.

    !!! info "Levy Area"

        Can be initialised with `levy_area` set to `diffrax.BrownianIncrement`, or
        `diffrax.SpaceTimeLevyArea`. If `levy_area=diffrax.SpaceTimeLevyArea`, then it
        also computes space-time Lévy area `H`. This is an additional source of
        randomness required for certain stochastic Runge--Kutta solvers; see
        [`diffrax.AbstractSRK`][] for more information.

        An error will be thrown during tracing if Lévy area is required but is not
        available.

        The choice here will impact the Brownian path, so even with the same key, the
        trajectory will be different depending on the value of `levy_area`.

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
        simulates space-time Levy area. This is due to Section 6.1 and Theorem 6.1.6 of
        ```bibtex
        @phdthesis{foster2020a,
          publisher = {University of Oxford},
          school = {University of Oxford},
          title = {Numerical approximations for stochastic differential equations},
          author = {Foster, James M.},
          year = {2020}
        }
        ```

        In addition, the implementation here is a further improvement on these by using
        an interpolation method which ensures the conditional 2nd moments are correct.
    """

    t0: RealScalarLike
    t1: RealScalarLike
    tol: RealScalarLike
    shape: PyTree[jax.ShapeDtypeStruct] = eqx.field(static=True)
    levy_area: type[Union[BrownianIncrement, SpaceTimeLevyArea]] = eqx.field(
        static=True
    )
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
            Union[BrownianIncrement, SpaceTimeLevyArea]
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
            levy_out = jtu.tree_map(_levy_diff, self.shape, levy_0, levy_1)

        levy_out = levy_tree_transpose(self.shape, levy_out)
        # now map [0,1] back onto [self.t0, self.t1]
        levy_out = self._denormalise_bm_inc(levy_out)
        assert isinstance(levy_out, (BrownianIncrement, SpaceTimeLevyArea))
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
    ) -> Union[
        tuple[RealScalarLike, Array], tuple[RealScalarLike, Array, Array, Array]
    ]:
        shape, dtype = struct.shape, struct.dtype
        tdtype = complex_to_real_dtype(dtype)

        t0 = jnp.zeros((), tdtype)
        r = jnp.asarray(r, tdtype)

        if self.levy_area is SpaceTimeLevyArea:
            state_key, init_key_w, init_key_la = jr.split(key, 3)
            bhh_1 = jr.normal(init_key_la, shape, dtype) / math.sqrt(12)
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
                _w_inc,
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

            _w = _split_interval(_cond, _w_stu, _w_inc)
            _bkk = None
            if self.levy_area is SpaceTimeLevyArea:
                assert _bhh_stu is not None and _bhh_st_tu is not None
                _bhh = _split_interval(_cond, _bhh_stu, _bhh_st_tu)
            elif self.levy_area is BrownianIncrement:
                _bhh = None
            else:
                assert False

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
        su = 2.0**-final_state.level

        sr = jax.nn.relu(r - s)
        # make sure su = sr + ru regardless of cancellation error
        ru = jax.nn.relu(su - sr)

        w_s, w_u, w_su = final_state.w_s_u_su
        if self.levy_area is SpaceTimeLevyArea:
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
            return r, w_r

        else:
            assert False

        return r, w_r, hh_r, bhh_r

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
         there for the sake of a future extension with "space-time-time" Levy area
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
        su = 2.0**-_state.level
        st = su / 2
        s = _state.s
        t = s + st
        root_su = jnp.sqrt(su)

        w_s, w_u, w_su = _state.w_s_u_su

        if self.levy_area is SpaceTimeLevyArea:
            assert _state.bhh_s_u_su is not None
            assert _state.bkk_s_u_su is None
            bhh_s, bhh_u, bhh_su = _state.bhh_s_u_su

            z1_key, z2_key = jr.split(midpoint_key, 2)
            z1 = jr.normal(z1_key, shape, dtype)
            z2 = jr.normal(z2_key, shape, dtype)
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
            w_stu = (w_s, w_t, w_u)
            with jax.numpy_dtype_promotion("standard"):
                bhh_t = bhh_s + bhh_st + 0.5 * (t * w_s - s * w_t)
            bhh_stu = (bhh_s, bhh_t, bhh_u)
            bkk_stu = None
            bkk_st_tu = None

        elif self.levy_area is BrownianIncrement:
            assert _state.bhh_s_u_su is None
            assert _state.bkk_s_u_su is None
            mean = 0.5 * w_su
            w_term2 = root_su / 2 * jr.normal(midpoint_key, shape, dtype)
            w_st = mean + w_term2
            w_tu = mean - w_term2
            w_st_tu = (w_st, w_tu)
            w_t = w_s + w_st
            w_stu = (w_s, w_t, w_u)
            bhh_stu, bhh_st_tu, bkk_stu, bkk_st_tu = None, None, None, None

        else:
            assert False

        return t, w_stu, w_st_tu, keys, bhh_stu, bhh_st_tu, bkk_stu, bkk_st_tu
