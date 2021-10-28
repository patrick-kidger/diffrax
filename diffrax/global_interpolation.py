import functools as ft
from typing import Optional, Tuple, Type

import equinox as eqx
import jax
import jax.lax as lax
import jax.numpy as jnp

from .custom_types import Array, DenseInfo, PyTree, Scalar
from .local_interpolation import AbstractLocalInterpolation
from .misc import fill_forward, unvmap
from .path import AbstractPath


class AbstractGlobalInterpolation(AbstractPath):
    ts: Array["times"]  # noqa: F821

    def __post_init__(self):
        assert self.ts.ndim == 1

    def _interpret_t(self, t: Scalar, left: bool) -> Tuple[Scalar, Scalar]:
        maxlen = self.ts.shape[0] - 2
        index = jnp.searchsorted(self.ts, t, side="left" if left else "right")
        index = jnp.clip(index - 1, a_min=0, a_max=maxlen)
        # Will never access the final element of `ts`; this is correct behaviour.
        fractional_part = t - self.ts[index]
        return index, fractional_part

    @property
    def t0(self):
        return self.ts[0]

    @property
    def t1(self):
        return self.ts[-1]


class LinearInterpolation(AbstractGlobalInterpolation):
    ys: PyTree["times", ...]  # noqa: F821

    def __post_init__(self):
        def _assert(_ys):
            assert _ys.shape[0] == self.ts.shape[0]

        jax.tree_map(_assert, self.ys)

    def derivative(self, t: Scalar, left: bool = True) -> PyTree:
        index, _ = self._interpret_t(t, left)

        def _index(_ys):
            return (_ys[index + 1] - _ys[index]) / (self.ts[index + 1] - self.ts[index])

        return jax.tree_map(_index, self.ys)

    def evaluate(
        self, t0: Scalar, t1: Optional[Scalar] = None, left: bool = True
    ) -> PyTree:
        if t1 is not None:
            return self.evaluate(t1, left=left) - self.evaluate(t0, left=left)
        index, fractional_part = self._interpret_t(t0, left)

        def _index(_ys):
            return _ys[index]

        prev_ys = jax.tree_map(_index, self.ys)
        next_ys = jax.tree_map(lambda _ys: _ys[index + 1], self.ys)
        prev_t = self.ts[index]
        next_t = self.ts[index + 1]
        diff_t = next_t - prev_t

        def _combine(_prev_ys, _next_ys):
            return _prev_ys + (_next_ys - _prev_ys) * (fractional_part / diff_t)

        return jax.tree_map(_combine, prev_ys, next_ys)


class CubicInterpolation(AbstractGlobalInterpolation):
    # d, c, b, a
    coeffs: Tuple[
        PyTree["times - 1", ...],  # noqa: F821
        PyTree["times - 1", ...],  # noqa: F821
        PyTree["times - 1", ...],  # noqa: F821
        PyTree["times - 1", ...],  # noqa: F821
    ]

    def __post_init__(self):
        def _assert(d, c, b, a):
            assert d.shape[0] + 1 == self.ts.shape[0]
            assert c.shape[0] + 1 == self.ts.shape[0]
            assert b.shape[0] + 1 == self.ts.shape[0]
            assert a.shape[0] + 1 == self.ts.shape[0]

        jax.tree_map(_assert, *self.coeffs)

    def derivative(self, t: Scalar, left: bool = True) -> PyTree:
        index, f = self._interpret_t(t, left)

        def _index(d, c, b, _):
            return b[index] + 2 * f * c[index] + 3 * f * d[index]

        return jax.tree_map(_index, *self.coeffs)

    def evaluate(
        self, t0: Scalar, t1: Optional[Scalar] = None, left: bool = True
    ) -> PyTree:
        if t1 is not None:
            return self.evaluate(t1, left=left) - self.evaluate(t0, left=left)
        index, f = self._interpret_t(t0, left)

        def _index(d, c, b, a):
            return a[index] + f * (b[index] + f * (c[index] + f * d[index]))

        return jax.tree_map(_index, *self.coeffs)


class DenseInterpolation(AbstractGlobalInterpolation):
    infos: DenseInfo
    direction: Scalar
    unravel_y: jax.tree_util.Partial
    interpolation_cls: Type[AbstractLocalInterpolation] = eqx.static_field()

    def __post_init__(self):
        def _assert(_d):
            assert _d.shape[0] + 1 == self.ts.shape[0]

        jax.tree_map(_assert, self.infos)

    def _get_local_interpolation(self, t: Scalar, left: bool):
        index, _ = self._interpret_t(t, left)
        prev_t = self.ts[index]
        next_t = self.ts[index + 1]

        def _index(_d):
            return _d[index]

        infos = jax.tree_map(_index, self.infos)
        return self.interpolation_cls(t0=prev_t, t1=next_t, **infos)

    def derivative(self, t: Scalar, left: bool = True) -> PyTree:
        # Passing `left` doesn't matter on a local interpolation, which is globally
        # continuous.
        t = t * self.direction
        out = self._get_local_interpolation(t, left).derivative(t)
        out = out * self.direction
        return self.unravel_y(out)

    def evaluate(
        self, t0: Scalar, t1: Optional[Scalar] = None, left: bool = True
    ) -> PyTree:
        if t1 is not None:
            return self.evaluate(t1, left=left) - self.evaluate(t0, left=left)
        t0 = t0 * self.direction
        # Passing `left` doesn't matter on a local interpolation, which is globally
        # continuous.
        return self.unravel_y(self._get_local_interpolation(t0, left).evaluate(t0))

    @property
    def t0(self):
        return self.ts[0] * self.direction

    @property
    def t1(self):
        return self.ts[-1] * self.direction


#
# The following interpolation routines are quite involved, as they are designed to
# handle NaNs (representing missing data), by interpolating over them.
#


def _check_ts(ts: Array["times"]) -> None:  # noqa: F821
    if ts.ndim != 1:
        raise ValueError(f"ts must be 1-dimensional; got {ts.ndim}.")
    if ts.shape[0] < 2:
        raise ValueError(f"ts must be of length at least 2; got {ts.shape[0]}")
    if not isinstance(ts, jax.core.Tracer) and jnp.any(unvmap(ts[:-1] >= ts[1:])):
        # Also catches any NaN times.
        raise ValueError("ts must be monotonically strictly increasing.")


def _interpolation_reverse(
    carry: Tuple[Array["channels"], Array["channels"]],  # noqa: F821
    value: Tuple[Array["channels"], Array["channels"]],  # noqa: F821
) -> Tuple[
    Tuple[Array["channels"], Array["channels"]],  # noqa: F821
    Tuple[Array["channels"], Array["channels"]],  # noqa: F821
]:
    next_ti, next_yi = carry
    ti, yi = value
    cond = jnp.isnan(yi)
    ti = jnp.where(cond, next_ti, ti)
    yi = jnp.where(cond, next_yi, yi)
    return (ti, yi), (ti, yi)


def _linear_interpolation_forward(
    carry: Tuple[Array["channels"], Array["channels"]],  # noqa: F821
    value: Tuple[
        Array["channels"],  # noqa: F821
        Array["channels"],  # noqa: F821
        Array["channels"],  # noqa: F821
        Array["channels"],  # noqa: F821
    ],
) -> Tuple[
    Tuple[Array["channels"], Array["channels"]], Array["channels"]  # noqa: F821
]:

    prev_ti, prev_yi = carry
    ti, yi, next_ti, next_yi = value
    cond = jnp.isnan(yi)
    carry_ti = jnp.where(cond, prev_ti, ti)
    carry_yi = jnp.where(cond, prev_yi, yi)

    _div = jnp.where(cond, next_ti - carry_ti, 1)
    y = carry_yi + (next_yi - carry_yi) * (ti - carry_ti) / _div

    return (carry_ti, carry_yi), y


def _linear_interpolation_impl(fill_forward_nans_at_end):
    def _linear_interpolation_impl_(operand):
        y0, ts, ys = operand

        _, (next_ts, next_ys) = lax.scan(
            _interpolation_reverse, (ts[-1], ys[-1]), (ts, ys), reverse=True
        )
        if fill_forward_nans_at_end:
            next_ys = fill_forward(next_ys)
        _, ys = lax.scan(
            _linear_interpolation_forward, (ts[0], y0), (ts, ys, next_ts, next_ys)
        )
        return ys

    return _linear_interpolation_impl_


@ft.partial(jax.jit, static_argnums=(0, 1))
def _linear_interpolation(
    rectilinear: Optional[int],
    fill_forward_nans_at_end: bool,
    ts: Array["times"],  # noqa: F821
    ys: Array["times", "channels"],  # noqa: F821
    replace_nans_at_start: Optional[Array["channels"]] = None,  # noqa: F821
) -> Array["times", "channels"]:  # noqa: F821
    if ts.ndim != 1:
        raise ValueError(f"ts should have 1 dimension, got {ts.ndim}.")
    if ys.ndim != 2:
        raise ValueError(f"ys should have 2 dimensions, got {ys.ndim}.")

    if rectilinear is not None:
        ys = fill_forward(ys)
        ys = jnp.repeat(ys, 2)
        ys = ys.at[:-1, rectilinear].set(ys.at[1:], rectilinear)
        ys = ys[:-1]

    # Do the check ourselves prior to broadcasting, for an informative error message.
    if ts.shape[0] != ys.shape[0]:
        if rectilinear is None:
            raise ValueError(
                "Must have 2 * ts.shape[0] = ys.shape[0] + 1 when using rectilinear "
                "interpolation."
            )
        else:
            raise ValueError("ts and ys must consist of elements of the same length.")
    ts = jnp.broadcast_to(ts[:, None], ys.shape)

    if replace_nans_at_start is None:
        y0 = ys[0]
    else:
        y0 = jnp.broadcast_to(replace_nans_at_start, ys[0].shape)
    cond = jnp.any(jnp.isnan(ys))
    operand = (y0, ts, ys)
    ys = lax.cond(
        cond,
        _linear_interpolation_impl(fill_forward_nans_at_end),
        lambda _: ys,
        operand,
    )
    return ys


def linear_interpolation(
    ts: Array["times"],  # noqa: F821
    ys: PyTree,
    *,
    rectilinear: Optional[int] = None,
    fill_forward_nans_at_end: bool = False,
    replace_nans_at_start: Optional[PyTree] = None,
) -> PyTree:
    _check_ts(ts)
    fn = ft.partial(_linear_interpolation, rectilinear, fill_forward_nans_at_end, ts)
    if replace_nans_at_start is None:
        return jax.tree_map(fn, ys)
    else:
        return jax.tree_map(fn, ys, replace_nans_at_start)


def _hermite_forward(
    carry: Tuple[Scalar, Array["channels"], Array["channels"]],  # noqa: F821
    value: Tuple[Scalar, Array["channels"]],  # noqa: F821
) -> Tuple[
    Tuple[Array["channels"], Array["channels"], Array["channels"]],  # noqa: F821
    Tuple[Array["channels"], Array["channels"], Array["channels"]],  # noqa: F821
]:

    prev_ti, prev_yi, prev_deriv_i = carry
    ti, yi = value
    deriv_i = (yi - prev_yi) / (ti - prev_ti)
    cond = jnp.isnan(yi)
    # deriv_cond is checked separately, as a trick to handle the first iteration: on
    # that we initialise prev_ti = nan, so that the check here fails and prev_deriv_i
    # specified is used.
    deriv_cond = jnp.isnan(deriv_i)
    carry_ti = jnp.where(cond, prev_ti, ti)
    carry_yi = jnp.where(cond, prev_yi, yi)
    carry_deriv_i = jnp.where(deriv_cond, prev_deriv_i, deriv_i)

    return (carry_ti, carry_yi, carry_deriv_i), (carry_ti, carry_yi, carry_deriv_i)


def _hermite_coeffs(t0, y0, deriv0, t1, y1):
    t_diff = t1 - t0
    deriv1 = (y1 - y0) / t_diff
    d_deriv = deriv1 - deriv0

    a = y0
    b = deriv0
    c = 2 * d_deriv / t_diff
    d = -d_deriv / t_diff ** 2

    return d, c, b, a


def _hermite_nan(operand):
    prev_ti, prev_yi, prev_deriv_i, ti, _, _, next_ti, next_yi = operand
    d, c, b, a = _hermite_coeffs(prev_ti, prev_yi, prev_deriv_i, next_ti, next_yi)
    ts = jnp.stack([prev_ti, next_ti])
    interpolation = CubicInterpolation(
        ts=ts, coeffs=(d[None], c[None], b[None], a[None])
    )
    interp_yi = interpolation.evaluate(ti)
    interp_deriv_i = interpolation.derivative(ti)
    return _hermite_coeffs(ti, interp_yi, interp_deriv_i, next_ti, next_yi)


def _hermite_no_nan(operand):
    _, _, _, ti, yi, deriv_i, next_ti, next_yi = operand
    return _hermite_coeffs(ti, yi, deriv_i, next_ti, next_yi)


def _hermite_cond(prev_ti, prev_yi, prev_deriv_i, ti, yi, deriv_i, next_ti, next_yi):
    operand = (prev_ti, prev_yi, prev_deriv_i, ti, yi, deriv_i, next_ti, next_yi)
    return lax.cond(jnp.isnan(yi), _hermite_nan, _hermite_no_nan, operand)


@ft.partial(jax.jit, static_argnums=0)
def _backward_hermite_coefficients(
    fill_forward_nans_at_end: bool,
    ts: Array["times"],  # noqa: F821
    ys: Array["times", "channels"],  # noqa: F821
    deriv0: Optional[Array["channels"]] = None,  # noqa: F821
    replace_nans_at_start: Optional[Array["channels"]] = None,  # noqa: F821
) -> Tuple[
    Array["channels"],  # noqa: F821
    Array["channels"],  # noqa: F821
    Array["channels"],  # noqa: F821
    Array["channels"],  # noqa: F821
]:  # noqa: F821
    if ts.ndim != 1:
        raise ValueError(f"`ts` should have 1 dimension, got {ts.ndim}.")
    if ys.ndim != 2:
        raise ValueError(f"`ys` should have 2 dimensions, got {ys.ndim}.")
    if deriv0 is not None and deriv0.shape != (ys.shape[1],):
        raise ValueError(
            "deriv0 should either be `None` or should have the same "
            "number of channels as `ys`."
        )
    ts = jnp.broadcast_to(ts[:, None], ys.shape)

    _, (next_ts, next_ys) = lax.scan(
        _interpolation_reverse, (ts[-1], ys[-1]), (ts[1:], ys[1:]), reverse=True
    )

    if fill_forward_nans_at_end:
        next_ys = fill_forward(next_ys)

    if deriv0 is None:
        deriv0 = (next_ys[0] - ys[0]) / (next_ts[0] - ts[0])

    t0 = jnp.full_like(ys[0], fill_value=jnp.nan)
    if replace_nans_at_start is None:
        y0 = ys[0]
    else:
        y0 = jnp.broadcast_to(replace_nans_at_start, ys[0].shape)
    _, (_ts, _ys, _derivs) = lax.scan(_hermite_forward, (t0, y0, deriv0), (ts, ys))

    prev_ts = _ts[:-1]
    prev_ys = _ys[:-1]
    prev_derivs = _derivs[:-1]
    ts = ts[:-1]
    ys = ys[:-1]
    derivs = _derivs[1:]

    ds, cs, bs, as_ = jax.vmap(jax.vmap(_hermite_cond))(
        prev_ts, prev_ys, prev_derivs, ts, ys, derivs, next_ts, next_ys
    )

    return ds, cs, bs, as_


def backward_hermite_coefficients(
    ts: Array["times"],  # noqa: F821
    ys: PyTree,
    *,
    deriv0: Optional[PyTree] = None,
    fill_forward_nans_at_end: bool = False,
    replace_nans_at_start: Optional[PyTree] = None,
) -> Tuple[PyTree, PyTree, PyTree, PyTree]:
    _check_ts(ts)
    fn = ft.partial(_backward_hermite_coefficients, fill_forward_nans_at_end, ts)
    if deriv0 is None:
        if replace_nans_at_start is None:
            coeffs = jax.tree_map(fn, ys)
        else:
            _fn = lambda ys, replace_nans_at_start: fn(ys, None, replace_nans_at_start)
            coeffs = jax.tree_map(_fn, ys, replace_nans_at_start)
    else:
        if replace_nans_at_start is None:
            coeffs = jax.tree_map(fn, ys, deriv0)
        else:
            coeffs = jax.tree_map(fn, ys, deriv0, replace_nans_at_start)
    ys_treedef = jax.tree_structure(ys)
    coeffs_treedef = jax.tree_structure((0, 0, 0, 0))
    return jax.tree_transpose(ys_treedef, coeffs_treedef, coeffs)
