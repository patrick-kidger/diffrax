import functools as ft
from typing import Optional, Tuple, Type

import equinox as eqx
import equinox.internal as eqxi
import jax
import jax.lax as lax
import jax.numpy as jnp
import jax.tree_util as jtu
from equinox.internal import ω

from .custom_types import Array, DenseInfos, Int, PyTree, Scalar
from .local_interpolation import AbstractLocalInterpolation
from .misc import fill_forward, left_broadcast_to
from .path import AbstractPath


class AbstractGlobalInterpolation(AbstractPath):
    ts: Array["times"]  # noqa: F821

    def __post_init__(self):
        if self.ts.ndim != 1:
            raise ValueError("`ts` must be one dimensional.")

    def _ts_size(self):
        return self.ts.shape[0]

    def _interpret_t(self, t: Scalar, left: bool) -> Tuple[Scalar, Scalar]:
        maxlen = self._ts_size() - 2
        index = jnp.searchsorted(self.ts, t, side="left" if left else "right")
        index = jnp.clip(index - 1, a_min=0, a_max=maxlen)
        # Will never access the final element of `ts`; this is correct behaviour.
        fractional_part = t - self.ts[index]
        return index, fractional_part

    @property
    def t0(self):
        """The start of the interval over which the interpolation is defined."""
        return self.ts[0]

    @property
    def t1(self):
        """The end of the interval over which the interpolation is defined."""
        return self.ts[-1]


class LinearInterpolation(AbstractGlobalInterpolation):
    """Linearly interpolates some data `ys` over the interval $[t_0, t_1]$ with knots
    at `ts`.

    !!! warning

        If using `LinearInterpolation` as part of a [`diffrax.ControlTerm`][], then the
        vector field will make a jump every time one of the knots `ts` is passed. If
        using an adaptive step size controller such as [`diffrax.PIDController`][],
        then this means the controller should be informed about the jumps, so that it
        can handle them appropriately:

        ```python
        ts = ...
        interp = LinearInterpolation(ts=ts, ...)
        term = ControlTerm(..., control=interp)
        stepsize_controller = PIDController(..., jump_ts=ts)
        ```
    """

    ys: PyTree[Array["times", ...]]  # noqa: F821

    def __post_init__(self):
        def _check(_ys):
            if _ys.shape[0] != self.ts.shape[0]:
                raise ValueError(
                    "Must have ts.shape[0] == ys.shape[0], that is to say the same "
                    "number of entries along the timelike dimension."
                )

        jtu.tree_map(_check, self.ys)

    @eqx.filter_jit
    def evaluate(
        self, t0: Scalar, t1: Optional[Scalar] = None, left: bool = True
    ) -> PyTree:
        r"""Evaluate the linear interpolation.

        **Arguments:**

        - `t0`: Any point in $[t_0, t_1]$ to evaluate the interpolation at.
        - `t1`: If passed, then the increment from `t1` to `t0` is evaluated instead.
        - `left`: Across jump points: whether to treat the path as left-continuous
            or right-continuous. [In practice linear interpolation is always continuous
            except around `NaN`s.]

        !!! faq "FAQ"

            Note that we use $t_0$ and $t_1$ to refer to the overall interval, as
            obtained via `instance.t0` and `instance.t1`. We use `t0` and `t1` to refer
            to some subinterval of $[t_0, t_1]$. This is an API that is used for
            consistency with the rest of the package, and just happens to be a little
            confusing here.

        **Returns:**

        If `t1` is not passed:

        The interpolation of the data. Suppose $t_j < t < t_{j+1}$, where $t$ is `t0`
        and $t_j$ and $t_{j+1}$ are some element of `ts` as passed in `__init__`.
        Then the value returned is
        $y_j + (y_{j+1} - y_j)\frac{t - t_j}{t_{j+1} - t_j}$.

        If `t1` is passed:

        As above, with $t$ taken to be both `t0` and `t1`, and the increment between
        them returned.
        """

        if t1 is not None:
            return self.evaluate(t1, left=left) - self.evaluate(t0, left=left)
        index, fractional_part = self._interpret_t(t0, left)

        def _index(_ys):
            return _ys[index]

        prev_ys = jtu.tree_map(_index, self.ys)
        next_ys = (self.ys**ω)[index + 1].ω
        prev_t = self.ts[index]
        next_t = self.ts[index + 1]
        diff_t = next_t - prev_t

        return (
            prev_ys**ω + (next_ys**ω - prev_ys**ω) * (fractional_part / diff_t)
        ).ω

    @eqx.filter_jit
    def derivative(self, t: Scalar, left: bool = True) -> PyTree:
        r"""Evaluate the derivative of the linear interpolation. Essentially equivalent
        to `jax.jvp(self.evaluate, (t,), (jnp.ones_like(t),))`.

        **Arguments:**

        - `t`: Any point in $[t_0, t_1]$ to evaluate the derivative at.
        - `left`: Whether to obtain the left-derivative or right-derivative at that
            point.

        **Returns:**

        The derivative of the interpolation of the data. Suppose $t_j < t < t_{j+1}$,
        where $t_j$ and $t_{j+1}$ are some elements of `ts` passed in `__init__`. Then
        the value returned is $\frac{y_{j+1} - y_j}{t_{j+1} - t_j}$.
        """

        index, _ = self._interpret_t(t, left)

        return (
            (ω(self.ys)[index + 1] - ω(self.ys)[index])
            / (self.ts[index + 1] - self.ts[index])
        ).ω


LinearInterpolation.__init__.__doc__ = """**Arguments:**

- `ts`: Some increasing collection of times.
- `ys`: The value of the data at those times.

Note that if `ys` has any missing data then you may wish to use
[`diffrax.linear_interpolation`][] or [`diffrax.rectilinear_interpolation`][] first to
interpolate over these.
"""


class CubicInterpolation(AbstractGlobalInterpolation):
    """Piecewise cubic spline interpolation over the interval $[t_0, t_1]$."""

    # d, c, b, a
    coeffs: Tuple[
        PyTree["times - 1", ...],  # noqa: F821
        PyTree["times - 1", ...],  # noqa: F821
        PyTree["times - 1", ...],  # noqa: F821
        PyTree["times - 1", ...],  # noqa: F821
    ]

    def __post_init__(self):
        def _check(d, c, b, a):
            error_msg = (
                "Each cubic coefficient must have `times - 1` entries, where "
                "`times = self.ts.shape[0]`."
            )
            if d.shape[0] + 1 != self.ts.shape[0]:
                raise ValueError(error_msg)
            if c.shape[0] + 1 != self.ts.shape[0]:
                raise ValueError(error_msg)
            if b.shape[0] + 1 != self.ts.shape[0]:
                raise ValueError(error_msg)
            if a.shape[0] + 1 != self.ts.shape[0]:
                raise ValueError(error_msg)

        jtu.tree_map(_check, *self.coeffs)

    @eqx.filter_jit
    def evaluate(
        self, t0: Scalar, t1: Optional[Scalar] = None, left: bool = True
    ) -> PyTree:
        r"""Evaluate the cubic interpolation.

        **Arguments:**

        - `t0`: Any point in $[t_0, t_1]$ to evaluate the interpolation at.
        - `t1`: If passed, then the increment from `t1` to `t0` is evaluated instead.
        - `left`: Across jump points: whether to treat the path as left-continuous
            or right-continuous. [In practice cubic interpolation is always continuous
            except around `NaN`s.]

        !!! faq "FAQ"

            Note that we use $t_0$ and $t_1$ to refer to the overall interval, as
            obtained via `instance.t0` and `instance.t1`. We use `t0` and `t1` to refer
            to some subinterval of $[t_0, t_1]$. This is an API that is used for
            consistency with the rest of the package, and just happens to be a little
            confusing here.

        **Returns:**

        If `t1` is not passed:

        The interpolation of the data at `t0`.

        If `t1` is passed:

        The increment between `t0` and `t1`.
        """

        if t1 is not None:
            return self.evaluate(t1, left=left) - self.evaluate(t0, left=left)
        index, frac = self._interpret_t(t0, left)

        d, c, b, a = self.coeffs

        return (
            ω(a)[index]
            + frac * (ω(b)[index] + frac * (ω(c)[index] + frac * ω(d)[index]))
        ).ω

    @eqx.filter_jit
    def derivative(self, t: Scalar, left: bool = True) -> PyTree:
        r"""Evaluate the derivative of the cubic interpolation. Essentially equivalent
        to `jax.jvp(self.evaluate, (t,), (jnp.ones_like(t),))`.


        **Arguments:**

        - `t`: Any point in $[t_0, t_1]$ to evaluate the derivative at.
        - `left`: Whether to obtain the left-derivative or right-derivative at that
            point. [In practice cubic interpolation is always continuously
            differentiable except around `NaN`s.]

        **Returns:**

        The derivative of the interpolation of the data.
        """

        index, frac = self._interpret_t(t, left)

        d, c, b, _ = self.coeffs

        return (ω(b)[index] + frac * (2 * ω(c)[index] + frac * 3 * ω(d)[index])).ω


CubicInterpolation.__init__.__doc__ = """**Arguments:**

- `ts`: Some increasing collection of times.
- `coeffs`: The coefficients at all but the last time.

Any kind of spline (natural, ...) may be used; simply pass the appropriate
coefficients.

In practice a good choice is typically "cubic Hermite splines with backward
differences", introduced in [this paper](https://arxiv.org/abs/2106.11028). Such
coefficients can be obtained using [`diffrax.backward_hermite_coefficients`][].

Letting `d, c, b, a = coeffs`, then for all `t` in the interval from `ts[i]` to
`ts[i + 1]` the interpolation is defined as
```python
d[i] * (t - ts[i]) ** 3 + c[i] * (t - ts[i]) ** 2 + b[i] * (t - ts[i]) + a[i]
```
"""


class DenseInterpolation(AbstractGlobalInterpolation):
    ts_size: Int
    infos: DenseInfos
    direction: Scalar
    interpolation_cls: Type[AbstractLocalInterpolation] = eqx.static_field()

    def __post_init__(self):
        def _check(_infos):
            assert _infos.shape[0] + 1 == self.ts.shape[0]

        jtu.tree_map(_check, self.infos)

    # DenseInterpolations typically get `ts` and `infos` that are way longer than they
    # need to be, and padded with `nan`s. This means the normal way of measuring how
    # many entries we have - ts.shape[0] - won't be correct.
    def _ts_size(self):
        return self.ts_size

    def _get_local_interpolation(self, t: Scalar, left: bool):
        index, _ = self._interpret_t(t, left)
        prev_t = self.ts[index]
        next_t = self.ts[index + 1]
        infos = ω(self.infos)[index].ω
        return self.interpolation_cls(t0=prev_t, t1=next_t, **infos)

    @eqx.filter_jit
    def evaluate(
        self, t0: Scalar, t1: Optional[Scalar] = None, left: bool = True
    ) -> PyTree:
        if t1 is not None:
            return self.evaluate(t1, left=left) - self.evaluate(t0, left=left)
        t0 = t0 * self.direction
        # Passing `left` doesn't matter on a local interpolation, which is globally
        # continuous.
        return self._get_local_interpolation(t0, left).evaluate(t0)

    @eqx.filter_jit
    def derivative(self, t: Scalar, left: bool = True) -> PyTree:
        # Passing `left` doesn't matter on a local interpolation, which is globally
        # continuous.
        t = t * self.direction
        out = self._get_local_interpolation(t, left).derivative(t)
        return (self.direction * out**ω).ω

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
        raise ValueError(f"`ts` must be 1-dimensional; got {ts.ndim}.")
    if ts.shape[0] < 2:
        raise ValueError(f"`ts` must be of length at least 2; got {ts.shape[0]}")
    # Also catches any NaN times.
    ts = eqxi.error_if(
        ts, ts[:-1] >= ts[1:], "`ts` must be monotonically strictly increasing."
    )
    return ts


def _interpolation_reverse(
    carry: Tuple[Array["channels":...], Array["channels":...]],  # noqa: F821
    value: Tuple[Array["channels":...], Array["channels":...]],  # noqa: F821
) -> Tuple[
    Tuple[Array["channels":...], Array["channels":...]],  # noqa: F821
    Tuple[Array["channels":...], Array["channels":...]],  # noqa: F821
]:
    next_ti, next_yi = carry
    ti, yi = value
    cond = jnp.isnan(yi)
    ti = jnp.where(cond, next_ti, ti)
    yi = jnp.where(cond, next_yi, yi)
    return (ti, yi), (ti, yi)


def _linear_interpolation_forward(
    carry: Tuple[Array["channels":...], Array["channels":...]],  # noqa: F821
    value: Tuple[
        Array["channels":...],  # noqa: F821
        Array["channels":...],  # noqa: F821
        Array["channels":...],  # noqa: F821
        Array["channels":...],  # noqa: F821
    ],
) -> Tuple[
    Tuple[Array["channels":...], Array["channels":...]],  # noqa: F821
    Array["channels":...],  # noqa: F821
]:

    prev_ti, prev_yi = carry
    ti, yi, next_ti, next_yi = value
    cond = jnp.isnan(yi)
    carry_ti = jnp.where(cond, prev_ti, ti)
    carry_yi = jnp.where(cond, prev_yi, yi)

    _div = jnp.where(cond, next_ti - carry_ti, 1)
    y = carry_yi + (next_yi - carry_yi) * (ti - carry_ti) / _div

    return (carry_ti, carry_yi), y


def _linear_interpolation(
    fill_forward_nans_at_end: bool,
    ts: Array["times"],  # noqa: F821
    ys: Array["times", "channels":...],  # noqa: F821
    replace_nans_at_start: Optional[Array["channels":...]] = None,  # noqa: F821
) -> Array["times", "channels":...]:  # noqa: F821

    ts = left_broadcast_to(ts, ys.shape)

    if replace_nans_at_start is None:
        y0 = ys[0]
    else:
        y0 = jnp.broadcast_to(replace_nans_at_start, ys[0].shape)

    _, (next_ts, next_ys) = lax.scan(
        _interpolation_reverse, (ts[-1], ys[-1]), (ts, ys), reverse=True
    )
    if fill_forward_nans_at_end:
        next_ys = fill_forward(next_ys)
    _, ys = lax.scan(
        _linear_interpolation_forward, (ts[0], y0), (ts, ys, next_ts, next_ys)
    )
    return ys


@eqx.filter_jit
def linear_interpolation(
    ts: Array["times"],  # noqa: F821
    ys: PyTree["times", ...],  # noqa: F821
    *,
    fill_forward_nans_at_end: bool = False,
    replace_nans_at_start: Optional[PyTree[...]] = None,
) -> PyTree["times", ...]:  # noqa: F821
    """Fill in any missing values via linear interpolation.

    Any missing values in `ys` (represented as `NaN`) are filled in by looking at the
    nearest non-`NaN` values either side, and linearly interpolating.

    This is often useful prior to using [`diffrax.LinearInterpolation`][] to create a
    continuous path from discrete observations.

    **Arguments:**

    - `ts`: The time of each observation.
    - `ys`: The observations themselves. Should use `NaN` to indicate those missing
        observations to interpolate over.
    - `fill_forward_nans_at_end`: By default `NaN` values at the end (with no non-`NaN`
        value after them) are left as `NaN`s. If this is set then they will instead
        be filled in using the last non-`NaN` value.
    - `replace_nans_at_start`: By default `NaN` values at the start (with no non-`NaN`
        value before them) are left as `NaN`s. If this is passed then it will be used
        to fill in such `NaN` values.

    **Returns:**

    As `ys`, but with `NaN` values filled in.
    """

    ts = _check_ts(ts)
    fn = ft.partial(_linear_interpolation, fill_forward_nans_at_end, ts)
    if replace_nans_at_start is None:
        return jtu.tree_map(fn, ys)
    else:
        return jtu.tree_map(fn, ys, replace_nans_at_start)


def _rectilinear_interpolation(
    ts: Array["times"],  # noqa: F821
    replace_nans_at_start: Optional[Array["channels":...]],  # noqa: F821
    ys: Array["times", "channels":...],  # noqa: F821
) -> Tuple[
    Array["2 * times - 1"], Array["2 * times - 1", "channels":...]  # noqa: F821
]:
    ts = jnp.repeat(ts, 2, axis=0)[1:]
    ys = fill_forward(ys, replace_nans_at_start)
    ys = jnp.repeat(ys, 2, axis=0)[:-1]
    return ts, ys


@eqx.filter_jit
def rectilinear_interpolation(
    ts: Array["times"],  # noqa: F821
    ys: PyTree["times", ...],  # noqa: F821
    replace_nans_at_start: Optional[PyTree[...]] = None,
) -> Tuple[Array["2 * times - 1"], PyTree["2 * times - 1", ...]]:  # noqa: F821
    """Rectilinearly interpolates the input. This is a variant of linear interpolation
    that is particularly useful when using neural CDEs in a real-time scenario.

    This is often useful prior to using [`diffrax.LinearInterpolation`][] to create a
    continuous path from discrete observations, in real-time scenarios.

    It is strongly recommended to have a read of the reference below if you are
    unfamiliar.

    ??? cite "Reference"

        ```bibtex
        @article{morrill2021cdeonline,
                title={{N}eural {C}ontrolled {D}ifferential {E}quations for {O}nline
                       {P}rediction {T}asks},
                author={Morrill, James and Kidger, Patrick and Yang, Lingyi and
                        Lyons, Terry},
                journal={arXiv:2106.11028},
                year={2021}
        }
        ```

    !!! example

        Suppose `ts = [t0, t1, t2, t3]` and `ys = [y0, y1, y2, y3]`. Then rectilinearly
        interpolating these produces `new_ts = [t0, t1, t1, t2, t2, t3, t3]` and
        `new_ys = [y0, y0, y1, y1, y2, y2, y3]`.

        This can be thought of as advancing time whilst keeping the data fixed, then
        keeping the data fixed whilst advancing time.

    **Arguments:**

    - `ts`: The time of each observation.
    - `ys`: The observations themselves. Should use `NaN` to indicate those missing
        observations to interpolate over.
    - `replace_nans_at_start`: By default `NaN` values at the start (with no non-`NaN`
        value before them) are left as `NaN`s. If this is passed then it will be used
        to fill in such `NaN` values.

    **Returns:**

    A new version of both `ts` and `ys`, subject to rectilinear interpolation.

    !!! example

        Suppose we wish to use a rectilinearly interpolated control to drive a neural
        CDE. Then this should be done something like the following:

        ```python
        ts = jnp.array([0., 1., 1.5, 2.])
        ys = jnp.array([5., 6., 5., 6.])
        ts, ys = rectilinear_interpolation(ts, ys)
        data = jnp.stack([ts, ys], axis=-1)
        interp_ts = jnp.arange(7)
        interp = LinearInterpolation(interp_ts, data)
        ```

        Note how time and observations are stacked together as the data of the
        interpolation (as usual for a neural CDE), and how the interpolation times
        are something we are free to pick.
    """

    ts = _check_ts(ts)
    if replace_nans_at_start is None:
        fn = ft.partial(_rectilinear_interpolation, ts, None)
        out = jtu.tree_map(fn, ys)
    else:
        fn = ft.partial(_rectilinear_interpolation, ts)
        out = jtu.tree_map(fn, replace_nans_at_start, ys)
    ys_treedef = jtu.tree_structure(ys)
    interp_treedef = jtu.tree_structure((0, 0))
    return jtu.tree_transpose(ys_treedef, interp_treedef, out)


def _hermite_forward(
    carry: Tuple[
        Array["channels":...],  # noqa: F821
        Array["channels":...],  # noqa: F821
        Array["channels":...],  # noqa: F821
    ],
    value: Tuple[Scalar, Array["channels":...]],  # noqa: F821
) -> Tuple[
    Tuple[
        Array["channels":...],  # noqa: F821
        Array["channels":...],  # noqa: F821
        Array["channels":...],  # noqa: F821
    ],
    Tuple[
        Array["channels":...],  # noqa: F821
        Array["channels":...],  # noqa: F821
        Array["channels":...],  # noqa: F821
    ],
]:

    prev_ti, prev_yi, prev_deriv_i = carry
    ti, yi, next_ti, next_yi = value
    first_deriv_i = (next_yi - yi) / (next_ti - ti)
    later_deriv_i = (yi - prev_yi) / (ti - prev_ti)
    deriv_i = jnp.where(jnp.isnan(prev_yi), first_deriv_i, later_deriv_i)
    cond = jnp.isnan(yi)
    carry_ti = jnp.where(cond, prev_ti, ti)
    carry_yi = jnp.where(cond, prev_yi, yi)
    carry_deriv_i = jnp.where(cond, prev_deriv_i, deriv_i)

    return (carry_ti, carry_yi, carry_deriv_i), (carry_ti, carry_yi, carry_deriv_i)


def _hermite_coeffs(t0, y0, deriv0, t1, y1):
    t_diff = t1 - t0
    deriv1 = (y1 - y0) / t_diff
    d_deriv = deriv1 - deriv0

    a = y0
    b = deriv0
    c = 2 * d_deriv / t_diff
    d = -d_deriv / t_diff**2

    return d, c, b, a


def _hermite_impl(prev_ti, prev_yi, prev_deriv_i, ti, next_ti, next_yi):
    d, c, b, a = _hermite_coeffs(prev_ti, prev_yi, prev_deriv_i, next_ti, next_yi)
    ts = jnp.stack([prev_ti, next_ti])
    interpolation = CubicInterpolation(
        ts=ts, coeffs=(d[None], c[None], b[None], a[None])
    )
    interp_yi = interpolation.evaluate(ti)
    interp_deriv_i = interpolation.derivative(ti)
    return _hermite_coeffs(ti, interp_yi, interp_deriv_i, next_ti, next_yi)


def _backward_hermite_coefficients(
    fill_forward_nans_at_end: bool,
    ts: Array["times"],  # noqa: F821
    ys: Array["times", "channels":...],  # noqa: F821
    deriv0: Optional[Array["channels":...]] = None,  # noqa: F821
    replace_nans_at_start: Optional[Array["channels":...]] = None,  # noqa: F821
) -> Tuple[
    Array["channels":...],  # noqa: F821
    Array["channels":...],  # noqa: F821
    Array["channels":...],  # noqa: F821
    Array["channels":...],  # noqa: F821
]:
    ts = left_broadcast_to(ts, ys.shape)

    _, (next_ts, next_ys) = lax.scan(
        _interpolation_reverse, (ts[-1], ys[-1]), (ts[1:], ys[1:]), reverse=True
    )

    if fill_forward_nans_at_end:
        next_ys = fill_forward(next_ys)

    t0 = ts[0]
    if replace_nans_at_start is None:
        y0 = ys[0]
    else:
        y0 = jnp.broadcast_to(replace_nans_at_start, ys[0].shape)
    if deriv0 is None:
        deriv0 = (next_ys[0] - y0) / (next_ts[0] - t0)
    else:
        deriv0 = jnp.broadcast_to(deriv0, ys[0].shape)
    ts = ts[:-1]
    ys = ys[:-1]
    _, (prev_ts, prev_ys, prev_derivs) = lax.scan(
        _hermite_forward, (t0, y0, deriv0), (ts[1:], ys[1:], next_ts[1:], next_ys[1:])
    )
    prev_ts = jnp.concatenate([t0[None], prev_ts])
    prev_ys = jnp.concatenate([y0[None], prev_ys])
    prev_derivs = jnp.concatenate([deriv0[None], prev_derivs])

    hermite_impl = _hermite_impl
    for _ in range(len(ys.shape)):
        hermite_impl = jax.vmap(hermite_impl)
    ds, cs, bs, as_ = hermite_impl(prev_ts, prev_ys, prev_derivs, ts, next_ts, next_ys)

    return ds, cs, bs, as_


@eqx.filter_jit
def backward_hermite_coefficients(
    ts: Array["times"],  # noqa: F821
    ys: PyTree["times", ...],  # noqa: F821
    *,
    deriv0: Optional[PyTree[...]] = None,
    fill_forward_nans_at_end: bool = False,
    replace_nans_at_start: Optional[PyTree[...]] = None,
) -> Tuple[
    PyTree["times - 1", ...],  # noqa: F821
    PyTree["times - 1", ...],  # noqa: F821
    PyTree["times - 1", ...],  # noqa: F821
    PyTree["times - 1", ...],  # noqa: F821
]:
    """Interpolates the data with a cubic spline. Specifically, this calculates the
    coefficients for Hermite cubic splines with backward differences.

    This is most useful prior to using [`diffrax.CubicInterpolation`][] to create a
    smooth path from discrete observations.

    ??? cite "Reference"

        Hermite cubic splines with backward differences were introduced in this paper:

        ```bibtex
        @article{morrill2021cdeonline,
                title={{N}eural {C}ontrolled {D}ifferential {E}quations for {O}nline
                       {P}rediction {T}asks},
                author={Morrill, James and Kidger, Patrick and Yang, Lingyi and
                        Lyons, Terry},
                journal={arXiv:2106.11028},
                year={2021}
        }
        ```

    **Arguments:**

    - `ts`: The time of each observation.
    - `ys`: The observations themselves. Should use `NaN` to indicate missing data.
    - `deriv0`: The derivative at `ts[0]`. If not passed then a forward difference of
        `(ys[i] - ys[0]) / (ts[i] - ts[0])` is used, where `i` is the index of the
        first non-`NaN` element of `ys`.
    - `fill_forward_nans_at_end`: By default `NaN` values at the end (with no non-`NaN`
        value after them) are left as `NaN`s. If this is set then they will instead
        be filled in using the last non-`NaN` value prior to fitting the cubic spline.
    - `replace_nans_at_start`: By default `NaN` values at the start (with no non-`NaN`
        value before them) are left as `NaN`s. If this is passed then it will be used
        to fill in such `NaN` values.

    **Returns:**

    The coefficients of the Hermite cubic spline. If `ts` has length $T$ then the
    coefficients will be of length $T - 1$, covering each of the intervals from `ts[0]`
    to `ts[1]`, and `ts[1]` to `ts[2]` etc.
    """

    ts = _check_ts(ts)
    fn = ft.partial(_backward_hermite_coefficients, fill_forward_nans_at_end, ts)
    if deriv0 is None:
        if replace_nans_at_start is None:
            coeffs = jtu.tree_map(fn, ys)
        else:
            _fn = lambda ys, replace_nans_at_start: fn(ys, None, replace_nans_at_start)
            coeffs = jtu.tree_map(_fn, ys, replace_nans_at_start)
    else:
        if replace_nans_at_start is None:
            coeffs = jtu.tree_map(fn, ys, deriv0)
        else:
            coeffs = jtu.tree_map(fn, ys, deriv0, replace_nans_at_start)
    ys_treedef = jtu.tree_structure(ys)
    coeffs_treedef = jtu.tree_structure((0, 0, 0, 0))
    return jtu.tree_transpose(ys_treedef, coeffs_treedef, coeffs)
