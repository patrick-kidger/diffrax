from dataclasses import field
from typing import Callable, Optional, Tuple

import jax.lax as lax
import jax.numpy as jnp

from ..custom_types import Array, PyTree, Scalar
from ..misc import nextafter, nextbefore, ravel_pytree, unvmap
from ..solution import RESULTS
from .base import AbstractStepSizeController


def _rms_norm(x: PyTree) -> Scalar:
    x, _ = ravel_pytree(x)
    if x.size == 0:
        return 0
    sqnorm = jnp.mean(x ** 2)
    cond = sqnorm == 0
    # Double-where trick to avoid NaN gradients.
    # See JAX issues #5039 and #1052.
    _sqnorm = jnp.where(cond, 1.0, sqnorm)
    return jnp.where(cond, 0.0, jnp.sqrt(_sqnorm))


# Empirical initial step selection algorithm from:
# E. Hairer, S. P. Norsett G. Wanner, "Solving Ordinary Differential Equations I:
# Nonstiff Problems", Sec. II.4, 2nd edition.
def _select_initial_step(
    t0: Scalar,
    y0: Array["state"],  # noqa: F821
    args: PyTree,
    solver_order: int,
    func_for_init: Callable[
        [Scalar, Array["state"], PyTree], Array["state"]  # noqa: F821
    ],
    unravel_y: callable,
    rtol: Scalar,
    atol: Scalar,
    norm: Callable[[Array], Scalar],
):
    f0 = func_for_init(t0, y0, args)
    scale = atol + jnp.abs(y0) * rtol
    d0 = norm(unravel_y(y0 / scale))
    d1 = norm(unravel_y(f0 / scale))

    h0 = jnp.where((d0 < 1e-5) | (d1 < 1e-5), 1e-6, 0.01 * (d0 / d1))

    t1 = t0 + h0
    y1 = y0 + h0 * f0
    f1 = func_for_init(t1, y1, args)
    d2 = norm(unravel_y((f1 - f0) / scale)) / h0

    h1 = jnp.where(
        (d1 <= 1e-15) | (d2 <= 1e-15),
        jnp.maximum(1e-6, h0 * 1e-3),
        (0.01 * jnp.maximum(d1, d2)) ** (1 / solver_order),
    )

    return jnp.minimum(100 * h0, h1)


def _scale_error_estimate(
    y_error: Array["state"],  # noqa: F821
    y0: Array["state"],  # noqa: F821
    y1_candidate: Array["state"],  # noqa: F821
    unravel_y: callable,
    rtol: Scalar,
    atol: Scalar,
    norm: Callable[[Array], Scalar],
) -> Scalar:
    scale = y_error / (atol + jnp.maximum(y0, y1_candidate) * rtol)
    scale = unravel_y(scale)
    return norm(scale)


_do_not_set_at_init = object()  # Is set during wrap instead


_ControllerState = Array[(), bool]


# https://diffeq.sciml.ai/stable/extras/timestepping/
# are good notes on different step size control algorithms.
class IController(AbstractStepSizeController):
    # Default tolerances taken from scipy.integrate.solve_ivp
    rtol: Scalar = 1e-3
    atol: Scalar = 1e-6
    safety: Scalar = 0.9
    ifactor: Scalar = 10.0
    dfactor: Scalar = 0.2
    norm: Callable = _rms_norm
    dtmin: Optional[Scalar] = None
    dtmax: Optional[Scalar] = None
    force_dtmin: bool = True
    unvmap_dt: bool = False
    step_ts: Optional[Array["steps"]] = None  # noqa: F821
    jump_ts: Optional[Array["steps"]] = None  # noqa: F821
    unravel_y: callable = field(repr=False, default=_do_not_set_at_init)
    direction: Scalar = field(repr=False, default=_do_not_set_at_init)

    def __post_init__(self):
        if self.jump_ts is not None and not jnp.issubdtype(
            self.jump_ts.dtype, jnp.inexact
        ):
            raise ValueError(
                f"jump_ts must be floating point, not {self.jump_ts.dtype}"
            )

    def wrap(self, unravel_y: callable, direction: Scalar):
        return type(self)(
            rtol=self.rtol,
            atol=self.atol,
            safety=self.safety,
            ifactor=self.ifactor,
            dfactor=self.dfactor,
            norm=self.norm,
            dtmin=self.dtmin,
            dtmax=self.dtmax,
            force_dtmin=self.force_dtmin,
            unvmap_dt=self.unvmap_dt,
            step_ts=self.step_ts,
            jump_ts=self.jump_ts,
            unravel_y=unravel_y,
            direction=direction,
        )

    def init(
        self,
        t0: Scalar,
        y0: Array["state"],  # noqa: F821
        dt0: Optional[Scalar],
        args: PyTree,
        solver_order: int,
        func_for_init: Callable[
            [Scalar, Array["state"], PyTree],  # noqa: F821
            Array["state"],  # noqa: F821
        ],
    ) -> Tuple[Scalar, _ControllerState]:
        if dt0 is None:
            dt0 = _select_initial_step(
                t0,
                y0,
                args,
                solver_order,
                func_for_init,
                self.unravel_y,
                self.rtol,
                self.atol,
                self.norm,
            )

        t1 = self._clip_step_ts(t0, t0 + dt0)
        t1, jump_next_step = self._clip_jump_ts(t0, t1)

        return t1, jump_next_step

    def adapt_step_size(
        self,
        t0: Scalar,
        t1: Scalar,
        y0: Array["state"],  # noqa: F821
        y1_candidate: Array["state"],  # noqa: F821
        args: PyTree,
        y_error: Optional[Array["state"]],  # noqa: F821
        solver_order: int,
        controller_state: _ControllerState,
    ) -> Tuple[Array[(), bool], Scalar, Scalar, Array[(), bool], _ControllerState, int]:
        del args
        if y_error is None:
            raise ValueError(
                "Cannot use adaptive step sizes with a solver that does not provide "
                "error estimates."
            )
        prev_dt = t1 - t0

        scaled_error = _scale_error_estimate(
            y_error, y0, y1_candidate, self.unravel_y, self.rtol, self.atol, self.norm
        )
        keep_step = scaled_error < 1
        if self.dtmin is not None:
            keep_step = keep_step | (prev_dt == self.dtmin)
        if self.unvmap_dt:
            keep_step = jnp.all(unvmap(keep_step))

        # Double-where trick to avoid NaN gradients.
        # See JAX issues #5039 and #1052.
        cond = scaled_error == 0
        _scaled_error = jnp.where(cond, 1.0, scaled_error)
        factor = lax.cond(
            cond,
            lambda _: self.ifactor,
            self._scale_factor,
            (solver_order, keep_step, _scaled_error),
        )
        if self.unvmap_dt:
            factor = jnp.min(unvmap(factor))

        dt = prev_dt * factor
        result = jnp.full_like(t0, RESULTS.successful)
        if self.dtmin is not None:
            if not self.force_dtmin:
                result = jnp.where(dt < self.dtmin, RESULTS.dt_min_reached, result)
            dt = jnp.maximum(dt, self.dtmin)

        if self.dtmax is not None:
            dt = jnp.minimum(dt, self.dtmax)

        made_jump = controller_state

        if jnp.issubdtype(t1.dtype, jnp.inexact):
            _t1 = jnp.where(made_jump, nextafter(t1), t1)
        else:
            _t1 = t1
        next_t0 = jnp.where(keep_step, _t1, t0)

        next_t1 = self._clip_step_ts(next_t0, next_t0 + dt)
        next_t1, jump_next_step = self._clip_jump_ts(next_t0, next_t1)

        controller_state = jump_next_step

        return keep_step, next_t0, next_t1, made_jump, controller_state, result

    def _scale_factor(self, operand):
        order, keep_step, scaled_error = operand
        dfactor = jnp.where(keep_step, 1, self.dfactor)
        exponent = 1 / order
        return jnp.clip(
            self.safety / scaled_error ** exponent, a_min=dfactor, a_max=self.ifactor
        )

    def _clip_step_ts(self, t0: Scalar, t1: Scalar) -> Scalar:
        if self.step_ts is None:
            return t1
        # TODO: it should be possible to switch this O(nlogn) for just O(n) by keeping
        # track of where we were last, and using that as a hint for the next search.
        t0_index = jnp.searchsorted(self.step_ts, t0)
        t1_index = jnp.searchsorted(self.step_ts, t1)
        # This minimum may or may not actually be necessary. The left branch is taken
        # iff t0_index < t1_index <= len(self.step_ts), so all valid t0_index s must
        # already satisfy the minimum.
        # However, that branch is actually executed unconditionally and then where'd,
        # so we clamp it just to be sure we're not hitting undefined behaviour.
        t1 = jnp.where(
            t0_index < t1_index,
            self.step_ts[jnp.minimum(t0_index, len(self.step_ts) - 1)],
            t1,
        )
        return t1

    def _clip_jump_ts(self, t0: Scalar, t1: Scalar) -> Tuple[Scalar, bool]:
        if self.jump_ts is None:
            return t1, jnp.full_like(t1, fill_value=False, dtype=bool)
        if not jnp.issubdtype(t1.dtype, jnp.inexact):
            raise ValueError(
                "t0, t1, dt0 must be floating point when specifying jump_t. Got "
                f"{t1.dtype}."
            )
        t0_index = jnp.searchsorted(self.step_ts, t0)
        t1_index = jnp.searchsorted(self.step_ts, t1)
        cond = t0_index < t1_index
        t1 = jnp.where(
            cond,
            nextbefore(self.jump_ts[jnp.minimum(t0_index, len(self.step_ts) - 1)]),
            t1,
        )
        jump_next_step = jnp.where(cond, True, False)
        return t1, jump_next_step
