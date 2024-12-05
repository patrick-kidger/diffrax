from collections.abc import Callable
from typing import cast, Optional, TypeVar

import equinox as eqx
import equinox.internal as eqxi
import jax
import jax.lax as lax
import jax.numpy as jnp
import jax.tree_util as jtu
import lineax.internal as lxi
import optimistix as optx
from diffrax import AbstractTerm, ODETerm, RESULTS
from diffrax._custom_types import (
    Args,
    BoolScalarLike,
    IntScalarLike,
    RealScalarLike,
    VF,
    Y,
)
from diffrax._misc import static_select, upcast_or_raise
from diffrax._step_size_controller import AbstractAdaptiveStepSizeController
from equinox.internal import ω
from jaxtyping import Array, PyTree, Real
from lineax.internal import complex_to_real_dtype


ω = cast(Callable, ω)


def _select_initial_step(
    terms: PyTree[AbstractTerm],
    t0: RealScalarLike,
    y0: Y,
    args: Args,
    func: Callable[
        [PyTree[AbstractTerm], RealScalarLike, Y, Args],
        VF,
    ],
    error_order: RealScalarLike,
    rtol: RealScalarLike,
    atol: RealScalarLike,
    norm: Callable[[PyTree], RealScalarLike],
) -> RealScalarLike:
    # TODO: someone needs to figure out an initial step size algorithm for SDEs.
    if not isinstance(terms, ODETerm):
        return 0.01

    def fn(carry):
        t, y, _h0, _d1, _f, _ = carry
        f = func(terms, t, y, args)
        return t, y, _h0, _d1, _f, f

    def intermediate(carry):
        _, _, _, _, _, f0 = carry
        d0 = norm((y0**ω / scale**ω).ω)
        d1 = norm((f0**ω / scale**ω).ω)
        _cond = (d0 < 1e-5) | (d1 < 1e-5)
        _d1 = jnp.where(_cond, 1, d1)
        h0 = jnp.where(_cond, 1e-6, 0.01 * (d0 / _d1))
        t1 = t0 + h0
        y1 = (y0**ω + h0 * f0**ω).ω
        return t1, y1, h0, d1, f0, f0

    scale = (atol + ω(y0).call(jnp.abs) * rtol).ω
    dummy_h = t0
    dummy_d = eqxi.eval_empty(norm, y0)
    dummy_f = eqxi.eval_empty(lambda: func(terms, t0, y0, args))
    _, _, h0, d1, f0, f1 = eqxi.scan_trick(
        fn, [intermediate], (t0, y0, dummy_h, dummy_d, dummy_f, dummy_f)
    )
    d2 = norm(((f1**ω - f0**ω) / scale**ω).ω) / h0
    max_d = jnp.maximum(d1, d2)
    h1 = jnp.where(
        max_d <= 1e-15,
        jnp.maximum(1e-6, h0 * 1e-3),
        (0.01 / max_d) ** (1 / error_order),
    )
    return jnp.minimum(100 * h0, h1)


_ControllerState = TypeVar("_ControllerState")
_Dt0 = TypeVar("_Dt0", None, RealScalarLike, Optional[RealScalarLike])

_PidState = tuple[
    BoolScalarLike, BoolScalarLike, RealScalarLike, RealScalarLike, RealScalarLike
]


def _none_or_array(x):
    if x is None:
        return None
    else:
        return jnp.asarray(x)


class OldPIDController(
    AbstractAdaptiveStepSizeController[_PidState, Optional[RealScalarLike]]
):
    r"""See the doc of diffrax.PIDController for more information."""

    rtol: RealScalarLike
    atol: RealScalarLike
    pcoeff: RealScalarLike = 0
    icoeff: RealScalarLike = 1
    dcoeff: RealScalarLike = 0
    dtmin: Optional[RealScalarLike] = None
    dtmax: Optional[RealScalarLike] = None
    force_dtmin: bool = True
    step_ts: Optional[Real[Array, " steps"]] = eqx.field(
        default=None, converter=_none_or_array
    )
    jump_ts: Optional[Real[Array, " jumps"]] = eqx.field(
        default=None, converter=_none_or_array
    )
    factormin: RealScalarLike = 0.2
    factormax: RealScalarLike = 10.0
    norm: Callable[[PyTree], RealScalarLike] = optx.rms_norm
    safety: RealScalarLike = 0.9
    error_order: Optional[RealScalarLike] = None

    def __check_init__(self):
        if self.jump_ts is not None and not jnp.issubdtype(
            self.jump_ts.dtype, jnp.inexact
        ):
            raise ValueError(
                f"jump_ts must be floating point, not {self.jump_ts.dtype}"
            )

    def wrap(self, direction: IntScalarLike):
        step_ts = None if self.step_ts is None else self.step_ts * direction
        jump_ts = None if self.jump_ts is None else self.jump_ts * direction
        return eqx.tree_at(
            lambda s: (s.step_ts, s.jump_ts),
            self,
            (step_ts, jump_ts),
            is_leaf=lambda x: x is None,
        )

    def init(
        self,
        terms: PyTree[AbstractTerm],
        t0: RealScalarLike,
        t1: RealScalarLike,
        y0: Y,
        dt0: Optional[RealScalarLike],
        args: Args,
        func: Callable[[PyTree[AbstractTerm], RealScalarLike, Y, Args], VF],
        error_order: Optional[RealScalarLike],
    ) -> tuple[RealScalarLike, _PidState]:
        del t1
        if dt0 is None:
            error_order = self._get_error_order(error_order)
            dt0 = _select_initial_step(
                terms,
                t0,
                y0,
                args,
                func,
                error_order,
                self.rtol,
                self.atol,
                self.norm,
            )

            dt0 = lax.stop_gradient(dt0)
        if self.dtmax is not None:
            dt0 = jnp.minimum(dt0, self.dtmax)
        if self.dtmin is None:
            at_dtmin = jnp.array(False)
        else:
            at_dtmin = dt0 <= self.dtmin
            dt0 = jnp.maximum(dt0, self.dtmin)

        t1 = self._clip_step_ts(t0, t0 + dt0)
        t1, jump_next_step = self._clip_jump_ts(t0, t1)

        y_leaves = jtu.tree_leaves(y0)
        if len(y_leaves) == 0:
            y_dtype = lxi.default_floating_dtype()
        else:
            y_dtype = jnp.result_type(*y_leaves)
        return t1, (
            jump_next_step,
            at_dtmin,
            dt0,
            jnp.array(1.0, dtype=complex_to_real_dtype(y_dtype)),
            jnp.array(1.0, dtype=complex_to_real_dtype(y_dtype)),
        )

    def adapt_step_size(
        self,
        t0: RealScalarLike,
        t1: RealScalarLike,
        y0: Y,
        y1_candidate: Y,
        args: Args,
        y_error: Optional[Y],
        error_order: RealScalarLike,
        controller_state: _PidState,
    ) -> tuple[
        BoolScalarLike,
        RealScalarLike,
        RealScalarLike,
        BoolScalarLike,
        _PidState,
        RESULTS,
    ]:
        del args
        if y_error is None and y0 is not None:
            # y0 is not None check is included to handle the edge case that the state
            # is just a trivial `None` PyTree. In this case `y_error` has the same
            # PyTree structure and thus overlaps with our special usage of `None` to
            # indicate a lack of error estimate.
            raise RuntimeError(
                "Cannot use adaptive step sizes with a solver that does not provide "
                "error estimates."
            )
        (
            made_jump,
            at_dtmin,
            prev_dt,
            prev_inv_scaled_error,
            prev_prev_inv_scaled_error,
        ) = controller_state
        error_order = self._get_error_order(error_order)
        prev_dt = jnp.where(made_jump, prev_dt, t1 - t0)

        #
        # Figure out how things went on the last step: error, and whether to
        # accept/reject it.
        #

        def _scale(_y0, _y1_candidate, _y_error):
            # In case the solver steps into a region for which the vector field isn't
            # defined.
            _nan = jnp.isnan(_y1_candidate).any()
            _y1_candidate = jnp.where(_nan, _y0, _y1_candidate)
            _y = jnp.maximum(jnp.abs(_y0), jnp.abs(_y1_candidate))
            with jax.numpy_dtype_promotion("standard"):
                return _y_error / (self.atol + _y * self.rtol)

        scaled_error = self.norm(jtu.tree_map(_scale, y0, y1_candidate, y_error))
        keep_step = scaled_error < 1
        if self.dtmin is not None:
            keep_step = keep_step | at_dtmin
        # Make sure it's not a Python scalar and thus getting a ZeroDivisionError.
        inv_scaled_error = 1 / jnp.asarray(scaled_error)
        inv_scaled_error = lax.stop_gradient(
            inv_scaled_error
        )  # See note in init above.
        # Note: if you ever remove this lax.stop_gradient, then you'll need to do a lot
        # of work to get safe gradients through these operations.
        # When `inv_scaled_error` has a (non-symbolic) zero cotangent, and `y_error`
        # is either zero or inf, then we get a `0 * inf = nan` on the backward pass.

        #
        # Adjust next step size
        #

        _zero_coeff = lambda c: isinstance(c, (int, float)) and c == 0
        coeff1 = (self.icoeff + self.pcoeff + self.dcoeff) / error_order
        coeff2 = -cast(RealScalarLike, self.pcoeff + 2 * self.dcoeff) / error_order
        coeff3 = self.dcoeff / error_order
        factor1 = 1 if _zero_coeff(coeff1) else inv_scaled_error**coeff1
        factor2 = 1 if _zero_coeff(coeff2) else prev_inv_scaled_error**coeff2
        factor3 = 1 if _zero_coeff(coeff3) else prev_prev_inv_scaled_error**coeff3
        factormin = jnp.where(keep_step, 1, self.factormin)
        factor = jnp.clip(
            self.safety * factor1 * factor2 * factor3,
            min=factormin,
            max=self.factormax,
        )
        # Once again, see above. In case we have gradients on {i,p,d}coeff.
        # (Probably quite common for them to have zero tangents if passed across
        # a grad API boundary as part of a larger model.)
        factor = lax.stop_gradient(factor)
        factor = eqxi.nondifferentiable(factor)
        dt = prev_dt * factor.astype(jnp.result_type(prev_dt))

        # E.g. we failed an implicit step, so y_error=inf, so inv_scaled_error=0,
        # so factor=factormin, and we shrunk our step.
        # If we're using a PI or PID controller we shouldn't then force shrinking on
        # the next or next two steps as well!
        pred = (inv_scaled_error == 0) | jnp.isinf(inv_scaled_error)
        inv_scaled_error = jnp.where(pred, 1, inv_scaled_error)

        #
        # Clip next step size based on dtmin/dtmax
        #

        result = RESULTS.successful
        if self.dtmax is not None:
            dt = jnp.minimum(dt, self.dtmax)
        if self.dtmin is None:
            at_dtmin = jnp.array(False)
        else:
            if not self.force_dtmin:
                result = RESULTS.where(dt < self.dtmin, RESULTS.dt_min_reached, result)
            at_dtmin = dt <= self.dtmin
            dt = jnp.maximum(dt, self.dtmin)

        #
        # Clip next step size based on step_ts/jump_ts
        #

        if jnp.issubdtype(jnp.result_type(t1), jnp.inexact):
            # Two nextafters. If made_jump then t1 = prevbefore(jump location)
            # so now _t1 = nextafter(jump location)
            # This is important because we don't know whether or not the jump is as a
            # result of a left- or right-discontinuity, so we have to skip the jump
            # location altogether.
            _t1 = static_select(made_jump, eqxi.nextafter(eqxi.nextafter(t1)), t1)
        else:
            _t1 = t1
        next_t0 = jnp.where(keep_step, _t1, t0)
        next_t1 = self._clip_step_ts(next_t0, next_t0 + dt)
        next_t1, next_made_jump = self._clip_jump_ts(next_t0, next_t1)

        inv_scaled_error = jnp.where(keep_step, inv_scaled_error, prev_inv_scaled_error)
        prev_inv_scaled_error = jnp.where(
            keep_step, prev_inv_scaled_error, prev_prev_inv_scaled_error
        )
        controller_state = (
            next_made_jump,
            at_dtmin,
            dt,
            inv_scaled_error,
            prev_inv_scaled_error,
        )
        return keep_step, next_t0, next_t1, made_jump, controller_state, result

    def _get_error_order(self, error_order: Optional[RealScalarLike]) -> RealScalarLike:
        # Attribute takes priority, if the user knows the correct error order better
        # than our guess.
        error_order = error_order if self.error_order is None else self.error_order
        if error_order is None:
            raise ValueError(
                "The order of convergence for the solver has not been specified; pass "
                "`PIDController(..., error_order=...)` manually instead. If solving "
                "an ODE then this should be equal to the (global) order plus one. If "
                "solving an SDE then should be equal to the (global) order plus 0.5."
            )
        return error_order

    def _clip_step_ts(self, t0: RealScalarLike, t1: RealScalarLike) -> RealScalarLike:
        if self.step_ts is None:
            return t1

        step_ts0 = upcast_or_raise(
            self.step_ts,
            t0,
            "`PIDController.step_ts`",
            "time (the result type of `t0`, `t1`, `dt0`, `SaveAt(ts=...)` etc.)",
        )
        step_ts1 = upcast_or_raise(
            self.step_ts,
            t1,
            "`PIDController.step_ts`",
            "time (the result type of `t0`, `t1`, `dt0`, `SaveAt(ts=...)` etc.)",
        )
        # TODO: it should be possible to switch this O(nlogn) for just O(n) by keeping
        # track of where we were last, and using that as a hint for the next search.
        t0_index = jnp.searchsorted(step_ts0, t0, side="right")
        t1_index = jnp.searchsorted(step_ts1, t1, side="right")
        # This minimum may or may not actually be necessary. The left branch is taken
        # iff t0_index < t1_index <= len(self.step_ts), so all valid t0_index s must
        # already satisfy the minimum.
        # However, that branch is actually executed unconditionally and then where'd,
        # so we clamp it just to be sure we're not hitting undefined behaviour.
        t1 = jnp.where(
            t0_index < t1_index,
            step_ts1[jnp.minimum(t0_index, len(self.step_ts) - 1)],
            t1,
        )
        return t1

    def _clip_jump_ts(
        self, t0: RealScalarLike, t1: RealScalarLike
    ) -> tuple[RealScalarLike, BoolScalarLike]:
        if self.jump_ts is None:
            return t1, False
        assert jnp.issubdtype(self.jump_ts.dtype, jnp.inexact)
        if not jnp.issubdtype(jnp.result_type(t0), jnp.inexact):
            raise ValueError(
                "`t0`, `t1`, `dt0` must be floating point when specifying `jump_ts`. "
                f"Got {jnp.result_type(t0)}."
            )
        if not jnp.issubdtype(jnp.result_type(t1), jnp.inexact):
            raise ValueError(
                "`t0`, `t1`, `dt0` must be floating point when specifying `jump_ts`. "
                f"Got {jnp.result_type(t1)}."
            )
        jump_ts0 = upcast_or_raise(
            self.jump_ts,
            t0,
            "`PIDController.jump_ts`",
            "time (the result type of `t0`, `t1`, `dt0`, `SaveAt(ts=...)` etc.)",
        )
        jump_ts1 = upcast_or_raise(
            self.jump_ts,
            t1,
            "`PIDController.jump_ts`",
            "time (the result type of `t0`, `t1`, `dt0`, `SaveAt(ts=...)` etc.)",
        )
        t0_index = jnp.searchsorted(jump_ts0, t0, side="right")
        t1_index = jnp.searchsorted(jump_ts1, t1, side="right")
        next_made_jump = t0_index < t1_index
        t1 = jnp.where(
            next_made_jump,
            eqxi.prevbefore(jump_ts1[jnp.minimum(t0_index, len(self.jump_ts) - 1)]),
            t1,
        )
        return t1, next_made_jump
