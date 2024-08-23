from collections.abc import Callable
from typing import Generic, Optional, TYPE_CHECKING, TypeVar

import equinox as eqx
import equinox.internal as eqxi
import jax
import jax.numpy as jnp
from jaxtyping import Array, PyTree, Real

from .._custom_types import (
    Args,
    BoolScalarLike,
    IntScalarLike,
    RealScalarLike,
    VF,
    Y,
)
from .._misc import static_select, upcast_or_raise
from .._solution import RESULTS
from .._term import AbstractTerm
from .adaptive_base import AbstractAdaptiveStepSizeController
from .base import AbstractStepSizeController


_ControllerState = TypeVar("_ControllerState")
_Dt0 = TypeVar("_Dt0", None, RealScalarLike, Optional[RealScalarLike])


class _JumpStepState(eqx.Module, Generic[_ControllerState]):
    made_jump: BoolScalarLike
    prev_dt: RealScalarLike
    step_index: IntScalarLike
    jump_index: IntScalarLike
    rejected_index: IntScalarLike
    rejected_buffer: Optional[Array]
    step_ts: Optional[Array]
    jump_ts: Optional[Array]
    inner_state: _ControllerState

    def get(self):
        return (
            self.made_jump,
            self.prev_dt,
            self.step_index,
            self.jump_index,
            self.rejected_index,
            self.rejected_buffer,
            self.step_ts,
            self.jump_ts,
            self.inner_state,
        )


def _none_or_array(x):
    if x is None:
        return None
    else:
        return jnp.asarray(x)


def _get_t(i: IntScalarLike, ts: Array) -> RealScalarLike:
    i_min_len = jnp.minimum(i, len(ts) - 1)
    return jnp.where(i == len(ts), jnp.inf, ts[i_min_len])


def _clip_ts(
    t0: RealScalarLike,
    t1: RealScalarLike,
    i: IntScalarLike,
    ts: Optional[Array],
    check_inexact: bool,
) -> tuple[RealScalarLike, BoolScalarLike]:
    if ts is None:
        return t1, False

    if check_inexact:
        assert jnp.issubdtype(ts.dtype, jnp.inexact)
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

    _t1 = _get_t(i, ts)
    next_made_jump = _t1 <= t1
    _t1 = jnp.where(next_made_jump, _t1, t1)
    return _t1, next_made_jump


def _find_index(t: RealScalarLike, ts: Optional[Array]) -> IntScalarLike:
    if ts is None:
        return 0

    ts = upcast_or_raise(
        ts,
        t,
        "`JumpStepWrapper.step_ts`",
        "time (the result type of `t0`, `t1`, `dt0`, `SaveAt(ts=...)` etc.)",
    )
    return jnp.searchsorted(ts, t, side="right")


def _revisit_rejected(
    t0: RealScalarLike,
    t1: RealScalarLike,
    i_reject: IntScalarLike,
    rejected_buffer: Optional[Array],
) -> RealScalarLike:
    if rejected_buffer is None:
        return t1
    _t1 = _get_t(i_reject, rejected_buffer)
    _t1 = jnp.minimum(_t1, t1)
    return _t1


# EXPLANATION OF STEP_TS AND JUMP_TS
# -----------------------------------
# The `step_ts` and `jump_ts` are used to force the solver to step to certain times.
# They mostly act in the same way, except that when we hit an element of `jump_ts`,
# the controller must return `made_jump = True`, so that the diffeqsolve function
# knows that the vector field has a discontinuity at that point. In addition, the
# exact time of the jump will be skipped using jnp.prevbefore and jnp.nextafter.
# So now to the explanation of the two (we will use `step_ts` as an example, but the
# same applies to `jump_ts`):
#
# If `step_ts` is not None, we assume it is a sorted array of times.
# At the start of the run, the init function finds the smallest index `i_step` such
# that `step_ts[i_step] > t0`. At init and after each step of the solver, the
# controller will propose a step t1_next, and we will clip it to
# `t1_next = min(t1_next, step_ts[i_step])`.
# At the start of the next step, if the step ended at t1 == step_ts[i_step] and
# if the controller decides to keep the step, then this time has been successfully
# stepped to and we increment `i_step` by 1.
# We use a convenience function _get_t(i, ts) which returns ts[i] if i < len(ts) and
# infinity otherwise.

# EXPLANATION OF REVISITING REJECTED STEPS
# ----------------------------------------
# We use a "stack" of rejected steps, composed of a buffer `rejected_buffer` of length
# `rejected_step_buffer_len` and a counter `i_reject`. The "stack" are all the items
# in `rejected_buffer[i_reject:]` with `rejected_buffer[i_reject]` being the top of
# the stack.
# When `i_reject == rejected_step_buffer_len`, the stack is empty.
# At the start of the run, `i_reject = rejected_step_buffer_len`. Each time a step is
# rejected `i_reject -=1` and `rejected_buffer[i_reject] = t1`. Each time a step ends at
# `t1 == rejected_buffer[i_reject]`, we increment `i_reject` by 1 (even if the step was
# rejected, in which case we will re-add `t1` to the stack immediately).
# We clip the next step to `t1_next = min(t1_next, rejected_buffer[i_reject])`.
# If `i_reject < 0` then an error is raised.


class JumpStepWrapper(
    AbstractStepSizeController[_JumpStepState[_ControllerState], _Dt0]
):
    """Wraps an existing step controller and adds the ability to specify `step_ts`
    and `jump_ts`. The former are times to which the controller should step and the
    latter are times at which the vector field has a discontinuity (jump)."""

    controller: AbstractAdaptiveStepSizeController[_ControllerState, _Dt0]
    step_ts: Optional[Real[Array, " steps"]]
    jump_ts: Optional[Real[Array, " jumps"]]
    rejected_step_buffer_len: Optional[int] = eqx.field(static=True)
    callback_on_reject: Optional[Callable] = eqx.field(static=True)

    @eqxi.doc_remove_args("_callback_on_reject")
    def __init__(
        self,
        controller,
        step_ts=None,
        jump_ts=None,
        rejected_step_buffer_len=None,
        _callback_on_reject=None,
    ):
        r"""
        **Arguments**:

        - `controller`: The controller to wrap.
            Can be any diffrax.AbstractAdaptiveStepSizeController.
        - `step_ts`: Denotes extra times that must be stepped to.
        - `jump_ts`: Denotes extra times that must be stepped to, and at which the
            vector field has a known discontinuity. (This is used to force FSAL solvers
            so re-evaluate the vector field.)
        - `rejected_step_buffer_len`: The length of the buffer storing rejected steps.
            Can either be None or a positive integer.
            If it is > 0, then the controller will revisit rejected steps. This is
            useful for SDEs, where the solution is guaranteed to be correct if the
            SDE is evaluated at all times at which the Brownian motion (BM) is
            evaluated. Since the BM is also evaluated at rejected steps, we must later
            evaluate the SDE at these times as well.
        """
        self.controller = controller
        self.step_ts = _none_or_array(step_ts)
        self.jump_ts = _none_or_array(jump_ts)
        if rejected_step_buffer_len is not None:
            assert rejected_step_buffer_len > 0
        self.rejected_step_buffer_len = rejected_step_buffer_len
        self.callback_on_reject = _callback_on_reject

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
        controller = self.controller.wrap(direction)
        return eqx.tree_at(
            lambda s: (s.step_ts, s.jump_ts, s.controller),
            self,
            (step_ts, jump_ts, controller),
            is_leaf=lambda x: x is None,
        )

    def init(
        self,
        terms: PyTree[AbstractTerm],
        t0: RealScalarLike,
        t1: RealScalarLike,
        y0: Y,
        dt0: _Dt0,
        args: Args,
        func: Callable[[PyTree[AbstractTerm], RealScalarLike, Y, Args], VF],
        error_order: Optional[RealScalarLike],
    ) -> tuple[RealScalarLike, _JumpStepState[_ControllerState]]:
        t1, inner_state = self.controller.init(
            terms, t0, t1, y0, dt0, args, func, error_order
        )
        dt_proposal = t1 - t0
        tdtype = jnp.result_type(t0, t1)

        if self.step_ts is None:
            step_ts = None
        else:
            # Upcast step_ts to the same dtype as t0, t1
            step_ts = upcast_or_raise(
                self.step_ts,
                jnp.zeros((), tdtype),
                "`JumpStepWrapper.step_ts`",
                "time (the result type of `t0`, `t1`, `dt0`, `SaveAt(ts=...)` etc.)",
            )

        if self.jump_ts is None:
            jump_ts = None
        else:
            # Upcast jump_ts to the same dtype as t0, t1
            jump_ts = upcast_or_raise(
                self.jump_ts,
                jnp.zeros((), tdtype),
                "`JumpStepWrapper.jump_ts`",
                "time (the result type of `t0`, `t1`, `dt0`, `SaveAt(ts=...)` etc.)",
            )

        if self.rejected_step_buffer_len is None:
            rejected_buffer = None
            i_reject = jnp.asarray(0)
        else:
            rejected_buffer = jnp.zeros(
                (self.rejected_step_buffer_len,) + jnp.shape(t1), dtype=tdtype
            )
            # rejected_buffer[len(rejected_buffer)] = jnp.inf (see def of _get_t)
            i_reject = jnp.asarray(self.rejected_step_buffer_len)

        # Find index of first element of step_ts/jump_ts greater than t0
        i_step = _find_index(t0, step_ts)
        i_jump = _find_index(t0, jump_ts)
        # Clip t1 to the next element of step_ts or jump_ts
        t1, _ = _clip_ts(t0, t1, i_step, step_ts, False)
        t1, jump_next_step = _clip_ts(t0, t1, i_jump, jump_ts, True)

        state = _JumpStepState(
            jump_next_step,
            dt_proposal,
            i_step,
            i_jump,
            i_reject,
            rejected_buffer,
            step_ts,
            jump_ts,
            inner_state,
        )

        return t1, state

    def adapt_step_size(
        self,
        t0: RealScalarLike,
        t1: RealScalarLike,
        y0: Y,
        y1_candidate: Y,
        args: Args,
        y_error: Optional[Y],
        error_order: RealScalarLike,
        controller_state: _JumpStepState[_ControllerState],
    ) -> tuple[
        BoolScalarLike,
        RealScalarLike,
        RealScalarLike,
        BoolScalarLike,
        _JumpStepState[_ControllerState],
        RESULTS,
    ]:
        (
            made_jump,
            prev_dt,
            i_step,
            i_jump,
            i_reject,
            rejected_buffer,
            step_ts,
            jump_ts,
            inner_state,
        ) = controller_state.get()

        # Let the controller do its thing
        (
            keep_step,
            next_t0,
            next_t1,
            _,
            inner_state,
            result,
        ) = self.controller.adapt_step_size(
            t0, t1, y0, y1_candidate, args, y_error, error_order, inner_state
        )

        # This is just a logging utility for testing purposes
        if self.callback_on_reject is not None:
            jax.debug.callback(self.callback_on_reject, keep_step, t1)

        # Check whether we stepped over an element of step_ts/jump_ts/rejected_buffer
        # This is all still bookkeeping for the PREVIOUS STEP.
        if step_ts is not None:
            # If we stepped to `t1 == step_ts[i_step]` and kept the step, then we
            # increment i_step and move on to the next t in step_ts.
            step_inc_cond = keep_step & (t1 == _get_t(i_step, step_ts))
            i_step = jnp.where(step_inc_cond, i_step + 1, i_step)

        if jump_ts is not None:
            next_jump_t = _get_t(i_jump, jump_ts)
            jump_inc_cond = keep_step & (t1 >= eqxi.prevbefore(next_jump_t))
            i_jump = jnp.where(jump_inc_cond, i_jump + 1, i_jump)

        if self.rejected_step_buffer_len is not None:
            assert rejected_buffer is not None
            # If the step ended at t1==rejected_buffer[i_reject], then we have
            # successfully stepped to this time and we increment i_reject.
            # We increment i_reject even if the step was rejected, because we will
            # re-add the rejected time to the buffer immediately.
            rjct_inc_cond = t1 == _get_t(i_reject, rejected_buffer)
            i_reject = jnp.where(rjct_inc_cond, i_reject + 1, i_reject)

            # If the step was rejected, then we need to store the rejected time in the
            # rejected buffer and decrement the rejected index.
            i_reject = jnp.where(keep_step, i_reject, i_reject - 1)
            i_reject = eqx.error_if(
                i_reject,
                i_reject < 0,
                "Maximum number of rejected steps reached. "
                "Consider increasing JumpStepWrapper.rejected_step_buffer_len.",
            )
            clipped_i = jnp.clip(i_reject, 0, self.rejected_step_buffer_len - 1)
            update_rejected_t = jnp.where(keep_step, rejected_buffer[clipped_i], t1)
            rejected_buffer = rejected_buffer.at[clipped_i].set(update_rejected_t)

        # Now move on to the NEXT STEP
        dt_proposal = next_t1 - next_t0
        # The following line is so that in case prev_dt was intended to be large,
        # but then clipped to very small (because of step_ts or jump_ts), we don't
        # want it to stick to very small steps (e.g. the PID controller can only
        # increase steps by a factor of 10 at a time).
        dt_proposal = jnp.where(
            keep_step, jnp.maximum(dt_proposal, prev_dt), dt_proposal
        )
        new_prev_dt = dt_proposal
        next_t1 = next_t0 + dt_proposal

        # If t1 hit a jump point, and the step was kept then we need to set
        # `next_t0 = nextafter(nextafter(t1))` to ensure that we really skip
        # over the jump and don't evaluate the vector field at the discontinuity.
        if jnp.issubdtype(jnp.result_type(next_t0), jnp.inexact):
            # Two nextafters. If made_jump then t1 = prevbefore(jump location)
            # so now _t1 = nextafter(jump location)
            # This is important because we don't know whether or not the jump is as a
            # result of a left- or right-discontinuity, so we have to skip the jump
            # location altogether.
            jump_keep = made_jump & keep_step
            next_t0 = static_select(
                jump_keep, eqxi.nextafter(eqxi.nextafter(next_t0)), next_t0
            )

        if TYPE_CHECKING:
            assert isinstance(
                next_t0, RealScalarLike
            ), f"type(next_t0) = {type(next_t0)}"

        # Clip the step to the next element of jump_ts or step_ts or
        # rejected_buffer. Important to do jump_ts last because otherwise
        # jump_next_step could be a false positive.
        next_t1 = _revisit_rejected(next_t0, next_t1, i_reject, rejected_buffer)
        next_t1, _ = _clip_ts(next_t0, next_t1, i_step, step_ts, False)
        next_t1, jump_next_step = _clip_ts(next_t0, next_t1, i_jump, jump_ts, True)

        state = _JumpStepState(
            jump_next_step,
            new_prev_dt,
            i_step,
            i_jump,
            i_reject,
            rejected_buffer,
            step_ts,
            jump_ts,
            inner_state,
        )

        return keep_step, next_t0, next_t1, made_jump, state, result
