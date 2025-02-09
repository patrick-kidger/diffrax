from collections.abc import Callable
from typing import cast, Generic, Optional, TypeVar

import equinox as eqx
import equinox.internal as eqxi
import jax
import jax.numpy as jnp
from jaxtyping import Array, PyTree, Real

from .._custom_types import (
    Args,
    BoolScalarLike,
    FloatScalarLike,
    IntScalarLike,
    RealScalarLike,
    VF,
    Y,
)
from .._misc import upcast_or_raise
from .._solution import is_okay, RESULTS
from .._term import AbstractTerm
from .base import AbstractStepSizeController


_ControllerState = TypeVar("_ControllerState")
_Dt0 = TypeVar("_Dt0", bound=Optional[RealScalarLike])


class _ClipState(eqx.Module, Generic[_ControllerState]):
    step_info: Optional[tuple[IntScalarLike, Array]]
    jump_info: Optional[tuple[IntScalarLike, Array]]
    reject_info: Optional[tuple[IntScalarLike, Array]]
    inner_state: _ControllerState


def _none_or_sorted_array(x):
    if x is None:
        return None
    else:
        return jnp.sort(jnp.asarray(x))


def _assert_floating(t: FloatScalarLike, name: str, dtype):
    t_dtype = jnp.result_type(t)
    if not jnp.issubdtype(t_dtype, jnp.floating):
        raise ValueError(f"{name} must be floating-point, got {t_dtype}")
    if t_dtype != dtype:
        raise ValueError(
            f"All timelike inputs must have the same dtype got both {dtype} and "
            f"{t_dtype}."
        )


def _get_t(i: IntScalarLike, ts: Array) -> RealScalarLike:
    # As `ts[i]`, but `ts[len(ts))]` returns `inf`.
    # `i` must be in `{0, 1, ..., len(ts)}`.
    if len(ts) == 0:
        return jnp.inf
    else:
        i_min_len = jnp.minimum(i, len(ts) - 1)
        return jnp.where(i == len(ts), jnp.inf, ts[i_min_len])


def _clip_t(
    t: FloatScalarLike,
    i: IntScalarLike,
    ts: Array,
    prevbefore: bool,
) -> FloatScalarLike:
    assert jnp.issubdtype(jnp.result_type(t), jnp.floating)
    assert jnp.result_type(t) == jnp.result_type(ts)
    _t = _get_t(i, ts)
    if prevbefore:
        _t = eqxi.prevbefore(_t)
    return jnp.minimum(_t, t)


def _bump_next_t0(next_t0, ts):
    # Our previous step may have been to prevbefore a jump.
    # In this case we want to bump our next step to occur nextafter the jump.
    # We don't test against just `jump_ts[jump_index]`. The index in the state
    # is intended only as a hint to improve the efficiency of
    # `_find_idx_with_hint`; it's not load-bearing. This is for safety, in case some
    # other stepsize control is going on. (TODO: do we want to keep it like this, or
    # do we want to switch to just the single check?)
    nextafter_next_t0 = eqxi.nextafter(next_t0)
    made_jump1 = jnp.any(nextafter_next_t0 == ts)
    # For safety we also test `next_t0 == ts`, just in case some other stepsize control
    # is going on. (I don't think this should actually be necessary.)
    made_jump2 = jnp.any(next_t0 == ts)
    # There are two nextafters. This is important because we don't know whether
    # or not the jump is a left- or a right-discontinuity, so we skip the jump
    # time altogether.
    next_t0 = jnp.where(made_jump1, eqxi.nextafter(nextafter_next_t0), next_t0)
    next_t0 = cast(Array, next_t0)
    next_t0 = jnp.where(made_jump2, nextafter_next_t0, next_t0)
    next_t0 = cast(Array, next_t0)
    return next_t0, made_jump1 | made_jump2


def _find_idx_with_hint(t: RealScalarLike, ts: Optional[Array], hint: IntScalarLike):
    # Find index of first element of `ts` strictly greater than `t`.
    # Uses a linear search starting from `hint`. The value `hint` is assumed to be in
    # `{0, 1, ..., len(ts)}`
    if ts is None:
        return 0

    def cond_up(_i):
        return (_i < len(ts)) & (ts[_i] <= t)

    def cond_down(_i):
        return (_i > 0) & (ts[_i - 1] > t)

    i = hint
    i = jax.lax.while_loop(cond_up, lambda _i: _i + 1, i)
    i = jax.lax.while_loop(cond_down, lambda _i: _i - 1, i)
    return i


class ClipStepSizeController(
    AbstractStepSizeController[_ClipState[_ControllerState], _Dt0]
):
    """Wraps an existing step controller with three pieces of functionality:

    - Have the solver step exactly to certain times ('`step_ts`').
    - Have the solver step to just before and just after certain time ('`jump_ts`').
    - Have the solver record the times of rejected steps, and step exactly to those
        times in future steps.

    In all cases this essentially corresponds to clipping steps so that any that are
    'too large' will instead by clipped from one of the three above cases.

    Stepping exactly to certain times can be useful if you want to ensure that your
    solution is highly accurate at that exact time point -- by default Diffrax will
    adaptively step wherever it likes, and then interpolate to produce the output values
    in `SaveAt(ts=...)`.

    Specifying jump times is needed for computational efficiency when solving
    differential equations for which the vector field has known jumps (e.g. due to a
    discontinuous forcing term). Otherwise an adaptive solver must reject many steps as
    it slows down to try and locate a jump. When using this, the solver will step to the
    floating point number immediately before the jump, and then resume solving from the
    floating point number immediately after it, with the jump itself not being
    evaluated.

    Revisiting rejected steps is needed when adaptively solving SDEs with noncommutative
    noise. Otherwise, a small bias may be introduced in the higher-order (Lévy area)
    terms of the solution, as it is possible to reject a step *because* of the samples
    drawn in these higher order terms.

    ??? Citation

        For more details on revisiting rejected steps when adaptively solving SDEs, see:

        ```bibtex
        @misc{foster2024convergenceadaptiveapproximationsstochastic,
            title={On the convergence of adaptive approximations for
                   stochastic differential equations},
            author={James Foster and Andraž Jelinčič},
            year={2024},
            eprint={2311.14201},
            archivePrefix={arXiv},
            primaryClass={math.NA},
            url={https://arxiv.org/abs/2311.14201},
        }
        ```
    """

    controller: AbstractStepSizeController[_ControllerState, _Dt0]
    step_ts: Optional[Real[Array, " steps"]]
    jump_ts: Optional[Real[Array, " jumps"]]
    store_rejected_steps: Optional[int] = eqx.field(static=True)
    callback_on_reject: Optional[Callable] = eqx.field(static=True)

    @eqxi.doc_remove_args("_callback_on_reject")
    def __init__(
        self,
        controller,
        step_ts=None,
        jump_ts=None,
        store_rejected_steps=None,
        _callback_on_reject=None,
    ):
        """**Arguments**:

        - `controller`: The controller to wrap.
            Can be any [`diffrax.AbstractAdaptiveStepSizeController`][].
        - `step_ts`: Denotes extra times that must be stepped to.
        - `jump_ts`: Denotes extra times that must be stepped to, and at which the
            vector field has a known discontinuity. (This is used to force FSAL solvers
            to re-evaluate the vector field.)
        `store_rejected_steps`: If this is set to a positive integer, then any
            rejected steps will have their time stored, and that time will be stepped to
            exactly in a later step. This is used when solving SDEs with noncommutative
            noise, for which this ensures that the distribution coming from Lévy area
            terms is correct. Setting this to e.g. `100` should be plenty, but if more
            consecutive steps are rejected, then a runtime error will be raised. (Note
            that this is not the total number of rejected steps in a solve, but just the
            maximum number of *consecutive* rejected steps.)
        """
        self.controller = controller
        self.step_ts = _none_or_sorted_array(step_ts)
        self.jump_ts = _none_or_sorted_array(jump_ts)
        if (store_rejected_steps is not None) and (store_rejected_steps <= 0):
            raise ValueError(
                "`store_rejected_steps must either be `None`"
                " or a non-negative integer."
            )
        self.store_rejected_steps = store_rejected_steps
        self.callback_on_reject = _callback_on_reject

    def __check_init__(self):
        if self.jump_ts is not None and not jnp.issubdtype(
            self.jump_ts.dtype, jnp.floating
        ):
            raise ValueError(
                f"jump_ts must be floating point, not {self.jump_ts.dtype}"
            )

    def wrap(self, direction: IntScalarLike):
        step_ts = None if self.step_ts is None else jnp.sort(self.step_ts * direction)
        jump_ts = None if self.jump_ts is None else jnp.sort(self.jump_ts * direction)
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
    ) -> tuple[RealScalarLike, _ClipState[_ControllerState]]:
        t_dtype = jnp.result_type(t0)
        _assert_floating(t0, "t0", t_dtype)
        _assert_floating(t1, "t1", t_dtype)
        if dt0 is not None:
            _assert_floating(dt0, "dt0", t_dtype)
        t1, inner_state = self.controller.init(
            terms, t0, t1, y0, dt0, args, func, error_order
        )
        _assert_floating(t1, "controller.init(...)", t_dtype)

        if self.step_ts is None:
            step_info = None
        else:
            step_ts = upcast_or_raise(
                self.step_ts,
                t_dtype,
                "`ClipStepSizeController.step_ts`",
                "time (the result type of `t0`, `t1`, `dt0`, `SaveAt(ts=...)` etc.)",
            )
            step_index = jnp.searchsorted(step_ts, t0, side="right")
            t1 = _clip_t(t1, step_index, step_ts, False)
            step_info = (step_index, step_ts)

        if self.jump_ts is None:
            jump_info = None
        else:
            jump_ts = upcast_or_raise(
                self.jump_ts,
                t_dtype,
                "`ClipStepSizeController.jump_ts`",
                "time (the result type of `t0`, `t1`, `dt0`, `SaveAt(ts=...)` etc.)",
            )
            jump_index = jnp.searchsorted(jump_ts, t0, side="right")
            t1 = _clip_t(t1, jump_index, jump_ts, True)
            jump_info = (jump_index, jump_ts)

        if self.store_rejected_steps is None:
            reject_info = None
        else:
            reject_ts = jnp.zeros(self.store_rejected_steps, dtype=t_dtype)
            reject_index = jnp.array(self.store_rejected_steps)
            reject_info = (reject_index, reject_ts)

        state = _ClipState(step_info, jump_info, reject_info, inner_state)
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
        controller_state: _ClipState[_ControllerState],
    ) -> tuple[
        BoolScalarLike,
        RealScalarLike,
        RealScalarLike,
        BoolScalarLike,
        _ClipState[_ControllerState],
        RESULTS,
    ]:
        t_dtype = jnp.result_type(t0)
        _assert_floating(t0, "t0", t_dtype)
        _assert_floating(t1, "t1", t_dtype)
        (
            keep_step,
            next_t0,
            next_t1,
            made_jump,
            inner_state,
            result,
        ) = self.controller.adapt_step_size(
            t0,
            t1,
            y0,
            y1_candidate,
            args,
            y_error,
            error_order,
            controller_state.inner_state,
        )
        _assert_floating(next_t0, "next_t0", t_dtype)
        _assert_floating(next_t1, "next_t1", t_dtype)

        # Logging utility for testing purposes
        callback_on_reject = self.callback_on_reject
        if callback_on_reject is not None:

            def callback(_keep_step, _t1):
                callback_on_reject(_keep_step, _t1)
                return _keep_step

            keep_step = jax.pure_callback(callback, keep_step, keep_step, t1)

        if controller_state.step_info is None:
            step_info = None
        else:
            step_index, step_ts = controller_state.step_info
            # We actaully bump `next_t0` past any `step_ts` whilst checking where to
            # clip `next_t1`. This is in case we have a set up like the following:
            # ```python
            # ClipStepSizeController(
            #     ClipStepSizeController(..., step_ts=[x]), jump_ts=[x]
            # )
            # ```
            # with a single value `x`. Otherwise in this case, the outer controller will
            # propose a step over the interval [something, prevbefore(x)], then on the
            # next step the inner controller will propose a step over [prevbefore(x), x]
            # which definitely isn't desired!
            _next_t0, _ = _bump_next_t0(next_t0, step_ts)
            step_index = _find_idx_with_hint(_next_t0, step_ts, step_index)
            next_t1 = _clip_t(next_t1, step_index, step_ts, False)
            step_info = step_index, step_ts
        if controller_state.jump_info is None:
            jump_info = None
        else:
            jump_index, jump_ts = controller_state.jump_info
            next_t0, made_jump2 = _bump_next_t0(next_t0, jump_ts)
            made_jump = made_jump | made_jump2
            jump_index = _find_idx_with_hint(next_t0, jump_ts, jump_index)
            next_t1 = _clip_t(next_t1, jump_index, jump_ts, True)
            jump_info = jump_index, jump_ts
        if controller_state.reject_info is None:
            reject_info = None
        else:
            assert self.store_rejected_steps is not None
            reject_index, reject_ts = controller_state.reject_info
            # If the step ended at `t1==reject_ts[reject_index],` then we have
            # successfully stepped to this time and we pop off this rejected time by
            # incrementing `reject_index`.
            # We do this increment even if the step is rejected, because we will
            # re-add the rejected time to the buffer immediately.
            rejected_t = _get_t(reject_index, reject_ts)
            result = RESULTS.where(
                (t1 > rejected_t) & is_okay(result), RESULTS.internal_error, result
            )
            reject_index = reject_index + jnp.where(t1 == rejected_t, 1, 0)
            # Now, if the step is rejected then we must store the rejected time in the
            # buffer.
            reject_index = reject_index - jnp.where(keep_step, 0, 1)
            result = RESULTS.where(
                (reject_index < 0) & is_okay(result), RESULTS.max_steps_rejected, result
            )
            new_rejected_t = jnp.where(keep_step, reject_ts[reject_index], t1)
            reject_ts = reject_ts.at[reject_index].set(new_rejected_t)
            next_t1 = _clip_t(next_t1, reject_index, reject_ts, False)
            reject_info = reject_index, reject_ts

        state = _ClipState(step_info, jump_info, reject_info, inner_state)
        return keep_step, next_t0, next_t1, made_jump, state, result
