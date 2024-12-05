from collections.abc import Callable
from typing import Generic, get_args, Optional, TYPE_CHECKING, TypeVar

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
from .base import AbstractStepSizeController


_ControllerState = TypeVar("_ControllerState")
_Dt0 = TypeVar("_Dt0", None, RealScalarLike, Optional[RealScalarLike])


class _JumpStepState(eqx.Module, Generic[_ControllerState]):
    jump_at_next_t1: BoolScalarLike
    step_index: IntScalarLike
    jump_index: IntScalarLike
    rejected_index: IntScalarLike
    rejected_buffer: Optional[Array]
    step_ts: Optional[Array]
    jump_ts: Optional[Array]
    inner_state: _ControllerState


def _none_or_sorted_array(x):
    if x is None:
        return None
    else:
        return jnp.sort(jnp.asarray(x))


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
    jump_at_t1 = _t1 <= t1
    _t1 = jnp.where(jump_at_t1, _t1, t1)
    return _t1, jump_at_t1


def _find_idx_with_hint(t: RealScalarLike, ts: Optional[Array], hint: IntScalarLike):
    # Find index of first element of ts greater than t
    # using linear search starting from hint.
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


class JumpStepWrapper(
    AbstractStepSizeController[_JumpStepState[_ControllerState], _Dt0]
):
    """Wraps an existing step controller and adds the ability to specify `step_ts`
    and `jump_ts`. It also enables the feature of revisiting rejected steps, which
    is useful when solving SDEs with an adaptive step controller.

    Explanation of `step_ts` and `jump_ts`:

    The `step_ts` and `jump_ts` are used to force the solver to step to certain times.
    They mostly act in the same way, except that when we hit an element of `jump_ts`,
    the controller must return `made_jump = True`, so that the diffeqsolve function
    knows that the vector field has a discontinuity at that point, in which case it
    re-evaluates it right after the jump point. In addition, the
    exact time of the jump will be skipped using eqxi.prevbefore and eqxi.nextafter.
    So now to the explanation of the two (we will use `step_ts` as an example, but the
    same applies to `jump_ts`):

    If `step_ts` is not None, we assume it is a sorted array of times.
    At the start of the run, the init function finds the smallest index `i_step` such
    that `step_ts[i_step] > t0`. At init and after each step of the solver, the
    controller will propose a step t1_next, and we will clip it to
    `t1_next = min(t1_next, step_ts[i_step])`.
    At the start of the next step, if the step ended at t1 == step_ts[i_step] and
    if the controller decides to keep the step, then this time has been successfully
    stepped to and we increment `i_step` by 1.
    We use a convenience function _get_t(i, ts) which returns ts[i] if i < len(ts) and
    infinity otherwise.

    Explanation of revisiting rejected steps:

    This feature should be used if and only if solving SDEs with non-commutative noise
    using an adaptive step controller.

    We use a "stack" of rejected steps, composed of a buffer `rejected_buffer` of length
    `rejected_step_buffer_len` and a counter `i_reject`. The "stack" are all the items
    in `rejected_buffer[i_reject:]` with `rejected_buffer[i_reject]` being the top of
    the stack.
    When `i_reject == rejected_step_buffer_len`, the stack is empty.
    At the start of the run, `i_reject = rejected_step_buffer_len`. Each time a step is
    rejected `i_reject -=1` and `rejected_buffer[i_reject] = t1`. Each time a step ends
    at `t1 == rejected_buffer[i_reject]`, we increment `i_reject` by 1 (even if the
    step was rejected, in which case we will re-add `t1` to the stack immediately).
    We clip the next step to `t1_next = min(t1_next, rejected_buffer[i_reject])`.
    If `i_reject < 0` then an error is raised.
    """

    # For more details on solving SDEs with adaptive stepping see
    # docs/api/stepsize_controller.md
    # I am putting this outside of the docstring, because this class appears in that
    # part of the docs and I don't want to repeat the same thing twice on one page.
    # For more details also refer to
    # ```bibtex
    #     @misc{foster2024convergenceadaptiveapproximationsstochastic,
    #         title={On the convergence of adaptive approximations for
    #                   stochastic differential equations},
    #         author={James Foster and Andraž Jelinčič},
    #         year={2024},
    #         eprint={2311.14201},
    #         archivePrefix={arXiv},
    #         primaryClass={math.NA},
    #         url={https://arxiv.org/abs/2311.14201},
    #     }
    # ```

    controller: AbstractStepSizeController[_ControllerState, _Dt0]
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
            Can be any [`diffrax.AbstractAdaptiveStepSizeController`][].
        - `step_ts`: Denotes extra times that must be stepped to.
        - `jump_ts`: Denotes extra times that must be stepped to, and at which the
            vector field has a known discontinuity. (This is used to force FSAL solvers
            to re-evaluate the vector field.)
        `rejected_step_buffer_len`: Length of the stack used to store rejected steps.
            Can either be `None` or a positive integer.
            If `None`, this feature will be off.
            If it is > 0, then the controller will revisit rejected steps.
            This should only be used when solving SDEs with an adaptive step size
            controller. For most SDEs, setting this to `100` should be plenty,
            but if more consecutive steps are rejected, then an error will be raised.
            (Note that this is not the total number of rejected steps in a solve,
            but just the number of rejected steps currently on the stack to be
            revisited.)
        """
        self.controller = controller
        self.step_ts = _none_or_sorted_array(step_ts)
        self.jump_ts = _none_or_sorted_array(jump_ts)
        if (rejected_step_buffer_len is not None) and (rejected_step_buffer_len <= 0):
            raise ValueError(
                "`rejected_step_buffer_len must either be `None`"
                " or a non-negative integer."
            )
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
    ) -> tuple[RealScalarLike, _JumpStepState[_ControllerState]]:
        t1, inner_state = self.controller.init(
            terms, t0, t1, y0, dt0, args, func, error_order
        )
        tdtype = jnp.result_type(t0, t1)

        if self.step_ts is None:
            step_ts = None
        else:
            # Upcast step_ts to the same dtype as t0, t1
            step_ts = upcast_or_raise(
                self.step_ts,
                tdtype,
                "`JumpStepWrapper.step_ts`",
                "time (the result type of `t0`, `t1`, `dt0`, `SaveAt(ts=...)` etc.)",
            )

        if self.jump_ts is None:
            jump_ts = None
        else:
            # Upcast jump_ts to the same dtype as t0, t1
            jump_ts = upcast_or_raise(
                self.jump_ts,
                tdtype,
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
        # just shortening the name
        st = controller_state
        i_step = st.step_index
        i_jump = st.jump_index
        i_reject = st.rejected_index

        # Let the controller do its thing
        (
            keep_step,
            next_t0,
            original_next_t1,
            jump_at_original_next_t1,
            inner_state,
            result,
        ) = self.controller.adapt_step_size(
            t0, t1, y0, y1_candidate, args, y_error, error_order, st.inner_state
        )
        next_t1 = original_next_t1

        # This is just a logging utility for testing purposes
        if self.callback_on_reject is not None:
            # jax.debug.callback(self.callback_on_reject, keep_step, t1)
            jax.experimental.io_callback(self.callback_on_reject, None, keep_step, t1)  # pyright: ignore

        # For step ts and jump ts find the index of the first element in jump_ts/step_ts
        # greater than next_t0. We use the hint i_step/i_jump to speed up the search.
        i_step = _find_idx_with_hint(next_t0, st.step_ts, i_step)
        i_jump = _find_idx_with_hint(next_t0, st.jump_ts, i_jump)

        if self.rejected_step_buffer_len is not None:
            rejected_buffer = st.rejected_buffer
            assert rejected_buffer is not None
            # If the step ended at t1==rejected_buffer[i_reject], then we have
            # successfully stepped to this time and we increment i_reject.
            # We increment i_reject even if the step was rejected, because we will
            # re-add the rejected time to the buffer immediately.
            rejected_t = _get_t(i_reject, rejected_buffer)
            rjct_inc_cond = t1 == rejected_t
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
        else:
            rejected_buffer = None

        # Now move on to the NEXT STEP

        # If t1 hit a jump point, and the step was kept then we need to set
        # `next_t0 = nextafter(nextafter(t1))` to ensure that we really skip
        # over the jump and don't evaluate the vector field at the discontinuity.
        if jnp.issubdtype(jnp.result_type(next_t0), jnp.inexact):
            # Two nextafters. If made_jump then t1 = prevbefore(jump location)
            # so now _t1 = nextafter(jump location)
            # This is important because we don't know whether or not the jump is as a
            # result of a left- or right-discontinuity, so we have to skip the jump
            # location altogether.
            jump_keep = st.jump_at_next_t1 & keep_step
            next_t0 = static_select(
                jump_keep, eqxi.nextafter(eqxi.nextafter(next_t0)), next_t0
            )

        if TYPE_CHECKING:  # if i don't seperate this out pyright complains
            assert isinstance(next_t0, RealScalarLike)
        else:
            assert isinstance(
                next_t0, get_args(RealScalarLike)
            ), f"type(next_t0) = {type(next_t0)}"

        # Clip the step to the next element of jump_ts or step_ts or
        # rejected_buffer. Important to do jump_ts last because otherwise
        # jump_at_next_t1 could be a false positive.
        next_t1 = _revisit_rejected(next_t0, next_t1, i_reject, rejected_buffer)
        next_t1, _ = _clip_ts(next_t0, next_t1, i_step, st.step_ts, False)
        next_t1, jump_at_next_t1 = _clip_ts(next_t0, next_t1, i_jump, st.jump_ts, True)

        # Let's prove that the line below is correct. Say the inner controller is
        # itself a JumpStepWrapper (JSW) with some inner_jump_ts. Then, given that
        # it propsed (next_t0, original_next_t1), there cannot be any jumps in
        # inner_jump_ts between next_t0 and original_next_t1. So if the next_t1
        # proposed by the outer JSW is different from the original_next_t1 then
        # next_t1 \in (next_t0, original_next_t1) and hence there cannot be a jump
        # in inner_jump_ts at next_t1. So the jump_at_next_t1 only depends on
        # jump_at_next_t1.
        # On the other hand if original_next_t1 == next_t1, then we just take an
        # OR of the two.
        jump_at_next_t1 = jnp.where(
            next_t1 == original_next_t1,
            jump_at_next_t1 | jump_at_original_next_t1,
            jump_at_next_t1,
        )

        # Here made_jump signifies whether there is a jump at t1. What the solver
        # needs, however, is whether there is a jump at next_t0, so these two will
        # only match when the step was kept. The case when the step was rejected is
        # handled in `_integrate.py` (search for "made_jump = static_select").
        made_jump = st.jump_at_next_t1

        state = _JumpStepState(
            jump_at_next_t1,
            i_step,
            i_jump,
            i_reject,
            rejected_buffer,
            st.step_ts,
            st.jump_ts,
            inner_state,
        )

        return keep_step, next_t0, next_t1, made_jump, state, result
