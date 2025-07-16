from collections.abc import Callable

import equinox as eqx
import equinox.internal as eqxi
import jax.numpy as jnp
from jaxtyping import Array, PyTree, Real

from .._custom_types import Args, IntScalarLike, RealScalarLike, VF, Y
from .._solution import RESULTS
from .._term import AbstractTerm
from .base import AbstractStepSizeController


# ConstantStepSizeState = (steps_completed, num_steps, t0_sim, t1_sim_or_dt0)
_ConstantStepSizeState = tuple[
    IntScalarLike, IntScalarLike, RealScalarLike, RealScalarLike
]


class ConstantStepSize(
    AbstractStepSizeController[_ConstantStepSizeState, RealScalarLike]
):
    """Use a constant step size, equal to the `dt0` argument of
    [`diffrax.diffeqsolve`][].
    """

    def wrap(self, direction: IntScalarLike):
        return self

    def init(
        self,
        terms: PyTree[AbstractTerm],
        t0: RealScalarLike,
        t1: RealScalarLike,
        y0: Y,
        dt0: RealScalarLike | None,
        args: Args,
        func: Callable[[PyTree[AbstractTerm], RealScalarLike, Y, Args], VF],
        error_order: RealScalarLike | None,
    ) -> tuple[RealScalarLike, _ConstantStepSizeState]:
        del terms, y0, args, func, error_order
        if dt0 is None:
            raise ValueError(
                "Constant step size solvers cannot select step size automatically; "
                "please pass a value for `dt0`."
            )
        steps_completed = jnp.asarray(1, dtype=jnp.int32)
        # Special case for infinite t1, allow termination based on conditional tests
        # Use num_steps=-1 to ensure finite int
        num_steps = jnp.where(
            jnp.isfinite(t1),
            jnp.astype(jnp.ceil((t1 - t0) / eqxi.nextafter(dt0)), jnp.int32),
            -1,
        )
        t1_sim_or_dt0 = jnp.where(jnp.isfinite(t1), t1, dt0)
        return t0 + dt0, (steps_completed, num_steps, t0, t1_sim_or_dt0)

    def adapt_step_size(
        self,
        t0: RealScalarLike,
        t1: RealScalarLike,
        y0: Y,
        y1_candidate: Y,
        args: Args,
        y_error: Y | None,
        error_order: RealScalarLike | None,
        controller_state: _ConstantStepSizeState,
    ) -> tuple[
        bool,
        RealScalarLike,
        RealScalarLike,
        bool,
        _ConstantStepSizeState,
        RESULTS,
    ]:
        del t0, y0, y1_candidate, args, y_error, error_order
        steps_already_completed, num_steps, t0_sim, t1_sim_or_dt0 = controller_state
        # Number of steps that will be completed when this function returns.
        steps_completed = steps_already_completed + 1

        time_dtype = jnp.result_type(t0_sim, t1_sim_or_dt0)

        # Calculate step size by calculating fraction of `t1 - t0` to avoid compounding
        # of truncation/rounding errors
        t1_next = jnp.where(
            num_steps >= 0,
            jnp.where(
                steps_completed == num_steps,
                t1_sim_or_dt0,
                t0_sim
                + (t1_sim_or_dt0 - t0_sim)
                * jnp.astype(steps_completed, time_dtype)
                / jnp.astype(num_steps, time_dtype),
            ),
            # Special case for non-finite t1_sim
            # in this t1_sim_or_dt0 is dt0
            t1 + t1_sim_or_dt0,
        )

        return (
            True,
            t1,
            t1_next,
            False,
            (steps_completed, num_steps, t0_sim, t1_sim_or_dt0),
            RESULTS.successful,
        )


ConstantStepSize.__init__.__doc__ = """**Arguments:**

None.
"""


class StepTo(AbstractStepSizeController[IntScalarLike, None]):
    """Make steps to just prespecified times."""

    ts: Real[Array, " times"] = eqx.field(converter=jnp.asarray)

    def __check_init__(self):
        if self.ts.ndim != 1:
            raise ValueError("`ts` must be one-dimensional.")
        if len(self.ts) < 2:
            raise ValueError("`ts` must have length at least 2.")

    def wrap(self, direction: IntScalarLike):
        ts = self.ts * direction
        # Only tested after we've set the direction.
        ts = eqxi.error_if(
            ts,
            ts[1:] <= ts[:-1],
            "`StepTo(ts=...)` must be strictly increasing (or strictly decreasing if "
            "t0 > t1).",
        )
        return type(self)(ts=ts)

    def init(
        self,
        terms: PyTree[AbstractTerm],
        t0: RealScalarLike,
        t1: RealScalarLike,
        y0: Y,
        dt0: None,
        args: Args,
        func: Callable[[PyTree[AbstractTerm], RealScalarLike, Y, Args], Y],
        error_order: RealScalarLike | None,
    ) -> tuple[RealScalarLike, IntScalarLike]:
        del y0, args, func, error_order
        if dt0 is not None:
            raise ValueError(
                "`dt0` should be `None`. Step location is already determined "
                f"by {type(self).__name__}(ts=...).",
            )
        ts = eqxi.error_if(
            self.ts,
            (t0 != self.ts[0]) | (t1 != self.ts[-1]),
            "Must have `t0==ts[0]` and `t1==ts[-1]`.",
        )
        return ts[1], 2

    def adapt_step_size(
        self,
        t0: RealScalarLike,
        t1: RealScalarLike,
        y0: Y,
        y1_candidate: Y,
        args: Args,
        y_error: Y | None,
        error_order: RealScalarLike | None,
        controller_state: IntScalarLike,
    ) -> tuple[bool, RealScalarLike, RealScalarLike, bool, IntScalarLike, RESULTS]:
        del t0, y0, y1_candidate, args, y_error, error_order
        return (
            True,
            t1,
            self.ts[controller_state],
            False,
            controller_state + 1,
            RESULTS.successful,
        )


StepTo.__init__.__doc__ = """**Arguments:**

- `ts`: The times to step to. Must be an increasing/decreasing sequence of times
    between the `t0` and `t1` (inclusive) passed to [`diffrax.diffeqsolve`][].
    Correctness of `ts` with respect to `t0` and `t1` as well as its
    monotonicity is checked by the implementation.
"""
