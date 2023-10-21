from collections.abc import Callable
from typing import Optional

import equinox as eqx
import equinox.internal as eqxi
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike, PyTree

from .._custom_types import IntScalarLike, Real, RealScalarLike
from .._solution import RESULTS
from .._term import AbstractTerm
from .base import AbstractStepSizeController


class ConstantStepSize(AbstractStepSizeController):
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
        y0: PyTree,
        dt0: Optional[RealScalarLike],
        args: PyTree,
        func: Callable[[RealScalarLike, PyTree[ArrayLike], PyTree], PyTree[ArrayLike]],
        error_order: Optional[RealScalarLike],
    ) -> tuple[RealScalarLike, RealScalarLike]:
        del terms, t1, y0, args, func, error_order
        if dt0 is None:
            raise ValueError(
                "Constant step size solvers cannot select step size automatically; "
                "please pass a value for `dt0`."
            )
        return t0 + dt0, dt0

    def adapt_step_size(
        self,
        t0: RealScalarLike,
        t1: RealScalarLike,
        y0: PyTree[ArrayLike],
        y1_candidate: PyTree[ArrayLike],
        args: PyTree,
        y_error: PyTree,
        error_order: RealScalarLike,
        controller_state: RealScalarLike,
    ) -> tuple[bool, RealScalarLike, RealScalarLike, bool, RealScalarLike, RESULTS]:
        del t0, y0, y1_candidate, args, y_error, error_order
        return (
            True,
            t1,
            t1 + controller_state,
            False,
            controller_state,
            RESULTS.successful,
        )


class StepTo(AbstractStepSizeController):
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
        y0: PyTree,
        dt0: None,
        args: PyTree,
        func: Callable[[RealScalarLike, PyTree[ArrayLike], PyTree], PyTree[ArrayLike]],
        error_order: Optional[RealScalarLike],
    ) -> tuple[RealScalarLike, int]:
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
        y0: Real[Array, " state"],
        y1_candidate: Real[Array, " state"],
        args: PyTree,
        y_error: Real[Array, " state"],
        error_order: Optional[RealScalarLike],
        controller_state: int,
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
