from typing import Optional, Tuple

import jax
import jax.numpy as jnp

from ..custom_types import Array, PyTree, Scalar
from ..misc import unvmap
from ..solution import RESULTS
from ..solver import AbstractSolver
from .base import AbstractStepSizeController


class ConstantStepSize(AbstractStepSizeController):
    """Use a constant step size, equal to the `dt0` argument of
    [`diffrax.diffeqsolve`][].
    """

    def wrap(self, unravel_y: callable, direction: Scalar):
        return self

    def init(
        self,
        t0: Scalar,
        t1: Scalar,
        y0: Array["state"],  # noqa: F821
        dt0: Optional[Scalar],
        args: PyTree,
        solver: AbstractSolver,
    ) -> Tuple[Scalar, Scalar]:
        del t1, y0, args, solver
        if dt0 is None:
            raise ValueError(
                "Constant step size solvers cannot select step size automatically; "
                "please pass a value for `dt0`."
            )
        return t0 + dt0, dt0

    def adapt_step_size(
        self,
        t0: Scalar,
        t1: Scalar,
        y0: Array["state"],  # noqa: F821
        y1_candidate: Array["state"],  # noqa: F821
        args: PyTree,
        y_error: Array["state"],  # noqa: F821
        solver_order: int,
        controller_state: Scalar,
    ) -> Tuple[bool, Scalar, Scalar, bool, Scalar, RESULTS]:
        del t0, y0, y1_candidate, args, y_error, solver_order
        return (
            True,
            t1,
            t1 + controller_state,
            False,
            controller_state,
            RESULTS.successful,
        )


@jax.jit
def _bad_ts(ts):
    if len(ts) < 2:
        return True
    return jnp.any(unvmap(ts[1:] <= ts[:-1]))


@jax.jit
def _bad_t0_t1(t0, t1, ts):
    return jnp.any(unvmap((t0 != ts[0]) | (t1 != ts[-1])))


class StepToLocation(AbstractStepSizeController):
    ts: Array["times"]  # noqa: F821

    def __post_init__(self):
        if _bad_ts(self.ts):
            raise ValueError("`ts` must be strictly increasing.")

    def wrap(self, unravel_y: callable, direction: Scalar):
        ts = self.ts * direction
        return type(self)(ts=ts)

    def init(
        self,
        t0: Scalar,
        t1: Scalar,
        y0: Array["state"],  # noqa: F821
        dt0: None,
        args: PyTree,
        solver: AbstractSolver,
    ) -> Tuple[Scalar, int]:
        del y0, args, solver
        if dt0 is not None:
            raise ValueError(
                "`dt0` should be `None`; step location is determined"
                f"by {type(self).__name__}(ts=...)."
            )
        if _bad_t0_t1(t0, t1, self.ts):
            raise ValueError("Must have `t0==ts[0]` and `t1==ts[-1]`.")
        return self.ts[1], 2

    def adapt_step_size(
        self,
        t0: Scalar,
        t1: Scalar,
        y0: Array["state"],  # noqa: F821
        y1_candidate: Array["state"],  # noqa: F821
        args: PyTree,
        y_error: Array["state"],  # noqa: F821
        solver_order: int,
        controller_state: int,
    ) -> Tuple[bool, Scalar, Scalar, bool, int, RESULTS]:
        del t0, y0, y1_candidate, args, y_error, solver_order
        return (
            True,
            t1,
            self.ts[controller_state],
            False,
            controller_state + 1,
            RESULTS.successful,
        )
