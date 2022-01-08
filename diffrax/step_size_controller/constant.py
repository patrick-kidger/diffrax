from typing import Optional, Tuple

from ..custom_types import Array, PyTree, Scalar
from ..misc import error_if
from ..solution import RESULTS
from ..solver import AbstractSolver
from ..term import AbstractTerm
from .base import AbstractStepSizeController


class ConstantStepSize(AbstractStepSizeController):
    """Use a constant step size, equal to the `dt0` argument of
    [`diffrax.diffeqsolve`][].
    """

    def wrap(self, unravel_y: callable, direction: Scalar):
        return self

    def init(
        self,
        terms: PyTree[AbstractTerm],
        t0: Scalar,
        t1: Scalar,
        y0: Array["state"],  # noqa: F821
        dt0: Optional[Scalar],
        args: PyTree,
        solver: AbstractSolver,
    ) -> Tuple[Scalar, Scalar]:
        del t1, y0, args, solver
        error_if(
            dt0 is None,
            "Constant step size solvers cannot select step size automatically; "
            "please pass a value for `dt0`.",
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


class StepTo(AbstractStepSizeController):
    """Make steps to just prespecified times."""

    ts: Array["times"]  # noqa: F821

    def __post_init__(self):
        error_if(len(self.ts) < 2, "`ts` must have length at least 2.")

    def wrap(self, unravel_y: callable, direction: Scalar):
        ts = self.ts * direction
        # Only tested after we've set the direction.
        error_if(ts[1:] <= ts[:-1], "`StepTo(ts=...)` must be strictly increasing.")
        return type(self)(ts=ts)

    def init(
        self,
        terms: PyTree[AbstractTerm],
        t0: Scalar,
        t1: Scalar,
        y0: Array["state"],  # noqa: F821
        dt0: None,
        args: PyTree,
        solver: AbstractSolver,
    ) -> Tuple[Scalar, int]:
        del y0, args, solver
        error_if(
            dt0 is not None,
            "`dt0` should be `None`; step location is determined"
            f"by {type(self).__name__}(ts=...).",
        )
        error_if(
            (t0 != self.ts[0]) | (t1 != self.ts[-1]),
            "Must have `t0==ts[0]` and `t1==ts[-1]`.",
        )
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


StepTo.__init__.__doc__ = """**Arguments:**

- `ts`: The times to step to. Must be an increasing/decreasing sequence of times
    between the `t0` and `t1` passed to `diffeqsolve`.
"""
