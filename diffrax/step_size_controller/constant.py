from typing import Callable, Optional, Tuple

from ..custom_types import Array, PyTree, Scalar, SquashTreeDef
from .base import AbstractStepSizeController
from .solution import RESULTS


class ConstantStepSize(AbstractStepSizeController):
    def init(
        self,
        t0: Scalar,
        y0: Array["state"],  # noqa: F821
        dt0: Optional[Scalar],
        args: PyTree,
        y_treedef: SquashTreeDef,
        solver_order: int,
        func: Callable[[SquashTreeDef, Scalar, Array["state"], PyTree], Array["state"]],  # noqa: F821
    ) -> Tuple[Scalar, Scalar]:
        del func, y_treedef, y0, args, solver_order
        if dt0 is None:
            raise ValueError(
                "Constant step size solvers cannot select step size automatically; please pass a "
                "value for `dt0`."
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
        y_treedef: SquashTreeDef,
        solver_order: int,
        controller_state: Scalar
    ) -> Tuple[bool, Scalar, Scalar, Scalar, int]:
        del t0, y0, args, y_error, y_treedef, solver_order
        return True, t1, t1 + controller_state, controller_state, RESULTS.successful
