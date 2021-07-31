from typing import Callable, Optional, Tuple

from ..custom_types import Array, PyTree, Scalar, SquashTreeDef
from .base import AbstractStepSizeController


class ConstantStepSize(AbstractStepSizeController):
    def init(
        self,
        func: Callable[[SquashTreeDef, Scalar, Array["state"], PyTree], Array["state"]],  # noqa: F821
        y_treedef: SquashTreeDef,
        t0: Scalar,
        y0: Array["state"],  # noqa: F821
        dt0: Optional[Scalar],
        args: PyTree,
        solver_order: int
    ) -> Tuple[Scalar, Scalar]:
        del func, y_treedef, y0, args, solver_order
        assert dt0 is not None
        return t0 + dt0, dt0

    def adapt_step_size(
        self,
        t0: Scalar,
        t1: Scalar,
        y0: Array["state":...],  # noqa: F821
        y1_candidate: Array["state":...],  # noqa: F821
        solver_state0,
        solver_state1_candidate,
        solver_order: int,
        controller_state: Scalar
    ) -> Tuple[bool, Scalar, Scalar, Scalar, int]:
        del t0, y0, solver_state0, solver_state1_candidate, solver_order
        return True, t1, t1 + controller_state, controller_state, 0
