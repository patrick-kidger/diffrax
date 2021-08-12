import abc
from typing import Callable, Optional, Tuple, TypeVar

import equinox as eqx

from ..custom_types import Array, PyTree, Scalar


_ControllerState = TypeVar("_ControllerState", bound=PyTree)


class AbstractStepSizeController(eqx.Module):
    @abc.abstractmethod
    def wrap(
        self, unravel_y: callable, direction: Scalar
    ) -> "AbstractStepSizeController":
        pass

    @abc.abstractmethod
    def init(
        self,
        t0: Scalar,
        y0: Array["state"],  # noqa: F821
        dt0: Optional[Scalar],
        args: PyTree,
        solver_order: int,
        func: Callable[
            [Scalar, Array["state"], PyTree],  # noqa: F821
            Array["state"],  # noqa: F821
        ],
    ) -> Tuple[Scalar, _ControllerState]:
        pass

    @abc.abstractmethod
    def adapt_step_size(
        self,
        t0: Scalar,
        t1: Scalar,
        y0: Array["state":...],  # noqa: F821
        y1_candidate: Array["state":...],  # noqa: F821
        args: PyTree,
        y_error: Array["state"],  # noqa: F821
        solver_order: int,
        controller_state: _ControllerState,
    ) -> Tuple[bool, Scalar, Scalar, _ControllerState, int]:
        pass
