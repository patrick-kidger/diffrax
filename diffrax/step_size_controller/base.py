import abc
from typing import Optional, Tuple, TypeVar

import equinox as eqx

from ..custom_types import Array, PyTree, Scalar
from ..solver import AbstractSolver


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
        t1: Scalar,
        y0: Array["state"],  # noqa: F821
        dt0: Optional[Scalar],
        args: PyTree,
        solver: AbstractSolver,
    ) -> Tuple[Scalar, _ControllerState]:
        # returns initial t1, initial controller state
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
    ) -> Tuple[Array[(), bool], Scalar, Scalar, Array[(), bool], _ControllerState, int]:
        # returns:
        # - Whether the step was accepted/rejected
        # - next t0
        # - next t1
        # - Whether a jump has just been made (i.e. FSAL is invalidated)
        # - Any controller state
        # - A result potentially indicating a failure condition
        pass
