import abc
from typing import Optional, Tuple, TypeVar

from .custom_types import Array, PyTree, Scalar


T = TypeVar('T', bound=PyTree)
T2 = TypeVar('T2', bound=PyTree)


class AbstractStepSizeController(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def init(t0: Scalar, dt0: Optional[Scalar]) -> Tuple[Scalar, T]:
        pass

    @abc.abstractmethod
    def adapt_step_size(
        self,
        t0: Scalar,
        t1: Scalar,
        y0: Array["state":...],  # noqa: F821
        y1_candidate: Array["state":...],  # noqa: F821
        solver_state0: T2,
        solver_state1_candidate: T2,
        controller_state: T
    ) -> Tuple[bool, Scalar, Scalar, T]:
        pass


class ConstantStepSize(AbstractStepSizeController):
    def init(t0: Scalar, dt0: Scalar) -> Tuple[Scalar, Scalar]:
        controller_state = dt0
        return t0 + dt0, controller_state

    def adapt_step_size(
        self,
        t0: Scalar,
        t1: Scalar,
        y0: Array["state":...],  # noqa: F821
        y1_candidate: Array["state":...],  # noqa: F821
        solver_state0: T2,
        solver_state1_candidate: T2,
        controller_state: Scalar
    ) -> Tuple[bool, Scalar, Scalar, Scalar]:
        del t0, y0, solver_state0, solver_state1_candidate
        keep_step = True
        dt0 = controller_state
        return keep_step, t1, t1 + dt0, controller_state
