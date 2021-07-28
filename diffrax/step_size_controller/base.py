import abc
from typing import Callable, Optional, Tuple, TypeVar

from ..custom_types import Array, PyTree, Scalar, SquashTreeDef
from ..solver import AbstractSolverState
from ..tree import tree_dataclass


T = TypeVar('T', bound=PyTree)
T2 = TypeVar('T2', bound=AbstractSolverState)


@tree_dataclass
class AbstractStepSizeController(metaclass=abc.ABCMeta):
    requested_state = frozenset()

    @abc.abstractmethod
    def init(
        self,
        func: Callable[[SquashTreeDef, Scalar, Array["state"], PyTree], Array["state"]],  # noqa: F821
        y_treedef: SquashTreeDef,
        t0: Scalar,
        y0: Array["state"],  # noqa: F821
        dt0: Optional[Scalar],
        args: PyTree,
        solver_order: int
    ) -> Tuple[Scalar, T]:
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
        solver_order: int,
        controller_state: T
    ) -> Tuple[bool, Scalar, Scalar, T]:
        pass
