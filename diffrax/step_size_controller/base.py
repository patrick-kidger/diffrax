import abc
import equinox as eqx
from typing import Callable, Optional, Tuple, TypeVar

from ..custom_types import Array, PyTree, Scalar, SquashTreeDef
from ..solver import AbstractSolverState


T = TypeVar('T', bound=PyTree)
T2 = TypeVar('T2', bound=AbstractSolverState)


class AbstractStepSizeController(eqx.Module):
    requested_state = frozenset()

    @abc.abstractmethod
    def init(
        self,
        t0: Scalar,
        y0: Array["state"],  # noqa: F821
        dt0: Optional[Scalar],
        args: PyTree,
        y_treedef: SquashTreeDef,
        solver_order: int,
        func: Callable[[SquashTreeDef, Scalar, Array["state"], PyTree], Array["state"]],  # noqa: F821
    ) -> Tuple[Scalar, T]:
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
        y_treedef: SquashTreeDef,
        solver_order: int,
        controller_state: T
    ) -> Tuple[bool, Scalar, Scalar, T, int]:
        pass
