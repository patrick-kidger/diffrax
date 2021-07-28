import abc
from typing import Any, Optional, Tuple, TypeVar

from ..custom_types import Array, PyTree, Scalar, SquashTreeDef
from ..tree import tree_dataclass


@tree_dataclass
class AbstractSolverState(metaclass=abc.ABCMeta):
    extras: dict[str, Any]


@tree_dataclass
class EmptySolverState(AbstractSolverState):
    pass


T = TypeVar('T', bound=Optional[AbstractSolverState])


@tree_dataclass
class AbstractSolver(metaclass=abc.ABCMeta):
    @property
    @abc.abstractmethod
    def order(self) -> int:
        pass

    @property
    def available_state(self) -> frozenset:
        return frozenset()

    @property
    @abc.abstractmethod
    def recommended_interpolation(self) -> T:
        pass

    def init(
        self,
        y_treedef: SquashTreeDef,
        t0: Scalar,
        t1: Scalar,
        y0: Array["state"],  # noqa: F821
        args: PyTree,
        requested_state: frozenset,
    ) -> T:
        return None

    @abc.abstractmethod
    def step(
        self,
        y_treedef: SquashTreeDef,
        t0: Scalar,
        t1: Scalar,
        y0: Array["state"],  # noqa: F821
        args: PyTree,
        solver_state: T,
        requested_state: frozenset,
    ) -> Tuple[Array["state"], T]:  # noqa: F821
        pass

    def func(self, y_treedef: SquashTreeDef, t: Scalar, y_: Array["state"],  # noqa: F821
             args: PyTree) -> Array["state"]:  # noqa: F821
        raise ValueError(f"func does not exist for solver of type {type(self)}.")
