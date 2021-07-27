import abc
from typing import Tuple, TypeVar

from ..custom_types import Array, PyTree, Scalar, SquashTreeDef
from ..interpolation import AbstractInterpolation

# Abusing types slightly to represent PyTrees with the same structure
T = TypeVar('T', bound=PyTree)
T2 = TypeVar('T2', bound=PyTree)


class AbstractSolver(metaclass=abc.ABCMeta):
    recommended_interpolation: AbstractInterpolation

    def init(self, y_treedef: SquashTreeDef, t0: Scalar, y0: Array["state"], args: PyTree) -> T:  # noqa: F821
        return None

    @abc.abstractmethod
    def step(
        self,
        y_treedef: SquashTreeDef,
        t0: Scalar,
        t1: Scalar,
        y0: Array["state"],  # noqa: F821
        args: PyTree,
        solver_state: T
    ) -> Tuple[Array["state"], T]:  # noqa: F821
        pass
