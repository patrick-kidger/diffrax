import abc
from typing import Tuple, Type, TypeVar

from ..custom_types import Array, PyTree, Scalar, SquashTreeDef
from ..interpolation import AbstractInterpolation, LinearInterpolation

# Abusing types slightly to represent PyTrees with the same structure
T = TypeVar('T', bound=PyTree)
T2 = TypeVar('T2', bound=PyTree)


class AbstractSolver(metaclass=abc.ABCMeta):
    recommended_interpolation: Type[AbstractInterpolation] = LinearInterpolation

    def init(
        self, y_treedef: SquashTreeDef, t0: Scalar, t1: Scalar, y0: Array["state"], args: PyTree  # noqa: F821
    ) -> T:  # noqa: F821
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
