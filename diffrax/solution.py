from typing import Optional

from .custom_types import Array, PyTree, Scalar
from .interpolation import AbstractInterpolation
from .path import AbstractPath
from .tree import tree_dataclass


@tree_dataclass
class Solution(AbstractPath):
    ts: Array
    ys: PyTree
    controller_states: Optional[list[PyTree]]
    solver_states: Optional[list[PyTree]]
    interpolation: AbstractInterpolation

    def derivative(self, t: Scalar) -> PyTree:
        return self.interpolation.derivative(t)

    def evaluate(self, t0: Scalar, t1: Optional[Scalar] = None) -> PyTree:
        return self.interpolation.evaluate(t0, t1)
