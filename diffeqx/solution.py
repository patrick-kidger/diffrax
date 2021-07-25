from typing import Any, Optional

from .custom_types import Array, PyTree
from .interpolation import AbstractInterpolation
from .path import AbstractPath


class Solution(AbstractPath):
    def __init__(self, ts: Array, ys: PyTree, controller_state: Optional[Any], solver_state: Optional[Any], interpolation: AbstractInterpolation):
        super().__init__()
        self.ts = ts
        self.ys = ys
        self.controller_state = controller_state
        self.solver_state = solver_state
        self.interpolation = interpolation

    def derivative(self, t: Scalar) -> PyTree:
        return self.interpolation.derivative(t)

    def evaluate(self, t0: Scalar, t1: Scalar) -> PyTree:
        return self.interpolation.evaluate(t0, t1)

