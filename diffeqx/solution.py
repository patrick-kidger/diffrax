from typing import Any, Optional

from .custom_types import Array, PyTree
from .interpolation import AbstractInterpolation
from .path import AbstractPath


class Solution(AbstractPath):
    def __init__(self, ts: Array, ys: PyTree, controller_states: Optional[list[PyTree]], solver_states: Optional[list[PyTree]], interpolation: AbstractInterpolation):
        super().__init__()
        self.ts = ts
        self.ys = ys
        self.controller_states = controller_states
        self.solver_states = solver_states
        self.interpolation = interpolation

    def derivative(self, t: Scalar) -> PyTree:
        return self.interpolation.derivative(t)

    def evaluate(self, t0: Scalar, t1: Optional[Scalar] = None) -> PyTree:
        return self.interpolation.evaluate(t0, t1)

