from typing import Optional

from .custom_types import Array, PyTree, Scalar
from .interpolation import AbstractInterpolation
from .path import AbstractPath
from .tree import tree_dataclass


# We use magic numbers, rather than informative strings, as these can be vmap'd etc. through JAX.
# Same reason we don't use enum.Enum here.
class RESULTS:
    successful = 0
    max_steps_reached = 1


@tree_dataclass
class Solution(AbstractPath):
    ts: Array
    ys: PyTree
    controller_states: Optional[list[PyTree]]
    solver_states: Optional[list[PyTree]]
    interpolation: AbstractInterpolation
    result: int  # from RESULTS

    def derivative(self, t: Scalar) -> PyTree:
        return self.interpolation.derivative(t)

    def evaluate(self, t0: Scalar, t1: Optional[Scalar] = None) -> PyTree:
        return self.interpolation.evaluate(t0, t1)
