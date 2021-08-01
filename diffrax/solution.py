from dataclasses import field
from typing import Optional

from .custom_types import Array, PyTree, Scalar
from .global_interpolation import DenseInterpolation
from .misc import ContainerMeta
from .path import AbstractPath


# We use magic numbers, rather than informative strings, as these can be vmap'd etc.
# through JAX. Same reason we don't use enum.Enum here.
class RESULTS(metaclass=ContainerMeta):
    successful = 0
    max_steps_reached = 1
    dt_min_reached = 2


class Solution(AbstractPath):
    t0: Scalar = field(init=True)
    t1: Scalar = field(init=True)  # override init=False in AbstractPath
    ts: Optional[Array]
    ys: Optional[PyTree]
    controller_states: Optional[list[PyTree]]
    solver_states: Optional[list[PyTree]]
    interpolation: Optional[DenseInterpolation]
    result: int  # from RESULTS

    def derivative(self, t: Scalar, left: bool = True) -> PyTree:
        if self.interpolation is None:
            raise ValueError(
                "Dense solution has not been saved; pass saveat.dense=True."
            )
        return self.interpolation.derivative(t, left)

    def evaluate(
        self, t0: Scalar, t1: Optional[Scalar] = None, left: bool = True
    ) -> PyTree:
        if self.interpolation is None:
            raise ValueError(
                "Dense solution has not been saved; pass saveat.dense=True."
            )
        return self.interpolation.evaluate(t0, t1, left)
