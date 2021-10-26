from dataclasses import field
from typing import Any, Dict, Optional

from .custom_types import Array, PyTree, Scalar
from .global_interpolation import DenseInterpolation
from .misc import ContainerMeta
from .path import AbstractPath


class RESULTS(metaclass=ContainerMeta):
    successful = ""
    max_steps_reached = "The maximum number of solver steps was reached."
    dt_min_reached = "The minimum step size was reached."
    nan_time = "NaN time encountered during timestepping."
    implicit_divergence = "Implicit method diverged."
    implicit_nonconvergence = (
        "Implicit method did not converge within the required number of iterations."
    )


class Solution(AbstractPath):
    t0: Scalar = field(init=True)
    t1: Scalar = field(init=True)  # override init=False in AbstractPath
    ts: Optional[Array]
    ys: Optional[PyTree]
    controller_state: Optional[PyTree]
    solver_state: Optional[PyTree]
    interpolation: Optional[DenseInterpolation]
    stats: Dict[str, Any]
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

    @property
    def message(self):
        return RESULTS[self.result]
