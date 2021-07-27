from dataclasses import dataclass
from typing import Optional

from .custom_types import Array, PyTree, Scalar
from .path import AbstractPath


@dataclass(frozen=True)
class AbstractInterpolation(AbstractPath):
    ts: Array
    ys: PyTree


class LinearInterpolation(AbstractInterpolation):
    def derivative(self, t: Scalar) -> PyTree:
        ...

    def evaluate(self, t0: Scalar, t1: Optional[Scalar] = None) -> PyTree:
        ...  # TODO. Think about point evaluations?
