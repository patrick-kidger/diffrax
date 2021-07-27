from typing import Optional

from .custom_types import Array, PyTree, Scalar
from .path import AbstractPath
from .tree import tree_dataclass


@tree_dataclass
class AbstractInterpolation(AbstractPath):
    ts: Array
    ys: PyTree


@tree_dataclass
class LinearInterpolation(AbstractInterpolation):
    def derivative(self, t: Scalar) -> PyTree:
        ...

    def evaluate(self, t0: Scalar, t1: Optional[Scalar] = None) -> PyTree:
        ...  # TODO. Think about point evaluations?
