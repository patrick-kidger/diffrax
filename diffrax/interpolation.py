from typing import Optional

from .custom_types import Array, PyTree, Scalar
from .path import AbstractPath
from .tree import tree_dataclass, tree_method


@tree_dataclass
class AbstractInterpolation(AbstractPath):
    ts: Array
    ys: PyTree


@tree_dataclass
class LinearInterpolation(AbstractInterpolation):
    @tree_method
    def derivative(self, t: Scalar) -> PyTree:
        ...

    @tree_method
    def evaluate(self, t0: Scalar, t1: Optional[Scalar] = None) -> PyTree:
        ...  # TODO. Think about point evaluations?
