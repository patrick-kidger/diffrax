import abc

from .custom_types import Array, Scalar, PyTree
from .path import AbstractPath


class AbstractInterpolation(AbstractPath):
    def __init__(self, *, ts: Array, ys: PyTree, **kwargs):
        super().__init__(**kwargs)
        self.ts = ts
        self.ys = ys


class LinearInterpolation(AbstractInterpolation):
    def derivative(self, t: Scalar) -> PyTree:
        ...

    def evaluate(self, t0: Scalar, t1: Scalar) -> PyTree:
        ...  # TODO. Think about point evaluations?
