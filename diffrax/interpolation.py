import jax.numpy as jnp
from typing import Optional

from .custom_types import Array, PyTree, Scalar
from .path import AbstractPath
from .tree import tree_dataclass


@tree_dataclass
class AbstractInterpolation(AbstractPath):
    ts: Array
    ys: PyTree

    requested_state = frozenset()

    def _interpret_t(self, t: Array):
        jnp.searchsorted(self.ts, t)
        ...


@tree_dataclass
class LinearInterpolation(AbstractInterpolation):
    def derivative(self, t: Scalar) -> PyTree:
        ...

    def evaluate(self, t0: Scalar, t1: Optional[Scalar] = None) -> PyTree:
        if t1 is not None:
            return self.evaluate(t1) - self.evaluate(t0)
        ...


@tree_dataclass
class FourthOrderPolynomialInterpolation(AbstractInterpolation):
    requested_state = frozenset({"k"})

    def derivative(self, t: Scalar) -> PyTree:
        ...

    def evaluate(self, t0: Scalar, t1: Optional[Scalar] = None) -> PyTree:
        if t1 is not None:
            return self.evaluate(t1) - self.evaluate(t0)
        ...
