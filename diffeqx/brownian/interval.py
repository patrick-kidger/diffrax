from .base import AbstractBrownianPath
from ..custom_types import PyTree, Scalar


class BrownianInterval(AbstractBrownianPath):
    def evaluate(self, t0: Scalar, t1: Scalar) -> PyTree
        ... # TODO

