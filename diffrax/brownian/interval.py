from ..custom_types import PyTree, Scalar
from .base import AbstractBrownianPath


class BrownianInterval(AbstractBrownianPath):
    def evaluate(self, t0: Scalar, t1: Scalar) -> PyTree:
        ...  # TODO
