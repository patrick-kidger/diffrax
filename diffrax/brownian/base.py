from ..custom_types import PyTree, Scalar
from ..path import AbstractPath


class AbstractBrownianPath(AbstractPath):
    def derivative(self, t: Scalar) -> PyTree:
        raise ValueError("Cannot take a derivative of a Brownian path.")
