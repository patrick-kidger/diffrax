from ..custom_types import PyTree, Scalar
from ..path import AbstractPath
from ..tree import tree_dataclass


@tree_dataclass
class AbstractBrownianPath(AbstractPath):
    def derivative(self, t: Scalar) -> PyTree:
        raise ValueError("Cannot take a derivative of a Brownian path.")
