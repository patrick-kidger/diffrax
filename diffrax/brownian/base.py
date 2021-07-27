from ..custom_types import PyTree, Scalar
from ..path import AbstractPath
from ..tree import tree_dataclass, tree_method


@tree_dataclass
class AbstractBrownianPath(AbstractPath):
    @tree_method
    def derivative(self, t: Scalar) -> PyTree:
        raise ValueError("Cannot take a derivative of a Brownian path.")
