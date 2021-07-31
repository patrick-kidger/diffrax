from ..custom_types import PyTree, Scalar
from ..jax_tricks import tree_dataclass
from ..path import AbstractPath


@tree_dataclass
class AbstractBrownianPath(AbstractPath):
    def derivative(self, t: Scalar) -> PyTree:
        raise ValueError("Cannot take a derivative of a Brownian path.")
