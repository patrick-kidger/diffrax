from ..custom_types import PyTree, Scalar
from ..jax_tricks import tree_dataclass
from .base import AbstractBrownianPath


@tree_dataclass
class BrownianInterval(AbstractBrownianPath):
    def evaluate(self, t0: Scalar, t1: Scalar) -> PyTree:
        ...  # TODO
