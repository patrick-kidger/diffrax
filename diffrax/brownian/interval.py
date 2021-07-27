from ..custom_types import PyTree, Scalar
from ..tree import tree_dataclass, tree_method
from .base import AbstractBrownianPath


@tree_dataclass
class BrownianInterval(AbstractBrownianPath):
    @tree_method
    def evaluate(self, t0: Scalar, t1: Scalar) -> PyTree:
        ...  # TODO
