import abc
from typing import Optional

from .custom_types import PyTree, Scalar
from .tree import tree_dataclass, tree_method


@tree_dataclass
class AbstractPath(metaclass=abc.ABCMeta):
    @tree_method
    def derivative(self, t: Scalar) -> PyTree:
        raise NotImplementedError("derivative has not been implemented")

    @abc.abstractmethod
    @tree_method
    def evaluate(self, t0: Scalar, t1: Optional[Scalar] = None) -> PyTree:
        pass
