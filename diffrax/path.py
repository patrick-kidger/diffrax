import abc
from typing import Optional

from .custom_types import PyTree, Scalar
from .tree import tree_dataclass


@tree_dataclass
class AbstractPath(metaclass=abc.ABCMeta):
    def derivative(self, t: Scalar) -> PyTree:
        raise NotImplementedError("derivative has not been implemented")

    @abc.abstractmethod
    def evaluate(self, t0: Scalar, t1: Optional[Scalar] = None) -> PyTree:
        pass
