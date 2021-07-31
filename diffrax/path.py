import abc
from typing import Optional

from .custom_types import PyTree, Scalar
from .misc import ABCModule


class AbstractPath(ABCModule):
    def derivative(self, t: Scalar) -> PyTree:
        raise NotImplementedError("derivative has not been implemented")

    @abc.abstractmethod
    def evaluate(self, t0: Scalar, t1: Optional[Scalar] = None) -> PyTree:
        pass
