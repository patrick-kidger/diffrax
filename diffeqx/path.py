import abc

from .custom_types import PyTree, Scalar


class AbstractPath(metaclass=abc.ABCMeta):
    def derivative(self, t: Scalar) -> PyTree:
        raise NotImplementedError("derivative has not been implemented")

    @abc.abstractmethod
    def evaluate(self, t0: Scalar, t1: Scalar) -> PyTree:
        pass

