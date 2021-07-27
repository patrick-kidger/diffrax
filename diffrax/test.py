from tree import tree_dataclass, tree_method
from abc import ABCMeta, abstractmethod


@tree_dataclass
class X(metaclass=ABCMeta):
    @abstractmethod
    @tree_method
    def f(x):
        pass

    @abstractmethod
    @tree_method
    def f2(x):
        pass

    @tree_method
    def g(x):
        pass

    @tree_method
    def g2(x):
        pass


@tree_dataclass
class Y(X):
    a: int
    b: float

    @tree_method
    def f(x):
        pass

    @tree_method
    def f2(x):
        pass
