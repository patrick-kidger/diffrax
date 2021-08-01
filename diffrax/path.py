import abc
from typing import Optional

import equinox as eqx

from .custom_types import PyTree, Scalar


class AbstractPath(eqx.Module):
    def derivative(self, t: Scalar) -> PyTree:
        raise NotImplementedError(
            "Derivative does not exist for path of type {type(self)}."
        )

    @abc.abstractmethod
    def evaluate(self, t0: Scalar, t1: Optional[Scalar] = None) -> PyTree:
        pass
