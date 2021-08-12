import abc
from dataclasses import field
from typing import Optional

import equinox as eqx

from .custom_types import PyTree, Scalar


class AbstractPath(eqx.Module):
    t0: Scalar = field(init=False)
    t1: Scalar = field(init=False)

    def derivative(self, t: Scalar) -> PyTree:
        raise NotImplementedError(
            f"Derivative does not exist for path of type {type(self)}."
        )

    @abc.abstractmethod
    def evaluate(self, t0: Scalar, t1: Optional[Scalar] = None) -> PyTree:
        pass
