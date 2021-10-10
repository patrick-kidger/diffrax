import abc
from dataclasses import field
from typing import Optional, Tuple

import jax.numpy as jnp
import numpy as np

from .custom_types import Array, Scalar
from .misc import linear_rescale
from .path import AbstractPath


class AbstractLocalInterpolation(AbstractPath):
    t0: Scalar = field(init=True)
    t1: Scalar = field(init=True)  # override init=False on AbstractPath


class LocalLinearInterpolation(AbstractLocalInterpolation):
    y0: Array["state"]  # noqa: F821
    y1: Array["state"]  # noqa: F821

    def evaluate(
        self, t0: Scalar, t1: Optional[Scalar] = None
    ) -> Array["state"]:  # noqa: F821
        if t1 is None:
            coeff = linear_rescale(self.t0, t0, self.t1)
            return self.y0 + coeff * (self.y1 - self.y0)
        else:
            return ((t1 - t0) / (self.t1 - self.t0)) * (self.y1 - self.y0)

    def derivative(self, t: Scalar) -> Array["state"]:  # noqa: F821
        return (self.y1 - self.y0) / (self.t1 - self.t0)


class FourthOrderPolynomialInterpolation(AbstractLocalInterpolation):
    coeffs: Tuple[
        Array["state"],  # noqa: F821
        Array["state"],  # noqa: F821
        Array["state"],  # noqa: F821
        Array["state"],  # noqa: F821
        Array["state"],  # noqa: F821
    ]

    def __init__(
        self,
        *,
        y0: Array["state"],  # noqa: F821
        y1: Array["state"],  # noqa: F821
        k: Array["order", "state"],  # noqa: F821
        **kwargs
    ):
        super().__init__(**kwargs)
        ymid = y0 + self.c_mid @ k
        f0 = k[0]
        f1 = k[-1]
        # TODO: rewrite as matrix-vector product?
        a = 2 * (f1 - f0) - 8 * (y1 + y0) + 16 * ymid
        b = 5 * f0 - 3 * f1 + 18 * y0 + 14 * y1 - 32 * ymid
        c = f1 - 4 * f0 - 11 * y0 - 5 * y1 + 16 * ymid
        self.coeffs = jnp.stack([a, b, c, f0, y0])

    @property
    @abc.abstractmethod
    def c_mid(self) -> np.ndarray:
        pass

    def evaluate(
        self, t0: Scalar, t1: Optional[Scalar] = None
    ) -> Array["state"]:  # noqa: F821
        if t1 is not None:
            return self.evaluate(t1) - self.evaluate(t0)
        return jnp.polyval(self.coeffs, linear_rescale(self.t0, t0, self.t1))

    def derivative(self, t: Scalar) -> Array["state"]:  # noqa: F821
        a, b, c, d, _ = self.coeffs
        t = linear_rescale(self.t0, t, self.t1)
        return jnp.polyval([4 * a, 3 * b, 2 * c, d], t) / (self.t1 - self.t0)
