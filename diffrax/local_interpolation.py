import abc
from dataclasses import field
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np

from .custom_types import Array, PyTree, Scalar
from .misc import linear_rescale, ω
from .path import AbstractPath


class AbstractLocalInterpolation(AbstractPath):
    t0: Scalar = field(init=True)
    t1: Scalar = field(init=True)  # override init=False on AbstractPath


class LocalLinearInterpolation(AbstractLocalInterpolation):
    y0: PyTree
    y1: PyTree

    def evaluate(
        self, t0: Scalar, t1: Optional[Scalar] = None, left: bool = True
    ) -> PyTree:
        del left
        if t1 is None:
            coeff = linear_rescale(self.t0, t0, self.t1)
            return (self.y0**ω + coeff * (self.y1**ω - self.y0**ω)).ω
        else:
            coeff = (t1 - t0) / (self.t1 - self.t0)
            return (coeff * (self.y1**ω - self.y0**ω)).ω


class ThirdOrderHermitePolynomialInterpolation(AbstractLocalInterpolation):
    coeffs: PyTree[Array[4, "dims":...]]  # noqa: F821

    def __init__(
        self,
        *,
        y0: PyTree[Array["dims":...]],  # noqa: F821
        y1: PyTree[Array["dims":...]],  # noqa: F821
        f0: PyTree[Array["dims":...]],  # noqa: F821
        f1: PyTree[Array["dims":...]],  # noqa: F821
        **kwargs
    ):
        super().__init__(**kwargs)

        def _calculate(_y0, _y1, _f0, _f1):
            _a = _f0 + _f1 + 2 * _y0 - 2 * _y1
            _b = -2 * _f0 - _f1 - 3 * _y0 + 3 * _y1
            return jnp.stack([_a, _b, _f0, _y0])

        self.coeffs = jax.tree_map(_calculate, y0, y1, f0, f1)

    @classmethod
    def from_k(
        cls,
        *,
        y0: PyTree[Array["dims":...]],  # noqa: F821
        y1: PyTree[Array["dims":...]],  # noqa: F821
        k: PyTree[Array["order", "dims":...]],  # noqa: F821
        **kwargs
    ):
        return cls(y0=y0, y1=y1, f0=ω(k)[0].ω, f1=ω(k)[-1].ω, **kwargs)

    def evaluate(
        self, t0: Scalar, t1: Optional[Scalar] = None, left: bool = True
    ) -> PyTree:
        del left
        if t1 is not None:
            return self.evaluate(t1) - self.evaluate(t0)

        t = linear_rescale(self.t0, t0, self.t1)

        def _eval(_coeffs):
            return jnp.polyval(_coeffs, t)

        return jax.tree_map(_eval, self.coeffs)


class FourthOrderPolynomialInterpolation(AbstractLocalInterpolation):
    coeffs: PyTree[Array[5, "dims":...]]  # noqa: F821

    def __init__(
        self,
        *,
        y0: PyTree[Array["dims":...]],  # noqa: F821
        y1: PyTree[Array["dims":...]],  # noqa: F821
        k: PyTree[Array["order", "dims":...]],  # noqa: F821
        **kwargs
    ):
        super().__init__(**kwargs)

        def _calculate(_y0, _y1, _k):
            _ymid = _y0 + jnp.tensordot(self.c_mid, _k, axes=1)
            _f0 = _k[0]
            _f1 = _k[-1]
            # TODO: rewrite as matrix-vector product?
            _a = 2 * (_f1 - _f0) - 8 * (_y1 + _y0) + 16 * _ymid
            _b = 5 * _f0 - 3 * _f1 + 18 * _y0 + 14 * _y1 - 32 * _ymid
            _c = _f1 - 4 * _f0 - 11 * _y0 - 5 * _y1 + 16 * _ymid
            return jnp.stack([_a, _b, _c, _f0, _y0])

        self.coeffs = jax.tree_map(_calculate, y0, y1, k)

    @property
    @abc.abstractmethod
    def c_mid(self) -> np.ndarray:
        pass

    def evaluate(
        self, t0: Scalar, t1: Optional[Scalar] = None, left: bool = True
    ) -> PyTree:
        del left
        if t1 is not None:
            return self.evaluate(t1) - self.evaluate(t0)

        t = linear_rescale(self.t0, t0, self.t1)

        def _eval(_coeffs):
            return jnp.polyval(_coeffs, t)

        return jax.tree_map(_eval, self.coeffs)
