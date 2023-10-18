import abc
from typing import Optional

import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np
from equinox.internal import ω
from jaxtyping import Array, PyTree, Shaped

from ._custom_types import RealScalarLike
from ._misc import linear_rescale
from ._path import AbstractPath


class AbstractLocalInterpolation(AbstractPath):
    pass


class LocalLinearInterpolation(AbstractLocalInterpolation):
    t0: RealScalarLike
    t1: RealScalarLike
    y0: PyTree[Array]
    y1: PyTree[Array]

    def evaluate(
        self, t0: RealScalarLike, t1: Optional[RealScalarLike] = None, left: bool = True
    ) -> PyTree[Array]:
        del left
        if t1 is None:
            coeff = linear_rescale(self.t0, t0, self.t1)
            return (self.y0**ω + coeff * (self.y1**ω - self.y0**ω)).ω
        else:
            coeff = (t1 - t0) / (self.t1 - self.t0)
            return (coeff * (self.y1**ω - self.y0**ω)).ω


class ThirdOrderHermitePolynomialInterpolation(AbstractLocalInterpolation):
    t0: RealScalarLike
    t1: RealScalarLike
    coeffs: PyTree[Shaped[Array, " 4 *dims"]]

    def __init__(
        self,
        *,
        t0: RealScalarLike,
        t1: RealScalarLike,
        y0: PyTree[Shaped[Array, " *dims"]],
        y1: PyTree[Shaped[Array, " *dims"]],
        k0: PyTree[Shaped[Array, " *dims"]],
        k1: PyTree[Shaped[Array, " *dims"]],
    ):
        def _calculate(_y0, _y1, _k0, _k1):
            _a = _k0 + _k1 + 2 * _y0 - 2 * _y1
            _b = -2 * _k0 - _k1 - 3 * _y0 + 3 * _y1
            return jnp.stack([_a, _b, _k0, _y0])

        self.t0 = t0
        self.t1 = t1
        self.coeffs = jtu.tree_map(_calculate, y0, y1, k0, k1)

    @classmethod
    def from_k(
        cls,
        *,
        y0: PyTree[Shaped[Array, " *dims"]],
        y1: PyTree[Shaped[Array, " *dims"]],
        k: PyTree[Shaped[Array, " order *dims"]],
        **kwargs,
    ):
        return cls(y0=y0, y1=y1, k0=ω(k)[0].ω, k1=ω(k)[-1].ω, **kwargs)

    def evaluate(
        self, t0: RealScalarLike, t1: Optional[RealScalarLike] = None, left: bool = True
    ) -> PyTree[Array]:
        del left
        if t1 is not None:
            return self.evaluate(t1) - self.evaluate(t0)

        t = linear_rescale(self.t0, t0, self.t1)

        def _eval(_coeffs):
            return jnp.polyval(_coeffs, t)

        return jtu.tree_map(_eval, self.coeffs)


class FourthOrderPolynomialInterpolation(AbstractLocalInterpolation):
    t0: RealScalarLike
    t1: RealScalarLike
    coeffs: PyTree[Shaped[Array, " 5 *dims"]]

    def __init__(
        self,
        *,
        t0: RealScalarLike,
        t1: RealScalarLike,
        y0: PyTree[Shaped[Array, " *dims"]],
        y1: PyTree[Shaped[Array, " *dims"]],
        k: PyTree[Shaped[Array, " order *dims"]],
    ):
        def _calculate(_y0, _y1, _k):
            _ymid = _y0 + jnp.tensordot(self.c_mid, _k, axes=1)
            _f0 = _k[0]
            _f1 = _k[-1]
            # TODO: rewrite as matrix-vector product?
            _a = 2 * (_f1 - _f0) - 8 * (_y1 + _y0) + 16 * _ymid
            _b = 5 * _f0 - 3 * _f1 + 18 * _y0 + 14 * _y1 - 32 * _ymid
            _c = _f1 - 4 * _f0 - 11 * _y0 - 5 * _y1 + 16 * _ymid
            return jnp.stack([_a, _b, _c, _f0, _y0])

        self.t0 = t0
        self.t1 = t1
        self.coeffs = jtu.tree_map(_calculate, y0, y1, k)

    @property
    @abc.abstractmethod
    def c_mid(self) -> np.ndarray:
        pass

    def evaluate(
        self, t0: RealScalarLike, t1: Optional[RealScalarLike] = None, left: bool = True
    ) -> PyTree[Array]:
        del left
        if t1 is not None:
            return self.evaluate(t1) - self.evaluate(t0)

        t = linear_rescale(self.t0, t0, self.t1)

        def _eval(_coeffs):
            return jnp.polyval(_coeffs, t)

        return jtu.tree_map(_eval, self.coeffs)
