from collections.abc import Callable
from typing import cast, TYPE_CHECKING

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np


if TYPE_CHECKING:
    from typing import ClassVar as AbstractVar
else:
    from equinox import AbstractVar
import jax.flatten_util as fu
from equinox.internal import ω
from jaxtyping import Array, ArrayLike, PyTree, Shaped

from ._custom_types import RealScalarLike, Y
from ._misc import linear_rescale
from ._path import AbstractPath


ω = cast(Callable, ω)


class AbstractLocalInterpolation(AbstractPath):
    pass


class LocalLinearInterpolation(AbstractLocalInterpolation):
    t0: RealScalarLike
    t1: RealScalarLike
    y0: Y
    y1: Y

    def evaluate(
        self, t0: RealScalarLike, t1: RealScalarLike | None = None, left: bool = True
    ) -> PyTree[Array]:
        del left
        with jax.numpy_dtype_promotion("standard"):
            if t1 is None:
                coeff = linear_rescale(self.t0, t0, self.t1)
                return (
                    (self.y0**ω + coeff * (self.y1**ω - self.y0**ω)).call(jnp.asarray).ω
                )
            else:
                coeff = (t1 - t0) / (self.t1 - self.t0)
                return (coeff * (self.y1**ω - self.y0**ω)).call(jnp.asarray).ω


class ThirdOrderHermitePolynomialInterpolation(AbstractLocalInterpolation):
    t0: RealScalarLike
    t1: RealScalarLike
    coeffs: PyTree[Shaped[Array, "4 ?*dims"], "Y"]

    def __init__(
        self, *, t0: RealScalarLike, t1: RealScalarLike, y0: Y, y1: Y, k0: Y, k1: Y
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
        t0: RealScalarLike,
        t1: RealScalarLike,
        y0: PyTree[Shaped[ArrayLike, " ?*dims"], "Y"],
        y1: PyTree[Shaped[ArrayLike, " ?*dims"], "Y"],
        k: PyTree[Shaped[Array, "order ?*dims"], "Y"],
    ):
        return cls(t0=t0, t1=t1, y0=y0, y1=y1, k0=ω(k)[0].ω, k1=ω(k)[-1].ω)

    def evaluate(
        self, t0: RealScalarLike, t1: RealScalarLike | None = None, left: bool = True
    ) -> PyTree[Array]:
        del left
        if t1 is not None:
            return self.evaluate(t1) - self.evaluate(t0)

        t = linear_rescale(self.t0, t0, self.t1)

        def _eval(_coeffs):
            with jax.numpy_dtype_promotion("standard"):
                return jnp.polyval(_coeffs, t)

        return jtu.tree_map(_eval, self.coeffs)


class FourthOrderPolynomialInterpolation(AbstractLocalInterpolation):
    t0: RealScalarLike
    t1: RealScalarLike
    coeffs: PyTree[Shaped[Array, "5 ?*y"], "Y"]

    c_mid: AbstractVar[np.ndarray]

    def __init__(
        self,
        *,
        t0: RealScalarLike,
        t1: RealScalarLike,
        y0: Y,
        y1: Y,
        k: PyTree[Shaped[Array, "order ?*y"], "Y"],
    ):
        def _calculate(_y0, _y1, _k):
            with jax.numpy_dtype_promotion("standard"):
                _ymid = _y0 + jnp.tensordot(self.c_mid, _k, axes=1).astype(_y0.dtype)
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

    def evaluate(
        self, t0: RealScalarLike, t1: RealScalarLike | None = None, left: bool = True
    ) -> PyTree[Array]:
        del left
        if t1 is not None:
            return self.evaluate(t1) - self.evaluate(t0)

        t = linear_rescale(self.t0, t0, self.t1)

        def _eval(_coeffs):
            with jax.numpy_dtype_promotion("standard"):
                return jnp.polyval(_coeffs, t)

        return jtu.tree_map(_eval, self.coeffs)


class RodasInterpolation(AbstractLocalInterpolation):
    r"""Interpolation method for Rodas type solver.

    ??? cite "Reference"
       ```bibtex
       @book{book,
         author = {Hairer, Ernst and Wanner, Gerhard},
         year = {1996},
         month = {01},
         pages = {},
         title = {Solving Ordinary Differential Equations II. Stiff and
                  Differential-Algebraic Problems},
         volume = {14},
         journal = {Springer Verlag Series in Comput. Math.},
         doi = {10.1007/978-3-662-09947-6}
         }
        ```
    """

    coeff: AbstractVar[np.ndarray]

    stage_poly_coeffs: PyTree[Shaped[Array, "order stage"], "float"]
    t0: RealScalarLike
    t1: RealScalarLike
    y0: PyTree[Shaped[ArrayLike, " ?*dims"], "Y"]
    k: PyTree[Shaped[Array, "stage ?*dims"], "float"]

    def __init__(
        self,
        *,
        t0: RealScalarLike,
        t1: RealScalarLike,
        y0: PyTree[Shaped[ArrayLike, " ?*dims"], "Y"],
        k: PyTree[Shaped[Array, "stage ?*dims"], "float"],
    ):
        stage_poly_coeffs = []
        for i in range(len(self.coeff)):
            if i == len(self.coeff) - 1:
                stage_poly_coeffs.append(self.coeff[i])
                continue
            stage_poly_coeffs.append(self.coeff[i] - self.coeff[i + 1])

        self.stage_poly_coeffs = jnp.array(
            np.transpose(stage_poly_coeffs), dtype=jnp.float64
        )
        self.y0 = y0
        self.k = k
        self.t0 = t0
        self.t1 = t1

    def evaluate(
        self, t0: RealScalarLike, t1: RealScalarLike | None = None, left: bool = True
    ) -> PyTree[Array]:
        del left
        if t1 is not None:
            return self.evaluate(t1) - self.evaluate(t0)

        t = linear_rescale(self.t0, t0, self.t1)
        weighted_increment = jax.vmap(
            lambda coeff, stage_k: (t * jnp.polyval(jnp.flip(coeff), t)) * stage_k
        )(self.stage_poly_coeffs, self.k)

        y0, unravel = fu.ravel_pytree(self.y0)
        y1 = y0 + jnp.sum(weighted_increment, axis=0)
        return unravel(y1)

    @classmethod
    def from_k(
        cls,
        *,
        t0: RealScalarLike,
        t1: RealScalarLike,
        y0: PyTree[Shaped[ArrayLike, " ?*dims"], "Y"],
        k: PyTree[Shaped[Array, "stage ?*dims"], "float"],
    ):
        return cls(t0=t0, t1=t1, y0=y0, k=k)


RodasInterpolation.__init__.__doc__ = """**Arguments:**
Let `k` and `order` denote the stages and order of the solver.
- `coeff`: The coefficients of the Rodas interpolation. They represent the coefficients
    of b(τ). Should be a numpy array of shape `(order - 1, k)`.
"""
