from collections.abc import Callable
from typing import ClassVar

import numpy as np

from .._local_interpolation import ThirdOrderHermitePolynomialInterpolation
from .runge_kutta import AbstractERK, ButcherTableau


_bosh3_tableau = ButcherTableau(
    a_lower=(
        np.array([1 / 2]),
        np.array([0.0, 3 / 4]),
        np.array([2 / 9, 1 / 3, 4 / 9]),
    ),
    b_sol=np.array([2 / 9, 1 / 3, 4 / 9, 0.0]),
    b_error=np.array([2 / 9 - 7 / 24, 1 / 3 - 1 / 4, 4 / 9 - 1 / 3, -1 / 8]),
    c=np.array([1 / 2, 3 / 4, 1.0]),
)


class Bosh3(AbstractERK):
    """Bogacki--Shampine's 3/2 method.

    3rd order explicit Runge--Kutta method. Has an embedded 2nd order method for
    adaptive step sizing. Uses 4 stages with FSAL. Uses 3rd order Hermite
    interpolation for dense/ts output.

    Also sometimes known as "Ralston's third order method".
    """

    tableau: ClassVar[ButcherTableau] = _bosh3_tableau
    interpolation_cls: ClassVar[
        Callable[..., ThirdOrderHermitePolynomialInterpolation]
    ] = ThirdOrderHermitePolynomialInterpolation.from_k

    def order(self, terms):
        return 3
