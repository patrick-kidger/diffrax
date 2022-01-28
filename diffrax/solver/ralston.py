import numpy as np

from ..local_interpolation import ThirdOrderHermitePolynomialInterpolation
from .runge_kutta import AbstractERK, ButcherTableau


_ralston_tableau = ButcherTableau(
    a_lower=(np.array([0.75]),),
    b_sol=np.array([1 / 3, 2 / 3]),
    b_error=np.array([2 / 3, -2 / 3]),
    c=np.array([0.75]),
)


class Ralston(AbstractERK):
    """Ralston's method.

    2nd order explicit Runge--Kutta method. Has an embedded Euler method for adaptive
    step sizing.
    """

    tableau = _ralston_tableau
    interpolation_cls = ThirdOrderHermitePolynomialInterpolation.from_k
    order = 2
