import numpy as np

from ..local_interpolation import ThirdOrderHermitePolynomialInterpolation
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
    adaptive step sizing.

    Also sometimes known as "Heun's third order method". (Not to be confused with
    [`diffrax.Heun`][], which is a second order method).
    """

    tableau = _bosh3_tableau
    interpolation_cls = ThirdOrderHermitePolynomialInterpolation.from_k

    def order(self, terms):
        return 3
