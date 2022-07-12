import numpy as np

from ..local_interpolation import ThirdOrderHermitePolynomialInterpolation
from .base import AbstractStratonovichSolver
from .runge_kutta import AbstractERK, ButcherTableau


_heun_tableau = ButcherTableau(
    a_lower=(np.array([1.0]),),
    b_sol=np.array([0.5, 0.5]),
    b_error=np.array([0.5, -0.5]),
    c=np.array([1.0]),
)


class Heun(AbstractERK, AbstractStratonovichSolver):
    """Heun's method.

    2nd order explicit Runge--Kutta method. Has an embedded Euler method for adaptive
    step sizing.

    Also sometimes known as either the "improved Euler method", "modified Euler method"
    or "explicit trapezoidal rule".

    Should not be confused with Heun's third order method, which is a different (higher
    order) method occasionally also just referred to as "Heun's method". (Which is
    available in Diffrax as [`diffrax.Bosh3`][].)

    When used to solve SDEs, converges to the Stratonovich solution.
    """

    tableau = _heun_tableau
    interpolation_cls = ThirdOrderHermitePolynomialInterpolation.from_k

    def order(self, terms):
        return 2

    def strong_order(self, terms):
        return 0.5
