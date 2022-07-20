import numpy as np

from ..local_interpolation import ThirdOrderHermitePolynomialInterpolation
from .base import AbstractStratonovichSolver
from .runge_kutta import AbstractERK, ButcherTableau


_midpoint_tableau = ButcherTableau(
    a_lower=(np.array([0.5]),),
    b_sol=np.array([0.0, 1.0]),
    b_error=np.array([1.0, -1.0]),
    c=np.array([0.5]),
)


class Midpoint(AbstractERK, AbstractStratonovichSolver):
    """Midpoint method.

    2nd order explicit Runge--Kutta method. Has an embedded Euler method for adaptive
    step sizing.

    Also sometimes known as the "modified Euler method".

    When used to solve SDEs, converges to the Stratonovich solution.
    """

    tableau = _midpoint_tableau
    interpolation_cls = ThirdOrderHermitePolynomialInterpolation.from_k

    def order(self, terms):
        return 2

    def strong_order(self, terms):
        return 0.5
