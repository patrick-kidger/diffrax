import numpy as np

from ..local_interpolation import FourthOrderPolynomialInterpolation
from .runge_kutta import AbstractERK, ButcherTableau


_ralston_tableau = ButcherTableau(
    a_lower=(np.array([0.75]),),
    b_sol=np.array([1 / 3, 2 / 3]),
    b_error=np.array([2 / 3, -2 / 3]),
    c=np.array([0.75]),
)


class _RalstonInterpolation(FourthOrderPolynomialInterpolation):
    # I don't think this is well-chosen -- I think this is just a simple choice to get
    # an approximation for y at the middle of each step, and that better choices are
    # probably available.
    c_mid = np.array([0, 0.5])


class Ralston(AbstractERK):
    """Ralston's method.

    2nd order explicit Runge--Kutta method. Has an embedded Euler method for adaptive
    step sizing.
    """

    tableau = _ralston_tableau
    interpolation_cls = _RalstonInterpolation
    order = 2
