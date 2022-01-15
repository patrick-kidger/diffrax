import numpy as np

from ..local_interpolation import FourthOrderPolynomialInterpolation
from .base import AbstractStratonovichSolver
from .runge_kutta import AbstractERK, ButcherTableau


_heun_tableau = ButcherTableau(
    a_lower=(np.array([1.0]),),
    b_sol=np.array([0.5, 0.5]),
    b_error=np.array([0.5, -0.5]),
    c=np.array([1.0]),
)


class _HeunInterpolation(FourthOrderPolynomialInterpolation):
    # I don't think this is well-chosen -- I think this is just a simple choice to get
    # an approximation for y at the middle of each step, and that better choices are
    # probably available.
    c_mid = np.array([0, 0.5])


class Heun(AbstractERK, AbstractStratonovichSolver):
    """Heun's method.

    2nd order explicit Runge--Kutta method. Has an embedded Euler method for adaptive
    step sizing.

    Also sometimes known as either the "improved Euler method", "modified Euler method"
    or "explicit trapezoidal rule".

    Should not be confused with Heun's third order method, which is a different (higher
    order) method occasionally also just referred to as "Heun's method".
    """

    tableau = _heun_tableau
    interpolation_cls = _HeunInterpolation
    order = 2
