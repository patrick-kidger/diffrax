import numpy as np

from ..local_interpolation import ThirdOrderHermitePolynomialInterpolation
from .runge_kutta import AbstractERK, ButcherTableau


_fehlberg2_tableau = ButcherTableau(
    a_lower=(np.array([1 / 2]), np.array([1 / 256, 255 / 256])),
    b_sol=np.array([1 / 512, 255 / 256, 1 / 512]),
    b_error=np.array([-1 / 512, 0, 1 / 512]),
    c=np.array([1 / 2, 1.0]),
)


class Fehlberg2(AbstractERK):
    """Fehlberg's method.

    2nd order explicit Runge--Kutta method. Has an embedded first order method for
    adaptive step sizing.
    """

    tableau = _fehlberg2_tableau
    interpolation_cls = ThirdOrderHermitePolynomialInterpolation.from_k

    def order(self, terms):
        return 2
