from collections.abc import Callable
from typing import ClassVar

import numpy as np

from .._local_interpolation import ThirdOrderHermitePolynomialInterpolation
from .base import AbstractStratonovichSolver
from .runge_kutta import AbstractERK, ButcherTableau


#
# Note that a lot of implementations actually get Ralston's method wrong.
# Ralston's method is the 2/3-method, not the 3/4-method.
# At some point it looks like someone wrote down the wrong thing on Wikipedia, and
# everyone just blindly copied it without realising.
# (Wikipedia has now been fixed.)
# Do the Taylor expansions yourself if you don't believe me!
# Credit to James Foster for pointing this one out to me.
#
_ralston_tableau = ButcherTableau(
    a_lower=(np.array([2 / 3]),),
    b_sol=np.array([0.25, 0.75]),
    b_error=np.array([0.75, -0.75]),
    c=np.array([2 / 3]),
)


class Ralston(AbstractERK, AbstractStratonovichSolver):
    """Ralston's method.

    2nd order explicit Runge--Kutta method. Has an embedded Euler method for adaptive
    step sizing. Uses 2 stages. Uses 2nd order Hermite interpolation for dense output.

    When used to solve SDEs, converges to the Stratonovich solution.
    """

    tableau: ClassVar[ButcherTableau] = _ralston_tableau
    interpolation_cls: ClassVar[
        Callable[..., ThirdOrderHermitePolynomialInterpolation]
    ] = ThirdOrderHermitePolynomialInterpolation.from_k

    def order(self, terms):
        return 2

    def strong_order(self, terms):
        return 0.5
