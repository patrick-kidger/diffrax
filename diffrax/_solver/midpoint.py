from collections.abc import Callable
from typing import ClassVar

import equinox.internal as eqxi
import numpy as np

from .._local_interpolation import ThirdOrderHermitePolynomialInterpolation
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
    step sizing. Uses 2 stages. Uses 2nd order Hermite interpolation for dense/ts
    output.

    Also sometimes known as the "modified Euler method".

    When used to solve SDEs, converges to the Stratonovich solution.
    """

    tableau: ClassVar[ButcherTableau] = _midpoint_tableau
    interpolation_cls: ClassVar[
        Callable[..., ThirdOrderHermitePolynomialInterpolation]
    ] = ThirdOrderHermitePolynomialInterpolation.from_k

    def order(self, terms):
        del terms
        return 2

    def strong_order(self, terms):
        del terms
        return 0.5


eqxi.doc_remove_args("scan_kind")(Midpoint.__init__)
Midpoint.__init__.__doc__ = """**Arguments:** None"""
