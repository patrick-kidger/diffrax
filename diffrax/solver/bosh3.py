from typing import Callable

import numpy as np

from ..custom_types import PyTree, Scalar
from ..local_interpolation import FourthOrderPolynomialInterpolation
from ..term import ODETerm
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


class _Bosh3Interpolation(FourthOrderPolynomialInterpolation):
    # I don't think this is well-chosen -- I think this is just a simple choice to get
    # an approximation for y at the middle of each step, and that better choices are
    # probably available.
    c_mid = np.array([0, 0.5, 0, 0])


class Bosh3(AbstractERK):
    """Bogacki--Shampine's 3/2 method.

    3rd order explicit Runge--Kutta method. Has an embedded 2nd order method.
    """

    tableau = _bosh3_tableau
    interpolation_cls = _Bosh3Interpolation
    order = 3


def bosh3(vector_field: Callable[[Scalar, PyTree, PyTree], PyTree], **kwargs) -> Bosh3:
    return Bosh3(term=ODETerm(vector_field=vector_field), **kwargs)
