from typing import Callable

import numpy as np

from ..custom_types import PyTree, Scalar
from ..local_interpolation import FourthOrderPolynomialInterpolation
from ..term import ODETerm
from .runge_kutta import AbstractERK, ButcherTableau


_fehlberg2_tableau = ButcherTableau(
    a_lower=(np.array([1 / 2]), np.array([1 / 256, 255 / 256])),
    b_sol=np.array([1 / 512, 255 / 256, 1 / 512]),
    b_error=np.array([-1 / 512, 0, 1 / 512]),
    c=np.array([1 / 2, 1.0]),
)


class _Fehlberg2Interpolation(FourthOrderPolynomialInterpolation):
    # I don't think this is well-chosen -- I think this is just a simple choice to get
    # an approximation for y at the middle of each step, and that better choices are
    # probably available.
    c_mid = np.array([0, 0.5, 0])


class Fehlberg2(AbstractERK):
    tableau = _fehlberg2_tableau
    interpolation_cls = _Fehlberg2Interpolation
    order = 2


def fehlberg2(vector_field: Callable[[Scalar, PyTree, PyTree], PyTree], **kwargs):
    return Fehlberg2(term=ODETerm(vector_field=vector_field), **kwargs)
