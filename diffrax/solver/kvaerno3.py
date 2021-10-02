from typing import Callable

import numpy as np

from ..custom_types import PyTree, Scalar
from ..local_interpolation import FourthOrderPolynomialInterpolation
from ..term import ODETerm
from .runge_kutta import AbstractESDIRK, ButcherTableau


# This γ notation is from the original paper. All the coefficients are described in
# terms of it.
γ = 0.43586652150
a21 = γ
a31 = (-4 * γ ** 2 + 6 * γ - 1) / (4 * γ)
a32 = (-2 * γ + 1) / (4 * γ)
a41 = (6 * γ - 1) / (12 * γ)
a42 = -1 / ((24 * γ - 12) * γ)
a43 = (-6 * γ ** 2 + 6 * γ - 1) / (6 * γ - 3)

_kvaerno3_tableau = ButcherTableau(
    alpha=np.array([2 * γ, 1.0, 1.0]),
    beta=(
        np.array([a21]),
        np.array([a31, a32]),
        np.array([a41, a42, a43]),
    ),
    c_sol=np.array([a41, a42, a43, γ]),
    c_error=np.array([0.0, 0.0, -1.0, 1.0]),
    diagonal=np.array([0, γ, γ, γ]),
)


class _Kvaerno3Interpolation(FourthOrderPolynomialInterpolation):
    # I don't think this is well-chosen -- I think this is just a simple choice to get
    # an approximation for y at the middle of each step, and that better choices are
    # probably available.
    c_mid = np.array([0, 0.5, 0, 0])


class Kvaerno3(AbstractESDIRK):
    tableau = _kvaerno3_tableau
    interpolation_cls = _Kvaerno3Interpolation
    order = 3


def kvaerno3(vector_field: Callable[[Scalar, PyTree, PyTree], PyTree], **kwargs):
    return Kvaerno3(term=ODETerm(vector_field=vector_field), **kwargs)
