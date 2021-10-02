from typing import Callable

import numpy as np

from ..custom_types import PyTree, Scalar
from ..local_interpolation import FourthOrderPolynomialInterpolation
from ..misc import copy_docstring_from
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
    c_error=np.array([a41 - a31, a42 - a32, a43 - γ, γ]),
    diagonal=np.array([0, γ, γ, γ]),
)


class _Kvaerno3Interpolation(FourthOrderPolynomialInterpolation):
    # I don't think this is well-chosen -- I think this is just a simple choice to get
    # an approximation for y at the middle of each step, and that better choices are
    # probably available.
    c_mid = np.array([0, 0.5, 0, 0])


class Kvaerno3(AbstractESDIRK):
    r"""Kvaerno's 3/2 method.

    A-L stable stiffly accurate 3rd order ESDIRK method. Has an embedded 2nd order
    method.

    @article{kvaerno2004singly,
      title={Singly diagonally implicit Runge--Kutta methods with an explicit first
             stage},
      author={Kv{\ae}rn{\o}, Anne},
      journal={BIT Numerical Mathematics},
      volume={44},
      number={3},
      pages={489--502},
      year={2004},
      publisher={Springer}
    }
    """
    tableau = _kvaerno3_tableau
    interpolation_cls = _Kvaerno3Interpolation
    order = 3


@copy_docstring_from(Kvaerno3)
def kvaerno3(vector_field: Callable[[Scalar, PyTree, PyTree], PyTree], **kwargs):
    return Kvaerno3(term=ODETerm(vector_field=vector_field), **kwargs)
