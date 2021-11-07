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
# See /devdocs/predictor_dirk.md
θ = 1 / (2 * γ)
α21 = 1.0
α31 = 1.0 - θ
α32 = θ
α41 = a31
α42 = a32
α43 = γ

_kvaerno3_tableau = ButcherTableau(
    a_lower=(
        np.array([a21]),
        np.array([a31, a32]),
        np.array([a41, a42, a43]),
    ),
    a_predictor=(np.array([α21]), np.array([α31, α32]), np.array([α41, α42, α43])),
    a_diagonal=np.array([0, γ, γ, γ]),
    b_sol=np.array([a41, a42, a43, γ]),
    b_error=np.array([a41 - a31, a42 - a32, a43 - γ, γ]),
    c=np.array([2 * γ, 1.0, 1.0]),
)


class _Kvaerno3Interpolation(FourthOrderPolynomialInterpolation):
    # I don't think this is well-chosen -- I think this is just a simple choice to get
    # an approximation for y at the middle of each step, and that better choices are
    # probably available.
    c_mid = np.array([0, 0.5, 0, 0])


class Kvaerno3(AbstractESDIRK):
    r"""Kvaerno's 3/2 method.

    A-L stable stiffly accurate 3rd order ESDIRK method. Has an embedded 2nd order
    method. Uses 4 stages.

    ??? Reference

        ```bibtex
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
        ```
    """
    tableau = _kvaerno3_tableau
    interpolation_cls = _Kvaerno3Interpolation
    order = 3


def kvaerno3(vector_field: Callable[[Scalar, PyTree, PyTree], PyTree], **kwargs):
    return Kvaerno3(term=ODETerm(vector_field=vector_field), **kwargs)
