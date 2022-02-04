import numpy as np

from ..local_interpolation import ThirdOrderHermitePolynomialInterpolation
from .runge_kutta import AbstractESDIRK, ButcherTableau


# This γ notation is from the original paper. All the coefficients are described in
# terms of it.
γ = 0.43586652150
a21 = γ
a31 = (-4 * γ**2 + 6 * γ - 1) / (4 * γ)
a32 = (-2 * γ + 1) / (4 * γ)
a41 = (6 * γ - 1) / (12 * γ)
a42 = -1 / ((24 * γ - 12) * γ)
a43 = (-6 * γ**2 + 6 * γ - 1) / (6 * γ - 3)
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


class Kvaerno3(AbstractESDIRK):
    r"""Kvaerno's 3/2 method.

    A-L stable stiffly accurate 3rd order ESDIRK method. Has an embedded 2nd order
    method for adaptive step sizing. Uses 4 stages.

    ??? cite "Reference"

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
    interpolation_cls = ThirdOrderHermitePolynomialInterpolation.from_k

    def order(self, terms):
        return 3
