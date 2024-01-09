from collections.abc import Callable
from typing import ClassVar

import numpy as np
import optimistix as optx

from .._local_interpolation import ThirdOrderHermitePolynomialInterpolation
from .._root_finder import VeryChord, with_stepsize_controller_tols
from .runge_kutta import AbstractESDIRK, ButcherTableau


# This γ notation is from the original paper. All the coefficients are described in
# terms of it.
#
# In passing: DifferentialEquations.jl actually gets this wrong. In her paper Kvaerno
# describes two different 4/3 methods. Both of them use the same 4-3 error estimate;
# the difference is that one is a "normal" 4/3 method, advancing the solution according
# to the 4th-order final stage, whilst the other is a "3/4" method, advancing the
# solution using the 3rd-order penultimate stage. Each approach mandates a different
# values of γ (but the same formulae for a21 etc.)
# DifferentialEquations.jl muddles these two up: it uses the "3/4" value for γ whilst
# advancing the solution according to the final stage.
γ = 0.5728160625


def poly(*args):
    return np.polyval(args, γ)


a21 = γ
a31 = poly(144, -180, 81, -15, 1) * γ / poly(12, -6, 1) ** 2
a32 = poly(-36, 39, -15, 2) * γ / poly(12, -6, 1) ** 2
a41 = poly(-144, 396, -330, 117, -18, 1) / (12 * γ**2 * poly(12, -9, 2))
a42 = poly(72, -126, 69, -15, 1) / (12 * γ**2 * poly(3, -1))
a43 = (poly(-6, 6, -1) * poly(12, -6, 1) ** 2) / (
    12 * γ**2 * poly(12, -9, 2) * poly(3, -1)
)
a51 = poly(288, -312, 120, -18, 1) / (48 * γ**2 * poly(12, -9, 2))
a52 = poly(24, -12, 1) / (48 * γ**2 * poly(3, -1))
a53 = -(poly(12, -6, 1) ** 3) / (
    48 * γ**2 * poly(3, -1) * poly(12, -9, 2) * poly(6, -6, 1)
)
a54 = poly(-24, 36, -12, 1) / poly(24, -24, 4)
c2 = γ + a21
c3 = γ + a31 + a32
c4 = 1.0
c5 = 1.0
# See /devdocs/predictor_dirk.md
θ1 = c3 / c2
θ2 = (c4 - c2) / (c3 - c2)
α21 = 1.0
α31 = 1 - θ1
α32 = θ1
α41 = 0
α42 = 1 - θ2
α43 = θ2
α51 = a41
α52 = a42
α53 = a43
α54 = γ

_kvaerno4_tableau = ButcherTableau(
    a_lower=(
        np.array([a21]),
        np.array([a31, a32]),
        np.array([a41, a42, a43]),
        np.array([a51, a52, a53, a54]),
    ),
    a_predictor=(
        np.array([α21]),
        np.array([α31, α32]),
        np.array([α41, α42, α43]),
        np.array([α51, α52, α53, α54]),
    ),
    a_diagonal=np.array([0, γ, γ, γ, γ]),
    b_sol=np.array([a51, a52, a53, a54, γ]),
    b_error=np.array([a51 - a41, a52 - a42, a53 - a43, a54 - γ, γ]),
    c=np.array([c2, c3, c4, c5]),
)


class Kvaerno4(AbstractESDIRK):
    r"""Kvaerno's 4/3 method.

    A-L stable stiffly accurate 4th order ESDIRK method. Has an embedded 3rd order
    method for adaptive step sizing. Uses 5 stages with FSAL. Uses 3rd order Hermite
    interpolation for dense/ts output.

    When solving an ODE over the interval $[t_0, t_1]$, note that this method will make
    some evaluations slightly past $t_1$.

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

    tableau: ClassVar[ButcherTableau] = _kvaerno4_tableau
    interpolation_cls: ClassVar[
        Callable[..., ThirdOrderHermitePolynomialInterpolation]
    ] = ThirdOrderHermitePolynomialInterpolation.from_k

    root_finder: optx.AbstractRootFinder = with_stepsize_controller_tols(VeryChord)()
    root_find_max_steps: int = 10

    def order(self, terms):
        return 4
