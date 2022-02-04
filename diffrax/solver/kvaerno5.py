import numpy as np

from ..local_interpolation import ThirdOrderHermitePolynomialInterpolation
from .runge_kutta import AbstractESDIRK, ButcherTableau


γ = 0.26
a21 = γ
a31 = 0.13
a32 = 0.84033320996790809
a41 = 0.22371961478320505
a42 = 0.47675532319799699
a43 = -0.06470895363112615
a51 = 0.16648564323248321
a52 = 0.10450018841591720
a53 = 0.03631482272098715
a54 = -0.13090704451073998
a61 = 0.13855640231268224
a62 = 0
a63 = -0.04245337201752043
a64 = 0.02446657898003141
a65 = 0.61943039072480676
a71 = 0.13659751177640291
a72 = 0
a73 = -0.05496908796538376
a74 = -0.04118626728321046
a75 = 0.62993304899016403
a76 = 0.06962479448202728

# Predictors taken from
# https://github.com/SciML/OrdinaryDiffEq.jl/blob/54fb35870fa402fc95d665cd5f9502e2759ea436/src/tableaus/sdirk_tableaus.jl#L1444  # noqa: E501
# https://github.com/SciML/OrdinaryDiffEq.jl/blob/54fb35870fa402fc95d665cd5f9502e2759ea436/src/perform_step/kencarp_kvaerno_perform_step.jl#L1123  # noqa: E501
# This is with the exception of α21, which is mistakenly set to zero.
#
# See also /devdocs/predictor_dirk.md
α21 = 1.0
α31 = -1.366025403784441
α32 = 2.3660254037844357
α41 = -0.19650552613122207
α42 = 0.8113579546496623
α43 = 0.38514757148155954
α51 = 0.10375304369958693
α52 = 0.937994698066431
α53 = -0.04174774176601781
α61 = -0.17281112873898072
α62 = 0.6235784481025847
α63 = 0.5492326806363959
α71 = a61
α72 = a62
α73 = a63
α74 = a64
α75 = a65
α76 = γ

_kvaerno5_tableau = ButcherTableau(
    a_lower=(
        np.array([a21]),
        np.array([a31, a32]),
        np.array([a41, a42, a43]),
        np.array([a51, a52, a53, a54]),
        np.array([a61, a62, a63, a64, a65]),
        np.array([a71, a72, a73, a74, a75, a76]),
    ),
    a_diagonal=np.array([0, γ, γ, γ, γ, γ, γ]),
    a_predictor=(
        np.array([α21]),
        np.array([α31, α32]),
        np.array([α41, α42, α43]),
        np.array([α51, α52, α53, 0]),
        np.array([α61, α62, α63, 0, 0]),
        np.array([α71, α72, α73, α74, α75, α76]),
    ),
    b_sol=np.array([a71, a72, a73, a74, a75, a76, γ]),
    b_error=np.array(
        [a71 - a61, a72 - a62, a73 - a63, a74 - a64, a75 - a65, a76 - γ, γ]
    ),
    c=np.array(
        [0.52, 1.230333209967908, 0.8957659843500759, 0.43639360985864756, 1.0, 1.0]
    ),
)


class Kvaerno5(AbstractESDIRK):
    r"""Kvaerno's 5/4 method.

    A-L stable stiffly accurate 5th order ESDIRK method. Has an embedded 4th order
    method for adaptive step sizing. Uses 7 stages.

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
    tableau = _kvaerno5_tableau
    interpolation_cls = ThirdOrderHermitePolynomialInterpolation.from_k

    def order(self, terms):
        return 5
