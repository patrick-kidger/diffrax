from typing import Callable

import numpy as np

from ..custom_types import PyTree, Scalar
from ..local_interpolation import FourthOrderPolynomialInterpolation
from ..term import ODETerm
from .runge_kutta import AbstractESDIRK, ButcherTableau


a21 = 0.26
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

_kvaerno5_tableau = ButcherTableau(
    alpha=np.array(
        [0.52, 1.230333209967908, 0.8957659843500759, 0.43639360985864756, 1.0, 1.0]
    ),
    beta=(
        np.array([a21]),
        np.array([a31, a32]),
        np.array([a41, a42, a43]),
        np.array([a51, a52, a53, a54]),
        np.array([a61, a62, a63, a64, a65]),
        np.array([a71, a72, a73, a74, a75, a76]),
    ),
    c_sol=np.array([a71, a72, a73, a74, a75, a76, 0.26]),
    c_error=np.array(
        [a71 - a61, a72 - a62, a73 - a63, a74 - a64, a75 - a65, a76 - 0.26, 0.26]
    ),
    diagonal=np.array([0, 0.26, 0.26, 0.26, 0.26, 0.26, 0.26]),
)


class _Kvaerno5Interpolation(FourthOrderPolynomialInterpolation):
    # I don't think this is well-chosen -- I think this is just a simple choice to get
    # an approximation for y at the middle of each step, and that better choices are
    # probably available.
    c_mid = np.array([0, 0, 0, 0, 0, 0.5, 0])


class Kvaerno5(AbstractESDIRK):
    r"""Kvaerno's 5/4 method.

    A-L stable stiffly accurate 5th order ESDIRK method. Has an embedded 4th order
    method.

    When solving an ODE over the interval [t0, t1], note that this method will make
    some evaluations slightly past t1.

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
    tableau = _kvaerno5_tableau
    interpolation_cls = _Kvaerno5Interpolation
    order = 5


def kvaerno5(vector_field: Callable[[Scalar, PyTree, PyTree], PyTree], **kwargs):
    return Kvaerno5(term=ODETerm(vector_field=vector_field), **kwargs)
