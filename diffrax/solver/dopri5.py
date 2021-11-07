from typing import Callable

import numpy as np

from ..custom_types import PyTree, Scalar
from ..local_interpolation import FourthOrderPolynomialInterpolation
from ..term import ODETerm
from .runge_kutta import AbstractERK, ButcherTableau


_dopri5_tableau = ButcherTableau(
    a_lower=(
        np.array([1 / 5]),
        np.array([3 / 40, 9 / 40]),
        np.array([44 / 45, -56 / 15, 32 / 9]),
        np.array([19372 / 6561, -25360 / 2187, 64448 / 6561, -212 / 729]),
        np.array([9017 / 3168, -355 / 33, 46732 / 5247, 49 / 176, -5103 / 18656]),
        np.array([35 / 384, 0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84]),
    ),
    b_sol=np.array([35 / 384, 0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84, 0]),
    b_error=np.array(
        [
            35 / 384 - 1951 / 21600,
            0,
            500 / 1113 - 22642 / 50085,
            125 / 192 - 451 / 720,
            -2187 / 6784 - -12231 / 42400,
            11 / 84 - 649 / 6300,
            -1.0 / 60.0,
        ]
    ),
    c=np.array([1 / 5, 3 / 10, 4 / 5, 8 / 9, 1.0, 1.0]),
)


class _Dopri5Interpolation(FourthOrderPolynomialInterpolation):
    c_mid = np.array(
        [
            6025192743 / 30085553152 / 2,
            0,
            51252292925 / 65400821598 / 2,
            -2691868925 / 45128329728 / 2,
            187940372067 / 1594534317056 / 2,
            -1776094331 / 19743644256 / 2,
            11237099 / 235043384 / 2,
        ]
    )


class Dopri5(AbstractERK):
    r"""Dormand-Prince's 5/4 method.

    5th order Runge--Kutta method. Has an embedded 4th order method.

    ??? Reference

        ```bibtex
        @article{dormand1980family,
            author={Dormand, J. R. and Prince, P. J.},
            title={A family of embedded {R}unge--{K}utta formulae},
            journal={J. Comp. Appl. Math},
            year={1980},
            volume={6},
            pages={19--26}
        }
        ```
    """
    tableau = _dopri5_tableau
    interpolation_cls = _Dopri5Interpolation
    order = 5


def dopri5(vector_field: Callable[[Scalar, PyTree, PyTree], PyTree], **kwargs):
    return Dopri5(term=ODETerm(vector_field=vector_field), **kwargs)
