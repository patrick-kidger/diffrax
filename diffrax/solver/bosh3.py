from typing import Callable

from ..custom_types import PyTree, Scalar
from ..local_interpolation import FourthOrderPolynomialInterpolation
from ..misc import frozenarray
from ..term import ODETerm
from .runge_kutta import ButcherTableau, RungeKutta


_bosh3_tableau = ButcherTableau(
    alpha=frozenarray([1 / 2, 3 / 4, 1.]),
    beta=(frozenarray([1 / 2]), frozenarray([0., 3 / 4]), frozenarray([2 / 9, 1 / 3, 4 / 9]),
          ),
    c_sol=frozenarray([2 / 9, 1 / 3, 4 / 9, 0.]),
    c_error=frozenarray([2 / 9 - 7 / 24, 1 / 3 - 1 / 4, 4 / 9 - 1 / 3, -1 / 8]),
    c_mid=frozenarray([0, 0.5, 0, 0])
)


class _Bosh3Interpolation(FourthOrderPolynomialInterpolation):
    # I don't think this is well-chosen -- I think this is just a simple choice to get an
    # approximation for y at the middle of each step, and that better choices are probably available.
    c_mid = frozenarray([0, 0.5, 0, 0])


def bosh3(vector_field: Callable[[Scalar, PyTree, PyTree], PyTree], **kwargs,):
    return RungeKutta(
        terms=(ODETerm(vector_field=vector_field),),
        tableau=_bosh3_tableau,
        interpolation_cls=_Bosh3Interpolation,
        **kwargs
    )
