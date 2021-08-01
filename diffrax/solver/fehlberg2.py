from typing import Callable

from ..custom_types import PyTree, Scalar
from ..local_interpolation import FourthOrderPolynomialInterpolation
from ..misc import frozenarray
from ..term import ODETerm
from .runge_kutta import ButcherTableau, RungeKutta


_fehlberg2_tableau = ButcherTableau(
    alpha=frozenarray([1 / 2, 1.0]),
    beta=(frozenarray([1 / 2]), frozenarray([1 / 256, 255 / 256])),
    c_sol=frozenarray([1 / 512, 255 / 256, 1 / 512]),
    c_error=frozenarray([-1 / 512, 0, 1 / 512]),
)


class _Fehlberg2Interpolation(FourthOrderPolynomialInterpolation):
    # I don't think this is well-chosen -- I think this is just a simple choice to get
    # an approximation for y at the middle of each step, and that better choices are
    # probably available.
    c_mid = frozenarray([0, 0.5, 0])


def fehlberg2(
    vector_field: Callable[[Scalar, PyTree, PyTree], PyTree],
    **kwargs,
):
    return RungeKutta(
        terms=(ODETerm(vector_field=vector_field),),
        tableau=_fehlberg2_tableau,
        interpolation_cls=_Fehlberg2Interpolation,
        **kwargs,
    )
