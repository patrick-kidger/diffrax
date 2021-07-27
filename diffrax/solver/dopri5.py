from typing import Callable

from ..custom_types import PyTree, Scalar
from ..misc import frozenarray
from ..term import ODETerm
from .runge_kutta import ButcherTableau, RungeKutta


_dopri5_tableau = ButcherTableau(
    alpha=frozenarray([1 / 5, 3 / 10, 4 / 5, 8 / 9, 1., 1.]),
    beta=(
        frozenarray([1 / 5]),
        frozenarray([3 / 40, 9 / 40]),
        frozenarray([44 / 45, -56 / 15, 32 / 9]),
        frozenarray([19372 / 6561, -25360 / 2187, 64448 / 6561, -212 / 729]),
        frozenarray([9017 / 3168, -355 / 33, 46732 / 5247, 49 / 176, -5103 / 18656]),
        frozenarray([35 / 384, 0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84])
    ),
    c_sol=frozenarray([35 / 384, 0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84, 0]),
    c_error=frozenarray([
        35 / 384 - 1951 / 21600,
        0,
        500 / 1113 - 22642 / 50085,
        125 / 192 - 451 / 720,
        -2187 / 6784 - -12231 / 42400,
        11 / 84 - 649 / 6300,
        -1. / 60.
    ])
)


def dopri5(vector_field: Callable[[Scalar, PyTree, PyTree], PyTree], **kwargs,):
    return RungeKutta(terms=(ODETerm(vector_field=vector_field),), tableau=_dopri5_tableau, **kwargs)
