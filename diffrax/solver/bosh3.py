from typing import Callable

from ..custom_types import PyTree, Scalar
from ..misc import frozenarray
from ..term import ODETerm
from .runge_kutta import ButcherTableau, RungeKutta


_bosh3_tableau = ButcherTableau(
    alpha=frozenarray([1 / 2, 3 / 4, 1.]),
    beta=(frozenarray([1 / 2]), frozenarray([0., 3 / 4]), frozenarray([2 / 9, 1 / 3, 4 / 9]),
          ),
    c_sol=frozenarray([2 / 9, 1 / 3, 4 / 9, 0.]),
    c_error=frozenarray([2 / 9 - 7 / 24, 1 / 3 - 1 / 4, 4 / 9 - 1 / 3, -1 / 8])
)


def bosh3(vector_field: Callable[[Scalar, PyTree, PyTree], PyTree], **kwargs,):
    return RungeKutta(terms=(ODETerm(vector_field=vector_field),), tableau=_bosh3_tableau, **kwargs)
