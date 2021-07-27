from typing import Callable

from ..custom_types import PyTree, Scalar
from ..misc import frozenarray
from ..term import ODETerm
from .runge_kutta import ButcherTableau, RungeKutta


_fehlberg2_tableau = ButcherTableau(
    alpha=frozenarray([1 / 2, 1.]),
    beta=(frozenarray([1 / 2]), frozenarray([1 / 256, 255 / 256]),
          ),
    c_sol=frozenarray([1 / 512, 255 / 256, 1 / 512]),
    c_error=frozenarray([-1 / 512, 0, 1 / 512])
)


def fehlberg2(vector_field: Callable[[Scalar, PyTree, PyTree], PyTree], **kwargs,):
    return RungeKutta(terms=(ODETerm(vector_field=vector_field),), tableau=_fehlberg2_tableau, **kwargs)
