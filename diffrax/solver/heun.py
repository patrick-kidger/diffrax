from typing import Callable, Optional

from ..brownian import AbstractBrownianPath
from ..custom_types import PyTree, Scalar
from ..misc import frozenarray
from ..term import ControlTerm, ODETerm
from .runge_kutta import ButcherTableau, RungeKutta


_heun_tableau = ButcherTableau(
    alpha=frozenarray([1.]), beta=(frozenarray([1.]),), c_sol=frozenarray([0.5, 0.5]), c_error=frozenarray([0.5, -0.5])
)


def heun(
    drift: Callable[[Scalar, PyTree, PyTree], PyTree],
    diffusion: Optional[Callable[[Scalar, PyTree, PyTree], PyTree]] = None,
    bm: Optional[AbstractBrownianPath] = None,
    **kwargs,
):
    if diffusion is None:
        assert bm is None
        return RungeKutta(terms=(ODETerm(vector_field=drift),), tableau=_heun_tableau, **kwargs)
    else:
        assert bm is not None
        return RungeKutta(
            terms=(ODETerm(vector_field=drift), ControlTerm(vector_field=diffusion, control=bm)),
            tableau=_heun_tableau,
            **kwargs
        )
