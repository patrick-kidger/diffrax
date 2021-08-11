from typing import Callable, Optional

from ..brownian import AbstractBrownianPath
from ..custom_types import PyTree, Scalar
from ..local_interpolation import FourthOrderPolynomialInterpolation
from ..misc import frozenarray
from ..term import ControlTerm, ODETerm
from .runge_kutta import ButcherTableau, RungeKutta


_heun_tableau = ButcherTableau(
    alpha=frozenarray([1.0]),
    beta=(frozenarray([1.0]),),
    c_sol=frozenarray([0.5, 0.5]),
    c_error=frozenarray([0.5, -0.5]),
)


class _HeunInterpolation(FourthOrderPolynomialInterpolation):
    # I don't think this is well-chosen -- I think this is just a simple choice to get
    # an approximation for y at the middle of each step, and that better choices are
    # probably available.
    c_mid = frozenarray([0, 0.5])


class Heun(RungeKutta):
    tableau = _heun_tableau
    interpolation_cls = _HeunInterpolation


def heun(
    vector_field: Callable[[Scalar, PyTree, PyTree], PyTree],
    diffusion: Optional[Callable[[Scalar, PyTree, PyTree], PyTree]] = None,
    bm: Optional[AbstractBrownianPath] = None,
    **kwargs,
):
    if diffusion is None:
        if bm is not None:
            raise ValueError
        return Heun(
            terms=(ODETerm(vector_field=vector_field),),
            **kwargs
        )
    else:
        if bm is None:
            raise ValueError
        return heun(
            terms=(
                ODETerm(vector_field=vector_field),
                ControlTerm(vector_field=diffusion, control=bm),
            ),
            **kwargs
        )
