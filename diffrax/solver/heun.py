from typing import Callable, Optional

import numpy as np

from ..brownian import AbstractBrownianPath
from ..custom_types import PyTree, Scalar
from ..local_interpolation import FourthOrderPolynomialInterpolation
from ..term import ControlTerm, MultiTerm, ODETerm
from .runge_kutta import AbstractERK, ButcherTableau


_heun_tableau = ButcherTableau(
    alpha=np.array([1.0]),
    beta=(np.array([1.0]),),
    c_sol=np.array([0.5, 0.5]),
    c_error=np.array([0.5, -0.5]),
)


class _HeunInterpolation(FourthOrderPolynomialInterpolation):
    # I don't think this is well-chosen -- I think this is just a simple choice to get
    # an approximation for y at the middle of each step, and that better choices are
    # probably available.
    c_mid = np.array([0, 0.5])


class Heun(AbstractERK):
    tableau = _heun_tableau
    interpolation_cls = _HeunInterpolation
    order = 2


def heun(
    vector_field: Callable[[Scalar, PyTree, PyTree], PyTree],
    diffusion: Optional[Callable[[Scalar, PyTree, PyTree], PyTree]] = None,
    bm: Optional[AbstractBrownianPath] = None,
    **kwargs,
):
    if diffusion is None:
        if bm is not None:
            raise ValueError
        return Heun(term=ODETerm(vector_field=vector_field), **kwargs)
    else:
        if bm is None:
            raise ValueError
        term = MultiTerm(
            terms=(
                ODETerm(vector_field=vector_field),
                ControlTerm(vector_field=diffusion, control=bm),
            )
        )
        return Heun(term=term, **kwargs)
