from collections.abc import Callable
from typing import cast, ClassVar

import numpy as np
import optimistix as optx
from equinox.internal import ω

from .._local_interpolation import ThirdOrderHermitePolynomialInterpolation
from .._root_finder import VeryChord, with_stepsize_controller_tols
from .base import AbstractImplicitSolver
from .runge_kutta import (
    AbstractRungeKutta,
    ButcherTableau,
    CalculateJacobian,
    MultiButcherTableau,
)


ω = cast(Callable, ω)


# See
# https://docs.kidger.site/diffrax/devdocs/predictor_dirk/
# for the construction of the a_predictor tableau, which is new here.
_implicit_tableau = ButcherTableau(
    a_lower=(
        np.array([1 / 6]),
        np.array([1 / 3, 0]),
        np.array([3 / 8, 0, 3 / 8]),
    ),
    b_sol=np.array([3 / 8, 0, 3 / 8, 1 / 4]),
    b_error=np.array(
        [1 / 8, 0, -3 / 8, 1 / 4]
    ),  # just Heun; could maybe do something else
    c=np.array([1 / 3, 2 / 3, 1]),
    a_diagonal=np.array([0, 1 / 6, 1 / 3, 1 / 4]),
    a_predictor=(
        np.array([1.0]),
        np.array([-1.0, 2.0]),
        np.array([-1.0, 2.0, 0.0]),  # arbitrary choice for this one
    ),
)
_explicit_tableau = ButcherTableau(
    a_lower=(
        np.array([1 / 3]),
        np.array([1 / 6, 0.5]),
        np.array([0.5, -0.5, 1]),
    ),
    b_sol=np.array([0.5, -0.5, 1, 0]),
    b_error=np.array([0, 0.5, -1, 0.5]),  # just Heun; could maybe do something else
    c=np.array([1 / 3, 2 / 3, 1]),
)


class Sil3(AbstractRungeKutta, AbstractImplicitSolver):
    """Whitaker--Kar's fast-slow IMEX method.

    3rd order in the explicit (ERK) term; 2nd order in the implicit (EDIRK) term. Uses
    a 2nd-order embedded Heun method for adaptive step sizing. Uses 4 stages with FSAL.
    Uses 2nd order Hermite interpolation for dense/ts output.

    This should be called with `terms=MultiTerm(explicit_term, implicit_term)`.

    ??? Reference

        ```bibtex
        @article{whitaker2013implicit,
          author={Jeffrey S. Whitaker and Sajal K. Kar},
          title={Implicit–Explicit Runge–Kutta Methods for Fast–Slow Wave Problems},
          journal={Monthly Weather Review},
          year={2013},
          publisher={American Meteorological Society},
          volume={141},
          number={10},
          doi={https://doi.org/10.1175/MWR-D-13-00132.1},
          pages={3426--3434},
        }
        ```
    """

    tableau: ClassVar[MultiButcherTableau] = MultiButcherTableau(
        _explicit_tableau, _implicit_tableau
    )
    calculate_jacobian: ClassVar[CalculateJacobian] = CalculateJacobian.every_stage

    @staticmethod
    def interpolation_cls(t0, t1, y0, y1, k):
        k_explicit, k_implicit = k
        k0 = (ω(k_explicit)[0] + ω(k_implicit)[0]).ω
        k1 = (ω(k_explicit)[-1] + ω(k_implicit)[-1]).ω
        return ThirdOrderHermitePolynomialInterpolation(
            t0=t0, t1=t1, y0=y0, y1=y1, k0=k0, k1=k1
        )

    root_finder: optx.AbstractRootFinder = with_stepsize_controller_tols(VeryChord)()
    root_find_max_steps: int = 10

    def order(self, terms):
        return 2
