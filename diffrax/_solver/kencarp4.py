from collections.abc import Callable
from typing import ClassVar

import numpy as np
import optimistix as optx

from .._root_finder import VeryChord, with_stepsize_controller_tols
from .base import AbstractImplicitSolver
from .kencarp3 import KenCarpInterpolation
from .runge_kutta import (
    AbstractRungeKutta,
    ButcherTableau,
    CalculateJacobian,
    MultiButcherTableau,
)


_γ = 0.25
_b_sol = np.array([82889 / 524892, 0, 15625 / 83664, 69875 / 102672, -2260 / 8211, _γ])
_b_sol_embedded = np.array(
    [
        4586570599 / 29645900160,
        0,
        178811875 / 945068544,
        814220225 / 1159782912,
        -3700637 / 11593932,
        61727 / 225920,
    ]
)
_b_error = _b_sol - _b_sol_embedded
_c = np.array([0.5, 83 / 250, 31 / 50, 17 / 20, 1.0])
_c_ratio = _c[1] / _c[0]
_c_ratio2 = _c[2] / _c[0]
_c_ratio3 = _c[3] / _c[2]
_c_ratio4 = _c[4] / _c[3]

_explicit_tableau = ButcherTableau(
    a_lower=(
        np.array([0.5]),
        np.array([13861 / 62500, 6889 / 62500]),
        np.array(
            [
                -116923316275 / 2393684061468,
                -2731218467317 / 15368042101831,
                9408046702089 / 11113171139209,
            ]
        ),
        np.array(
            [
                -451086348788 / 2902428689909,
                -2682348792572 / 7519795681897,
                12662868775082 / 11960479115383,
                3355817975965 / 11060851509271,
            ]
        ),
        np.array(
            [
                647845179188 / 3216320057751,
                73281519250 / 8382639484533,
                552539513391 / 3454668386233,
                3354512671639 / 8306763924573,
                4040 / 17871,
            ]
        ),
    ),
    b_sol=_b_sol,
    b_error=_b_error,
    c=_c,
)

_implicit_tableau = ButcherTableau(
    a_lower=(
        np.array([_γ]),
        np.array([8611 / 62500, -1743 / 31250]),
        np.array([5012029 / 34652500, -654441 / 2922500, 174375 / 388108]),
        np.array(
            [
                15267082809 / 155376265600,
                -71443401 / 120774400,
                730878875 / 902184768,
                2285395 / 8070912,
            ]
        ),
        _b_sol[:-1],
    ),
    b_sol=_b_sol,
    b_error=_b_error,
    c=_c,
    a_diagonal=np.array([0, _γ, _γ, _γ, _γ, _γ]),
    # See
    # https://docs.kidger.site/diffrax/devdocs/predictor_dirk/
    # for the construction of the a_predictor tableau, which is new here.
    # They do also discuss this a little bit in Sections 2.1.7 and 3.2.2, but don't
    # really pick any particular answer.
    a_predictor=(
        np.array([1.0]),
        np.array([1 - _c_ratio, _c_ratio]),
        np.array([1 - _c_ratio2, _c_ratio2, 0]),  # c3 < c2 so use first two stages
        np.array([1 - _c_ratio3, 0, 0, _c_ratio3]),  # arbitrarily use linear interp.
        np.array([1 - _c_ratio4, 0, 0, 0, _c_ratio4]),  # also arbitrary linear interp.
    ),
)


class _KenCarp4Interpolation(KenCarpInterpolation):
    coeffs = np.array(
        [
            [
                6818779379841 / 7100303317025,
                -54480133 / 30881146,
                6943876665148 / 7220017795957,
            ],
            [0.0, 0.0, 0.0],
            [
                2173542590792 / 12501825683035,
                -11436875 / 14766696,
                7640104374378 / 9702883013639,
            ],
            [
                -31592104683404 / 5083833661969,
                174696575 / 18121608,
                -20649996744609 / 7521556579894,
            ],
            [
                61146701046299 / 7138195549469,
                -12120380 / 966161,
                8854892464581 / 2390941311638,
            ],
            [
                -17219254887155 / 4939391667607,
                3843 / 706,
                -11397109935349 / 6675773540249,
            ],
        ]
    )


class KenCarp4(AbstractRungeKutta, AbstractImplicitSolver):
    """Kennedy--Carpenter's 4/3 IMEX method.

    4th order ERK-ESDIRK implicit-explicit (IMEX) method. The implicit part is stiffly
    accurate and A-L stable. Has an embedded 3rd order method for adaptive step sizing.
    Uses 6 stages. Uses 3rd order interpolation for dense/ts output.

    This should be called with `terms=MultiTerm(explicit_term, implicit_term)`.

    ??? Reference

        ```bibtex
        @article{kennedy2003additive,
          title={Additive Runge--Kutta schemes for convection-diffusion-reaction
                 equations},
          author={Kennedy, Christopher A and Carpenter, Mark H},
          journal={Applied numerical mathematics},
          volume={44},
          number={1-2},
          pages={139--181},
          year={2003},
          publisher={Elsevier}
        }
        ```
    """

    tableau: ClassVar[MultiButcherTableau] = MultiButcherTableau(
        _explicit_tableau, _implicit_tableau
    )
    calculate_jacobian: ClassVar[CalculateJacobian] = CalculateJacobian.second_stage
    interpolation_cls: ClassVar[
        Callable[..., _KenCarp4Interpolation]
    ] = _KenCarp4Interpolation

    root_finder: optx.AbstractRootFinder = with_stepsize_controller_tols(VeryChord)()
    root_find_max_steps: int = 10

    def order(self, terms):
        return 4
