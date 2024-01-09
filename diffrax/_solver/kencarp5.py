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


_γ = 41 / 200
_b_sol = np.array(
    [
        -872700587467 / 9133579230613,
        0,
        0,
        22348218063261 / 9555858737531,
        -1143369518992 / 8141816002931,
        -39379526789629 / 19018526304540,
        32727382324388 / 42900044865799,
        _γ,
    ]
)
_b_sol_embedded = np.array(
    [
        -975461918565 / 9796059967033,
        0,
        0,
        78070527104295 / 32432590147079,
        -548382580838 / 3424219808633,
        -33438840321285 / 15594753105479,
        3629800801594 / 4656183773603,
        4035322873751 / 18575991585200,
    ]
)
_b_error = _b_sol - _b_sol_embedded
_c = np.array(
    [
        41 / 100,
        2935347310677 / 11292855782101,
        1426016391358 / 7196633302097,
        92 / 100,
        24 / 100,
        3 / 5,
        1.0,
    ]
)
_c_ratio = _c[1] / _c[0]
_c_ratio2 = _c[2] / _c[0]
_c_ratio3 = _c[3] / _c[0]
_c_ratio4 = _c[4] / _c[1]
_c_ratio5 = _c[5] / _c[3]
_c_ratio6 = _c[6] / _c[3]

_explicit_tableau = ButcherTableau(
    a_lower=(
        np.array([41 / 100]),
        np.array([367902744464 / 2072280473677, 677623207551 / 8224143866563]),
        np.array([1268023523408 / 10340822734521, 0, 1029933939417 / 13636558850479]),
        np.array(
            [
                14463281900351 / 6315353703477,
                0,
                66114435211212 / 5879490589093,
                -54053170152839 / 4284798021562,
            ]
        ),
        np.array(
            [
                14090043504691 / 34967701212078,
                0,
                15191511035443 / 11219624916014,
                -18461159152457 / 12425892160975,
                -281667163811 / 9011619295870,
            ]
        ),
        np.array(
            [
                19230459214898 / 13134317526959,
                0,
                21275331358303 / 2942455364971,
                -38145345988419 / 4862620318723,
                -1 / 8,
                -1 / 8,
            ]
        ),
        np.array(
            [
                -19977161125411 / 11928030595625,
                0,
                -40795976796054 / 6384907823539,
                177454434618887 / 12078138498510,
                782672205425 / 8267701900261,
                -69563011059811 / 9646580694205,
                7356628210526 / 4942186776405,
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
        np.array([41 / 400, -567603406766 / 11931857230679]),
        np.array([683785636431 / 9252920307686, 0, -110385047103 / 1367015193373]),
        np.array(
            [
                3016520224154 / 10081342136671,
                0,
                30586259806659 / 12414158314087,
                -22760509404356 / 11113319521817,
            ]
        ),
        np.array(
            [
                218866479029 / 1489978393911,
                0,
                638256894668 / 5436446318841,
                -1179710474555 / 5321154724896,
                -60928119172 / 8023461067671,
            ]
        ),
        np.array(
            [
                1020004230633 / 5715676835656,
                0,
                25762820946817 / 25263940353407,
                -2161375909145 / 9755907335909,
                -211217309593 / 5846859502534,
                -4269925059573 / 7827059040719,
            ]
        ),
        _b_sol[:-1],
    ),
    b_sol=_b_sol,
    b_error=_b_error,
    c=_c,
    a_diagonal=np.array([0, _γ, _γ, _γ, _γ, _γ, _γ, _γ]),
    # See
    # https://docs.kidger.site/diffrax/devdocs/predictor_dirk/
    # for the construction of the a_predictor tableau, which is new here.
    # They do also discuss this a little bit in Sections 2.1.7 and 3.2.2, but don't
    # really pick any particular answer.
    a_predictor=(
        np.array([1.0]),
        np.array([1 - _c_ratio, _c_ratio]),
        np.array([1 - _c_ratio2, _c_ratio2, 0]),  # c3 < c2 so use first two stages
        np.array([1 - _c_ratio3, _c_ratio3, 0, 0]),  # c4 < c2 also
        np.array([1 - _c_ratio4, 0, _c_ratio4, 0, 0]),  # c3≈c6 so use that
        np.array([1 - _c_ratio5, 0, 0, 0, _c_ratio5, 0]),  # arbitrary linear interp
        np.array([1 - _c_ratio6, 0, 0, 0, _c_ratio6, 0, 0]),  # arbitrary linear interp
    ),
)


class _KenCarp5Interpolation(KenCarpInterpolation):
    coeffs = np.array(
        [
            [
                -9257016797708 / 5021505065439,
                43486358583215 / 12773830924787,
                -17674230611817 / 10670229744614,
            ],
            [0, 0, 0],
            [0, 0, 0],
            [
                26096422576131 / 11239449250142,
                -91478233927265 / 11067650958493,
                65168852399939 / 7868540260826,
            ],
            [
                92396832856987 / 20362823103730,
                -79368583304911 / 10890268929626,
                15494834004392 / 5936557850923,
            ],
            [
                30029262896817 / 10175596800299,
                -12239297817655 / 9152339842473,
                -99329723586156 / 26959484932159,
            ],
            [
                -26136350496073 / 3983972220547,
                115839755401235 / 10719374521269,
                -19024464361622 / 5461577185407,
            ],
            [
                -5289405421727 / 3760307252460,
                5843115559534 / 2180450260947,
                -6511271360970 / 6095937251113,
            ],
        ]
    )


class KenCarp5(AbstractRungeKutta, AbstractImplicitSolver):
    """Kennedy--Carpenter's 5/4 IMEX method.

    5th order ERK-ESDIRK implicit-explicit (IMEX) method. The implicit part is stiffly
    accurate and A-L stable. Has an embedded 4th order method for adaptive step sizing.
    Uses 8 stages. Uses 3rd order interpolation for dense/ts output.

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
        Callable[..., _KenCarp5Interpolation]
    ] = _KenCarp5Interpolation

    root_finder: optx.AbstractRootFinder = with_stepsize_controller_tols(VeryChord)()
    root_find_max_steps: int = 10

    def order(self, terms):
        return 5
