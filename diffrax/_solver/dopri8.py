from collections.abc import Callable
from typing import ClassVar

import equinox.internal as eqxi
import jax
import jax.numpy as jnp
import numpy as np
from equinox.internal import ω
from jaxtyping import Array, PyTree, Shaped

from .._custom_types import RealScalarLike, Y
from .._local_interpolation import AbstractLocalInterpolation
from .._misc import linear_rescale
from .base import vector_tree_dot
from .runge_kutta import AbstractERK, ButcherTableau


_dopri8_tableau = ButcherTableau(
    a_lower=(
        np.array([1 / 18]),
        np.array([1 / 48, 1 / 16]),
        np.array([1 / 32, 0, 3 / 32]),
        np.array([5 / 16, 0, -75 / 64, 75 / 64]),
        np.array([3 / 80, 0, 0, 3 / 16, 3 / 20]),
        np.array(
            [
                29443841 / 614563906,
                0,
                0,
                77736538 / 692538347,
                -28693883 / 1125000000,
                23124283 / 1800000000,
            ]
        ),
        np.array(
            [
                16016141 / 946692911,
                0,
                0,
                61564180 / 158732637,
                22789713 / 633445777,
                545815736 / 2771057229,
                -180193667 / 1043307555,
            ]
        ),
        np.array(
            [
                39632708 / 573591083,
                0,
                0,
                -433636366 / 683701615,
                -421739975 / 2616292301,
                100302831 / 723423059,
                790204164 / 839813087,
                800635310 / 3783071287,
            ]
        ),
        np.array(
            [
                246121993 / 1340847787,
                0,
                0,
                -37695042795 / 15268766246,
                -309121744 / 1061227803,
                -12992083 / 490766935,
                6005943493 / 2108947869,
                393006217 / 1396673457,
                123872331 / 1001029789,
            ]
        ),
        np.array(
            [
                -1028468189 / 846180014,
                0,
                0,
                8478235783 / 508512852,
                1311729495 / 1432422823,
                -10304129995 / 1701304382,
                -48777925059 / 3047939560,
                15336726248 / 1032824649,
                -45442868181 / 3398467696,
                3065993473 / 597172653,
            ]
        ),
        np.array(
            [
                185892177 / 718116043,
                0,
                0,
                -3185094517 / 667107341,
                -477755414 / 1098053517,
                -703635378 / 230739211,
                5731566787 / 1027545527,
                5232866602 / 850066563,
                -4093664535 / 808688257,
                3962137247 / 1805957418,
                65686358 / 487910083,
            ]
        ),
        np.array(
            [
                403863854 / 491063109,
                0,
                0,
                -5068492393 / 434740067,
                -411421997 / 543043805,
                652783627 / 914296604,
                11173962825 / 925320556,
                -13158990841 / 6184727034,
                3936647629 / 1978049680,
                -160528059 / 685178525,
                248638103 / 1413531060,
                0,
            ]
        ),
        np.array(
            [
                14005451 / 335480064,
                0,
                0,
                0,
                0,
                -59238493 / 1068277825,
                181606767 / 758867731,
                561292985 / 797845732,
                -1041891430 / 1371343529,
                760417239 / 1151165299,
                118820643 / 751138087,
                -528747749 / 2220607170,
                1 / 4,
            ]
        ),
    ),
    b_sol=np.array(
        [
            14005451 / 335480064,
            0,
            0,
            0,
            0,
            -59238493 / 1068277825,
            181606767 / 758867731,
            561292985 / 797845732,
            -1041891430 / 1371343529,
            760417239 / 1151165299,
            118820643 / 751138087,
            -528747749 / 2220607170,
            1 / 4,
            0,
        ]
    ),
    b_error=np.array(
        [
            14005451 / 335480064 - 13451932 / 455176623,
            0,
            0,
            0,
            0,
            -59238493 / 1068277825 - -808719846 / 976000145,
            181606767 / 758867731 - 1757004468 / 5645159321,
            561292985 / 797845732 - 656045339 / 265891186,
            -1041891430 / 1371343529 - -3867574721 / 1518517206,
            760417239 / 1151165299 - 465885868 / 322736535,
            118820643 / 751138087 - 53011238 / 667516719,
            -528747749 / 2220607170 - 2 / 45,
            1 / 4,
            0,
        ]
    ),
    c=np.array(
        [
            1 / 18,
            1 / 12,
            1 / 8,
            5 / 16,
            3 / 8,
            59 / 400,
            93 / 200,
            5490023248 / 9719169821,
            13 / 20,
            1201146811 / 1299019798,
            1,
            1,
            1,
        ]
    ),
)

_vmap_polyval = jax.vmap(jnp.polyval, in_axes=(0, None))


class _Dopri8Interpolation(AbstractLocalInterpolation):
    t0: RealScalarLike
    t1: RealScalarLike
    y0: Y
    k: PyTree[Shaped[Array, "14 ?*y"], "Y"]

    def __init__(self, *, t0, t1, y0, y1, k):
        del y1  # exists for API compatibility
        self.t0 = t0
        self.t1 = t1
        self.y0 = y0
        self.k = k

    eval_coeffs = np.array(
        [
            [
                -6.3448349392860401388,
                22.1396504998094068976,
                -30.0610568289666450593,
                19.9990069333683970610,
                -6.6910181737837595697,
                1,
            ],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [
                -39.6107919852202505218,
                116.4422149550342161651,
                -121.4999627731334642623,
                52.2273532792945524050,
                -7.6142658045872677172,
                0,
            ],
            [
                20.3761213808791436958,
                -67.1451318825957197185,
                83.1721004639847717481,
                -46.8919164181093621583,
                10.7281392630428866124,
                0,
            ],
            [
                7.3347098826795362023,
                -16.5672243527496524646,
                9.5724507555993664382,
                -0.1890893225010595467,
                0.5526637063753648783,
                0,
            ],
            [
                32.8801774352459155182,
                -89.9916014847245016028,
                87.8406057677205645007,
                -35.7075975946222072821,
                4.2186562625665153803,
                0,
            ],
            [
                -10.1588990526426760954,
                22.6237489648532849093,
                -17.4152107770762969005,
                6.2736448083240352160,
                -0.6627209125361597559,
                0,
            ],
            [
                -12.5401268098782561200,
                32.2362340167355370113,
                -28.5903289514790976966,
                10.3160881272450748458,
                -1.2636789001135462218,
                0,
            ],
            [
                29.5553001484516038033,
                -82.1020315488359848644,
                81.6630950584341412934,
                -34.7650769866611817349,
                5.4106037898590422230,
                0,
            ],
            [
                -41.7923486424390588923,
                116.2662185791119533462,
                -114.9375291377009418170,
                47.7457971078225540396,
                -7.0321379067945741781,
                0,
            ],
            [
                20.3006925822100825485,
                -53.9020777466385396792,
                50.2558364226176017553,
                -19.0082099341608028453,
                2.3537586759714983486,
                0,
            ],
        ]
    )

    def evaluate(
        self, t0: RealScalarLike, t1: RealScalarLike | None = None, left: bool = True
    ) -> Y:
        del left
        if t1 is not None:
            return self.evaluate(t1) - self.evaluate(t0)
        t = linear_rescale(self.t0, t0, self.t1)
        coeffs = _vmap_polyval(jnp.asarray(self.eval_coeffs, dtype=t.dtype), t) * t
        with jax.numpy_dtype_promotion("standard"):
            return (self.y0**ω + vector_tree_dot(coeffs, self.k) ** ω).ω


class Dopri8(AbstractERK):
    """Dormand--Prince's 8/7 method.

    8th order Runge--Kutta method. Has an embedded 7th order method for adaptive step
    sizing. Uses 14 stages with FSAL. Uses 8th order interpolation for dense/ts output.

    ??? cite "References"

        Coefficients from:
        ```bibtex
        @article{prince1981high,
            author={Prince, P. J and Dormand, J. R.},
            title={High order embedded {R}unge--{K}utta formulae},
            journal={J. Comp. Appl. Math},
            year={1981},
            volume={7},
            number={1},
            pages={67--75}
        }
        ```

        Dense interpolation from:
        ```bibtex
        @article{bogacki1990interpolating,
            author={Bogacki, P. and Shampine, L. F.},
            title={Interpolating high-order {R}unge--{K}utta formulas},
            journal={Comput. Math. with Appl.},
            year={1990},
            volume={20},
            number={3},
            pages={15--24},
            doi={https://doi.org/10.1016/0898-1221(90)90027-H}
        }
        ```
    """

    tableau: ClassVar[ButcherTableau] = _dopri8_tableau
    interpolation_cls: ClassVar[
        Callable[..., _Dopri8Interpolation]
    ] = _Dopri8Interpolation

    def order(self, terms):
        del terms
        return 8


eqxi.doc_remove_args("scan_kind")(Dopri8.__init__)
Dopri8.__init__.__doc__ = """**Arguments:** None"""
