from collections.abc import Callable
from typing import ClassVar, Optional, TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
import optimistix as optx


if TYPE_CHECKING:
    from typing import ClassVar as AbstractClassVar
else:
    from equinox import AbstractClassVar
from equinox.internal import ω
from jaxtyping import Array, PyTree, Shaped

from .._custom_types import RealScalarLike, Y
from .._local_interpolation import AbstractLocalInterpolation
from .._misc import linear_rescale
from .._root_finder import VeryChord, with_stepsize_controller_tols
from .base import AbstractImplicitSolver, vector_tree_dot
from .runge_kutta import (
    AbstractRungeKutta,
    ButcherTableau,
    CalculateJacobian,
    MultiButcherTableau,
)


_γ = 1767732205903 / 4055673282236
_b_sol = np.array(
    [
        1471266399579 / 7840856788654,
        -4482444167858 / 7529755066697,
        11266239266428 / 11593286722821,
        _γ,
    ]
)
_b_sol_embedded = np.array(
    [
        2756255671327 / 12835298489170,
        -10771552573575 / 22201958757719,
        9247589265047 / 10645013368117,
        2193209047091 / 5459859503100,
    ]
)
_b_error = _b_sol - _b_sol_embedded
_c = np.array([2 * _γ, 3 / 5, 1.0])
_c_ratio = _c[1] / _c[0]
_c_ratio2 = _c[2] / _c[0]

_explicit_tableau = ButcherTableau(
    a_lower=(
        np.array([2 * _γ]),
        np.array([5535828885825 / 10492691773637, 788022342437 / 10882634858940]),
        np.array(
            [
                6485989280629 / 16251701735622,
                -4246266847089 / 9704473918619,
                10755448449292 / 10357097424841,
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
        np.array([2746238789719 / 10658868560708, -640167445237 / 6845629431997]),
        _b_sol[:-1],
    ),
    b_sol=_b_sol,
    b_error=_b_error,
    c=_c,
    a_diagonal=np.array([0, _γ, _γ, _γ]),
    # See
    # https://docs.kidger.site/diffrax/devdocs/predictor_dirk/
    # for the construction of the a_predictor tableau, which is new here.
    # They do also discuss this a little bit in Sections 2.1.7 and 3.2.2, but don't
    # really pick any particular answer.
    a_predictor=(
        np.array([1.0]),
        np.array([1 - _c_ratio, _c_ratio]),
        np.array([1 - _c_ratio2, _c_ratio2, 0]),  # c3 < c2 so use first two stages
    ),
)


class KenCarpInterpolation(AbstractLocalInterpolation):
    t0: RealScalarLike
    t1: RealScalarLike
    y0: Y
    k: tuple[
        PyTree[Shaped[Array, "order ?*y"], "Y"], PyTree[Shaped[Array, "order ?*y"], "Y"]
    ]

    coeffs: AbstractClassVar[np.ndarray]

    def __init__(self, *, t0, t1, y0, y1, k):
        del y1  # exists for API compatibility
        self.t0 = t0
        self.t1 = t1
        self.y0 = y0
        self.k = k

    def evaluate(
        self, t0: RealScalarLike, t1: Optional[RealScalarLike] = None, left: bool = True
    ) -> PyTree:
        del left
        if t1 is not None:
            return self.evaluate(t1) - self.evaluate(t0)

        t = linear_rescale(self.t0, t0, self.t1)
        explicit_k, implicit_k = self.k
        k = (explicit_k**ω + implicit_k**ω).ω
        coeffs = t * jax.vmap(lambda row: jnp.polyval(row, t))(self.coeffs)
        with jax.numpy_dtype_promotion("standard"):
            return (self.y0**ω + vector_tree_dot(coeffs, k) ** ω).ω


class _KenCarp3Interpolation(KenCarpInterpolation):
    coeffs = np.array(
        [
            [-215264564351 / 13552729205753, 4655552711362 / 22874653954995],
            [17870216137069 / 13817060693119, -18682724506714 / 9892148508045],
            [-28141676662227 / 17317692491321, 34259539580243 / 13192909600954],
            [2508943948391 / 7218656332882, 584795268549 / 6622622206610],
        ]
    )


class KenCarp3(AbstractRungeKutta, AbstractImplicitSolver):
    """Kennedy--Carpenter's 3/2 IMEX method.

    3rd order ERK-ESDIRK implicit-explicit (IMEX) method. The implicit part is stiffly
    accurate and A-L stable. Has an embedded 2nd order method for adaptive step sizing.
    Uses 4 stages. Uses 2nd order interpolation for dense/ts output.

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
        Callable[..., _KenCarp3Interpolation]
    ] = _KenCarp3Interpolation

    root_finder: optx.AbstractRootFinder = with_stepsize_controller_tols(VeryChord)()
    root_find_max_steps: int = 10

    def order(self, terms):
        return 3
