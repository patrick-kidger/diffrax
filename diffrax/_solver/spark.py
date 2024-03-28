from typing import ClassVar

import numpy as np

from .base import AbstractStratonovichSolver
from .srk import (
    AbstractSRK,
    GeneralCoeffsWithError,
    SpaceTimeLevyAreaTableau,
    StochasticButcherTableau,
)


_x1 = (3 - np.sqrt(3)) / 6
_x2 = np.sqrt(3) / 3

cfs_w = GeneralCoeffsWithError(
    a=(np.array([0.5]), np.array([0.0, 1.0])),
    b=np.array([_x1, _x2, _x1]),
    b_error=np.array([_x1 - 0.5, _x2, _x1 - 0.5]),
)

cfs_hh = GeneralCoeffsWithError(
    a=(np.array([np.sqrt(3.0)]), np.array([0.0, 0.0])),
    b=np.array([1.0, 0.0, -1.0]),
    b_error=np.array([1.0, 0.0, -1.0]),
)

cfs_bm = SpaceTimeLevyAreaTableau[GeneralCoeffsWithError](
    coeffs_w=cfs_w,
    coeffs_hh=cfs_hh,
)

_tab = StochasticButcherTableau(
    c=np.array([0.5, 1.0]),
    b_sol=np.array([_x1, _x2, _x1]),
    a=[np.array([0.5]), np.array([0.0, 1.0])],
    b_error=np.array([_x1 - 0.5, _x2, _x1 - 0.5]),
    cfs_bm=cfs_bm,
)


class SPaRK(AbstractSRK, AbstractStratonovichSolver):
    r"""The Splitting Path Runge-Kutta method by James Foster.
    It uses three evaluations of the vector field per step and
    has the following strong orders of convergence:
    - 1.5 for SDEs with additive noise
    - 1.0 for SDEs with commutative noise
    - 0.5 for general SDEs.
    Despite being slower than methods like ShARK or SRA1, it works for a wider class
    of SDEs. It is based on Definition 1.6 from

    ??? cite "Reference"

        ```bibtex
        @misc{foster2023convergence,
            title={On the convergence of adaptive approximations
            for stochastic differential equations},
            author={James Foster},
            year={2023},
            archivePrefix={arXiv},
            primaryClass={math.NA}
        }
        ```
    """

    tableau: ClassVar[StochasticButcherTableau] = _tab

    def order(self, terms):
        return 2

    def strong_order(self, terms):
        return 1.5
