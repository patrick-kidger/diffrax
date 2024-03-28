from typing import ClassVar

import numpy as np

from .base import AbstractStratonovichSolver
from .srk import (
    AbstractSRK,
    GeneralCoeffs,
    SpaceTimeLevyAreaTableau,
    StochasticButcherTableau,
)


cfs_w = GeneralCoeffs(
    a=(np.array([0.0]), np.array([0.0, 5 / 6])),
    b=np.array([0.0, 0.4, 0.6]),
)

cfs_hh = GeneralCoeffs(
    a=(np.array([1.0]), np.array([1.0, 0.0])),
    b=np.array([0.0, 1.2, -1.2]),
)

cfs_bm = SpaceTimeLevyAreaTableau[GeneralCoeffs](
    coeffs_w=cfs_w,
    coeffs_hh=cfs_hh,
)

_tab = StochasticButcherTableau(
    c=np.array([0.0, 5 / 6]),
    b_sol=np.array([0.0, 0.4, 0.6]),
    b_error=None,
    a=[np.array([0.0]), np.array([0.0, 5 / 6])],
    cfs_bm=cfs_bm,
    ignore_stage_f=np.array([True, False, False]),
)


class GeneralShARK(AbstractSRK, AbstractStratonovichSolver):
    r"""A generalised version of the ShARK method which now works for
    any SDE, not only those with additive noise.
    Applied to SDEs with additive noise, it still has strong order 1.5.
    Uses two evaluations of the drift vector field and three evaluations
    of the diffusion vector field per step. For general SDEs, the strong
    error is similar to that of three steps of Heun's method.
    This is the recommended solver for general SDEs unless the noise vector filed is
    commutative in the Lie bracket, in which case SlowRK is recommended.

    Based on equation $(6.1)$ from

    ??? cite "Reference"

        ```bibtex
        @misc{foster2023high,
          title={High order splitting methods for SDEs satisfying
            a commutativity condition},
          author={James Foster and Goncalo dos Reis and Calum Strange},
          year={2023},
          eprint={2210.17543},
          archivePrefix={arXiv},
          primaryClass={math.NA}
        ```
    """

    tableau: ClassVar[StochasticButcherTableau] = _tab

    def order(self, terms):
        return 2

    def strong_order(self, terms):
        return 1.5
