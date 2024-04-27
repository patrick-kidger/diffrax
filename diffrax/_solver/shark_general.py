from typing import ClassVar

import numpy as np

from .base import AbstractStratonovichSolver
from .srk import AbstractSRK, GeneralCoeffs, StochasticButcherTableau


_coeffs_w = GeneralCoeffs(
    a=(np.array([0.0]), np.array([0.0, 5 / 6])),
    b_sol=np.array([0.0, 0.4, 0.6]),
    b_error=None,
)

_coeffs_hh = GeneralCoeffs(
    a=(np.array([1.0]), np.array([1.0, 0.0])),
    b_sol=np.array([0.0, 1.2, -1.2]),
    b_error=None,
)

_tab = StochasticButcherTableau(
    a=[np.array([0.0]), np.array([0.0, 5 / 6])],
    b_sol=np.array([0.0, 0.4, 0.6]),
    b_error=None,
    c=np.array([0.0, 5 / 6]),
    coeffs_w=_coeffs_w,
    coeffs_hh=_coeffs_hh,
    coeffs_kk=None,
    ignore_stage_f=np.array([True, False, False]),
    ignore_stage_g=None,
)


class GeneralShARK(AbstractSRK, AbstractStratonovichSolver):
    r"""ShARK method for Stratonovich SDEs.

    As compared to [`diffrax.ShARK`][] this can handle any SDE, not only those with
    additive noise.

    Makes two evaluations of the drift and three evaluations of the diffusion per step.
    Has the following orders of convergence:

    - 1.5 for SDEs with additive noise (but [`diffrax.ShARK`][] is recommended instead)
    - 1.0 for Stratonovich SDEs with commutative noise
    ([`diffrax.SlowRK`][] is recommended instead)
    - 0.5 for Stratonovich SDEs with general noise.

    For general Stratonovich SDEs this is equally precise as three steps of
    [`diffrax.Heun`][] or a single step of [`diffrax.SPaRK`][], while requiring
    one fewer evaluation of the drift, so this is the recommended choice for general
    SDEs with an expensive drift vector field. If embedded error estimation is needed
    (e.g. for adaptive time-stepping) then [`diffrax.SPaRK`][] is recommended instead.

    ??? cite "Reference"

        This solver is based on equation (6.1) from

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
        return 0.5
