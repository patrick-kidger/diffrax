from typing import ClassVar

import numpy as np

from .base import AbstractStratonovichSolver
from .srk import AbstractSRK, GeneralCoeffs, StochasticButcherTableau


_x1 = (3 - np.sqrt(3)) / 6
_x2 = np.sqrt(3) / 3

_coeffs_w = GeneralCoeffs(
    a=(np.array([0.5]), np.array([0.0, 1.0])),
    b_sol=np.array([_x1, _x2, _x1]),
    b_error=np.array([_x1 - 0.5, _x2, _x1 - 0.5]),
)

_coeffs_hh = GeneralCoeffs(
    a=(np.array([np.sqrt(3.0)]), np.array([0.0, 0.0])),
    b_sol=np.array([1.0, 0.0, -1.0]),
    b_error=np.array([1.0, 0.0, -1.0]),
)

_tab = StochasticButcherTableau(
    a=[np.array([0.5]), np.array([0.0, 1.0])],
    b_sol=np.array([_x1, _x2, _x1]),
    b_error=np.array([_x1 - 0.5, _x2, _x1 - 0.5]),
    c=np.array([0.5, 1.0]),
    coeffs_w=_coeffs_w,
    coeffs_hh=_coeffs_hh,
    coeffs_kk=None,
    ignore_stage_f=None,
    ignore_stage_g=None,
)


class SPaRK(AbstractSRK, AbstractStratonovichSolver):
    r"""The Splitting Path Runge-Kutta method.

    It uses three evaluations of the drift and diffusion per step, and has the following
    strong orders of convergence:

    - 1.5 for SDEs with additive noise (but [`diffrax.ShARK`][] is recommended instead)
    - 1.0 for Stratonovich SDEs with commutative noise
    ([`diffrax.SlowRK`][] is recommended instead)
    - 0.5 for Stratonovich SDEs with general noise.

    For general Stratonovich SDEs this is equally precise as three steps of
    [`diffrax.Heun`][] or a single step of [`diffrax.GeneralShARK`][]. Unlike those,
    this method has an embedded error estimate, so it is the recommended choice for
    adaptive time-stepping. Otherwise, [`diffrax.GeneralShARK`][] is more efficient.

    ??? cite "Reference"

        This solver is based on Definition 1.6 from

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
        return 0.5
