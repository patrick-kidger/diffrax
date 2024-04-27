from typing import ClassVar

import numpy as np

from .base import AbstractStratonovichSolver
from .srk import AbstractSRK, AdditiveCoeffs, StochasticButcherTableau


_coeffs_w = AdditiveCoeffs(
    a=np.array([0.0, 3 / 4]),
    b_sol=np.array(1.0),
)

_coeffs_hh = AdditiveCoeffs(
    a=np.array([0.0, 1.5]),
    b_sol=np.array(0.0),
)

_tab = StochasticButcherTableau(
    a=[np.array([3 / 4])],
    b_sol=np.array([1 / 3, 2 / 3]),
    b_error=np.array([-2 / 3, 2 / 3]),
    c=np.array([3 / 4]),
    coeffs_w=_coeffs_w,
    coeffs_hh=_coeffs_hh,
    coeffs_kk=None,
    ignore_stage_f=None,
    ignore_stage_g=None,
)


class SRA1(AbstractSRK, AbstractStratonovichSolver):
    r"""The SRA1 method for additive-noise SDEs.

    Makes two evaluations of the drift and diffusion per step and has a strong order
    1.5.

    See also [`diffrax.ShARK`][], which is very similar.

    ??? cite "Reference"

        ```bibtex
        @article{rossler2010runge
            author = {Andreas R\"{o}\ss{}ler},
            title = {Runge–Kutta Methods for the Strong Approximation of Solutions of
                     Stochastic Differential Equations},
            journal = {SIAM Journal on Numerical Analysis},
            volume = {48},
            number = {3},
            pages = {922--952},
            year = {2010},
            doi = {10.1137/09076636X},
        ```
    """

    tableau: ClassVar[StochasticButcherTableau] = _tab

    def order(self, terms):
        return 2

    def strong_order(self, terms):
        return 1.5
