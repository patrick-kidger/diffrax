from typing import ClassVar

import numpy as np

from .base import AbstractStratonovichSolver
from .srk import AbstractSRK, AdditiveCoeffs, StochasticButcherTableau


_coeffs_w = AdditiveCoeffs(
    a=np.array([0.5]),
    b_sol=np.array(1.0),
)

_coeffs_hh = AdditiveCoeffs(
    a=np.array([1.0]),
    b_sol=np.array(0.0),
)

_tab = StochasticButcherTableau(
    a=[],
    b_sol=np.array([1.0]),
    b_error=None,
    c=np.array([]),
    coeffs_w=_coeffs_w,
    coeffs_hh=_coeffs_hh,
    coeffs_kk=None,
    ignore_stage_f=None,
    ignore_stage_g=None,
)


class SEA(AbstractSRK, AbstractStratonovichSolver):
    r"""Shifted Euler method for SDEs with additive noise.

    Makes one evaluation of the drift and diffusion per step and has a strong order 1.
    Compared to [`diffrax.Euler`][], it has a better constant factor in the global
    error, and an improved local error of $O(h^2)$ instead of $O(h^{1.5})$.

    This solver is useful for solving additive-noise SDEs with as few drift and
    diffusion evaluations per step as possible.

    ??? cite "Reference"

        This solver is based on equation (5.8) in

        ```bibtex
        @article{foster2023high,
            title={High order splitting methods for SDEs satisfying a commutativity
                   condition},
            author={James Foster and Goncalo dos Reis and Calum Strange},
            year={2023},
            journal={arXiv:2210.17543},
        }
        ```
    """

    tableau: ClassVar[StochasticButcherTableau] = _tab

    def order(self, terms):
        return 1

    def strong_order(self, terms):
        return 1
