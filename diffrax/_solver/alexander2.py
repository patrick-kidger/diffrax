from collections.abc import Callable
from typing import ClassVar

import equinox.internal as eqxi
import numpy as np
import optimistix as optx

from .._local_interpolation import ThirdOrderHermitePolynomialInterpolation
from .._root_finder import VeryChord, with_stepsize_controller_tols
from .runge_kutta import AbstractSDIRK, ButcherTableau


gamma = 1 - 0.5 * np.sqrt(2)

_alexander2_tableau = ButcherTableau(
    a_lower=(np.array([1 - gamma]),),
    b_sol=np.array([1 - gamma, gamma]),
    b_error=np.array([1 - gamma, gamma - 1]),
    c=np.array([1.0]),
    a_diagonal=np.array([gamma, gamma]),
    a_predictor=(np.array([1.0]),),
)


class Alexander2(AbstractSDIRK):
    r"""Alexander's 2/1 method.

    A-L-stable stiffly accurate 2nd order SDIRK method. Has an embedded 1st
    order method for adaptive step sizing. Uses 2 stages. Uses 3rd order
    Hermite interpolation for dense/ts output.

    ??? cite "Reference"

        ```bibtex
        @article{alexander1977diagonally,
          title={Diagonally Implicit Runge--Kutta Methods for Stiff O.D.E.'s},
          author={Alexander, Roger},
          year={1977},
          journal={SIAM Journal on Numerical Analysis},
          volume={14},
          number={6},
          pages = {1006--1021}
        }
        ```
    """

    tableau: ClassVar[ButcherTableau] = _alexander2_tableau
    interpolation_cls: ClassVar[
        Callable[..., ThirdOrderHermitePolynomialInterpolation]
    ] = ThirdOrderHermitePolynomialInterpolation.from_k

    root_finder: optx.AbstractRootFinder = with_stepsize_controller_tols(VeryChord)()
    root_find_max_steps: int = 10

    def order(self, terms):
        del terms
        return 2


eqxi.doc_remove_args("scan_kind")(Alexander2.__init__)
Alexander2.__init__.__doc__ = """**Arguments:**

- `root_finder`: an [Optimistix](https://github.com/patrick-kidger/optimistix) root
    finder to solve the implicit problem at each stage.
- `root_find_max_steps`: the maximum number of steps that the root finder is allowed to
    make before unconditionally rejecting the step. (And trying again with whatever
    smaller step that adaptive stepsize controller proposes.)
"""
