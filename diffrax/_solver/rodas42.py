from typing import ClassVar

import numpy as np

from .rosenbrock import AbstractRosenbrock, RosenbrockTableau


_tableau = RosenbrockTableau(
    a_lower=(
        np.array([1.4028884]),
        np.array([0.6581212688557198, -1.320936088384301]),
        np.array([7.131197445744498, 16.02964143958207, -5.561572550509766]),
        np.array(
            [
                22.73885722420363,
                67.38147284535289,
                -31.21877493038560,
                0.7285641833203814,
            ]
        ),
        np.array(
            [
                22.73885722420363,
                67.38147284535289,
                -31.21877493038560,
                0.7285641833203814,
                1.0,
            ]
        ),
    ),
    c_lower=(
        np.array([-5.1043536]),
        np.array([-2.899967805418783, 4.040399359702244]),
        np.array([-32.64449927841361, -99.35311008728094, 49.99119122405989]),
        np.array(
            [
                -76.46023087151691,
                -278.5942120829058,
                153.9294840910643,
                10.97101866258358,
            ]
        ),
        np.array(
            [
                -76.29701586804983,
                -294.2795630511232,
                162.0029695867566,
                23.65166903095270,
                -7.652977706771382,
            ]
        ),
    ),
    α=np.array([0.0, 0.3507221, 0.2557041, 0.681779, 1.0, 1.0]),
    γ=np.array([0.25, -0.0690221, -0.0009672, -0.087979, 0.0, 0.0]),
    m_sol=np.array(
        [
            22.73885722420363,
            67.38147284535289,
            -31.21877493038560,
            0.7285641833203814,
            1.0,
            0.0,
        ]
    ),
    m_error=np.array(
        [
            22.73885722420363,
            67.38147284535289,
            -31.21877493038560,
            0.7285641833203814,
            1.0,
            1.0,
        ]
    ),
)


class Rodas42(AbstractRosenbrock):
    r"""Rodas42 method.

    4th order Rosenbrock method for solving stiff equations. Uses third-order Hermite
    polynomial interpolation for dense output.

    ??? cite "Reference"
       ```bibtex
       @book{book,
         author = {Hairer, Ernst and Wanner, Gerhard},
         year = {1996},
         month = {01},
         pages = {},
         title = {Solving Ordinary Differential Equations II. Stiff and Differential-Algebraic Problems},
         volume = {14},
         journal = {Springer Verlag Series in Comput. Math.},
         doi = {10.1007/978-3-662-09947-6}
         }
        ```
    """

    tableau: ClassVar[RosenbrockTableau] = _tableau

    def order(self, terms):
        del terms
        return 4


Rodas42.__init__.__doc__ = """**Arguments:** None"""
