from typing import ClassVar

import numpy as np

from .rosenbrock import AbstractRosenbrock, RosenbrockTableau


_tableau = RosenbrockTableau(
    a_lower=(
        np.array([1.544]),
        np.array([0.9466785280815826, 0.2557011698983284]),
        np.array([3.314825187068521, 2.896124015972201, 0.9986419139977817]),
        np.array(
            [
                1.221224509226641,
                6.019134481288629,
                12.53708332932087,
                -0.6878860361058950,
            ]
        ),
        np.array(
            [
                1.221224509226641,
                6.019134481288629,
                12.53708332932087,
                -0.6878860361058950,
                1,
            ]
        ),
    ),
    c_lower=(
        np.array([-5.6688]),
        np.array([-2.430093356833875, -0.2063599157091915]),
        np.array([-0.1073529058151375, -9.594562251023355, -20.47028614809616]),
        np.array(
            [
                7.496443313967647,
                -10.24680431464352,
                -33.99990352819905,
                11.70890893206160,
            ]
        ),
        np.array(
            [
                8.083246795921522,
                -7.981132988064893,
                -31.52159432874371,
                16.31930543123136,
                -6.058818238834054,
            ]
        ),
    ),
    α=np.array([0, 0.386, 0.21, 0.63, 1, 1]),
    γ=np.array([0.25, -0.1043, 0.1035, -0.0362, 0, 0]),
    m_sol=np.array(
        [
            1.221224509226641,
            6.019134481288629,
            12.53708332932087,
            -0.6878860361058950,
            1,
            1,
        ]
    ),
    m_error=np.array(
        [
            1.221224509226641,
            6.019134481288629,
            12.53708332932087,
            -0.6878860361058950,
            1,
            1,
        ]
    ),
)


class Rodas4(AbstractRosenbrock):
    r"""Rodas4 method.

    4rd order Rosenbrock method for solving stiff equation. Uses third-order Hermite
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


Rodas4.__init__.__doc__ = """**Arguments:** None"""
