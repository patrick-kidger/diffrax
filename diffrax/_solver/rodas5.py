from typing import ClassVar

import numpy as np

from .rosenbrock import AbstractRosenbrock, RosenbrockTableau


_tableau = RosenbrockTableau(
    a_lower=(
        np.array([2.0]),
        np.array([3.040894194418781, 1.041747909077569]),
        np.array([2.576417536461461, 1.622083060776640, -0.9089668560264532]),
        np.array(
            [
                2.760842080225597,
                1.446624659844071,
                -0.3036980084553738,
                0.2877498600325443,
            ]
        ),
        np.array(
            [
                -14.09640773051259,
                6.925207756232704,
                -41.47510893210728,
                2.343771018586405,
                24.13215229196062,
            ]
        ),
        np.array(
            [
                -14.09640773051259,
                6.925207756232704,
                -41.47510893210728,
                2.343771018586405,
                24.13215229196062,
                1.0,
            ]
        ),
        np.array(
            [
                -14.09640773051259,
                6.925207756232704,
                -41.47510893210728,
                2.343771018586405,
                24.13215229196062,
                1.0,
                1.0,
            ]
        ),
    ),
    c_lower=(
        np.array([-10.31323885133993]),
        np.array([-21.04823117650003, -7.234992135176716]),
        np.array([32.22751541853323, -4.943732386540191, 19.44922031041879]),
        np.array(
            [
                -20.69865579590063,
                -8.816374604402768,
                1.260436877740897,
                -0.7495647613787146,
            ]
        ),
        np.array(
            [
                -46.22004352711257,
                -17.49534862857472,
                -289.6389582892057,
                93.60855400400906,
                318.3822534212147,
            ]
        ),
        np.array(
            [
                34.20013733472935,
                -14.15535402717690,
                57.82335640988400,
                25.83362985412365,
                1.408950972071624,
                -6.551835421242162,
            ]
        ),
        np.array(
            [
                42.57076742291101,
                -13.80770672017997,
                93.98938432427124,
                18.77919633714503,
                -31.58359187223370,
                -6.685968952921985,
                -5.810979938412932,
            ]
        ),
    ),
    α=np.array(
        [
            0.0,
            0.38,
            0.3878509998321533,
            0.4839718937873840,
            0.4570477008819580,
            1.0,
            1.0,
            1.0,
        ]
    ),
    γ=np.array(
        [
            0.19,
            -0.1823079225333714636,
            -0.319231832186874912,
            0.3449828624725343,
            -0.377417564392089818,
            0.0,
            0.0,
            0.0,
        ]
    ),
    m_sol=np.array(
        [
            -14.09640773051259,
            6.925207756232704,
            -41.47510893210728,
            2.343771018586405,
            24.13215229196062,
            1.0,
            1.0,
            0.0,
        ]
    ),
    m_error=np.array(
        [
            -14.09640773051259,
            6.925207756232704,
            -41.47510893210728,
            2.343771018586405,
            24.13215229196062,
            1.0,
            1.0,
            1.0,
        ]
    ),
)


class Rodas5(AbstractRosenbrock):
    r"""Rodas5 method.

    5th order Rosenbrock method for solving stiff equations. Uses third-order Hermite
    polynomial interpolation for dense output.

    ??? cite "Reference"

        @mastersthesis{DiMarzo1993Rodas54,
          author       = {Di Marzo, Giovanna A.},
          title        = {RODAS5(4) -- M{\'e}thodes de {R}osenbrock d'ordre 5(4) adapt{\'e}es aux probl{\`e}mes diff{\'e}rentiels-alg{\'e}briques},
          school       = {Faculty of Science, University of Geneva},
          address      = {Geneva, Switzerland},
          year         = {1993},
          type         = {MSc Mathematics thesis},
          url          = {https://cui.unige.ch/~dimarzo/papers/DIPL93.pdf},
        }

    """

    tableau: ClassVar[RosenbrockTableau] = _tableau

    def order(self, terms):
        del terms
        return 5


Rodas5.__init__.__doc__ = """**Arguments:** None"""
