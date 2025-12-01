from typing import ClassVar

import numpy as np

from .rosenbrock import AbstractRosenbrock, RosenbrockTableau


_tableau = RosenbrockTableau(
    m_sol=np.array([2.0, 0.5773502691896258, 0.4226497308103742]),
    m_error=np.array([2.113248654051871, 1.0, 0.4226497308103742]),
    a_lower=(
        np.array([1.267949192431123]),
        np.array([1.267949192431123, 0.0]),
    ),
    c_lower=(
        np.array([-1.607695154586736]),
        np.array([-3.464101615137755, -1.732050807568877]),
    ),
    α=np.array([0.0, 1.0, 1.0]),
    γ=np.array(
        [
            0.7886751345948129,
            -0.2113248654051871,
            -1.0773502691896260,
        ]
    ),
)


class Ros3p(AbstractRosenbrock):
    r"""Ros3p method.

    3rd order Rosenbrock method for solving stiff equation. Uses third-order Hermite
    polynomial interpolation for dense output.

    ??? cite "Reference"

        ```bibtex
        @article{LangVerwer2001ROS3P,
          author    = {Lang, J. and Verwer, J.},
          title     = {ROS3P---An Accurate Third-Order Rosenbrock Solver Designed
                       for Parabolic Problems},
          journal   = {BIT Numerical Mathematics},
          volume    = {41},
          number    = {4},
          pages     = {731--738},
          year      = {2001},
          doi       = {10.1023/A:1021900219772}
         }
         ```
    """

    tableau: ClassVar[RosenbrockTableau] = _tableau

    def order(self, terms):
        del terms
        return 3


Ros3p.__init__.__doc__ = """**Arguments:** None"""
