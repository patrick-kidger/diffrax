from typing import Tuple

import jax

from ..custom_types import Bool, DenseInfo, PyTree, Scalar
from ..local_interpolation import LocalLinearInterpolation
from ..misc import ω
from ..solution import RESULTS
from ..term import AbstractTerm
from .base import AbstractSolver


_ErrorEstimate = None
_SolverState = Tuple[Scalar, PyTree]


# TODO: support arbitrary linear multistep methods
class LeapfrogMidpoint(AbstractSolver):
    r"""Leapfrog/midpoint method.

    2nd order linear multistep method.

    Note that this is referred to as the "leapfrog/midpoint method" as this is the name
    used by Shampine in the reference below. It should not be confused with any of the
    many other "leapfrog methods" (there are several), or with the "midpoint method"
    (which is usually taken to refer to the explicit Runge--Kutta method
    [`diffrax.Midpoint`][]).

    ??? cite "Reference"

        ```bibtex
        @article{shampine2009stability,
            title={Stability of the leapfrog/midpoint method},
            author={L. F. Shampine},
            journal={Applied Mathematics and Computation},
            volume={208},
            number={1},
            pages={293-298},
            year={2009},
        }
        ```
    """

    term_structure = jax.tree_structure(0)
    interpolation_cls = LocalLinearInterpolation

    def order(self, terms):
        return 2

    def init(
        self, terms: AbstractTerm, t0: Scalar, t1: Scalar, y0: PyTree, args: PyTree
    ) -> _SolverState:
        del terms, t1, args
        # Corresponds to making an explicit Euler step on the first step.
        return t0, y0

    def step(
        self,
        terms: AbstractTerm,
        t0: Scalar,
        t1: Scalar,
        y0: PyTree,
        args: PyTree,
        solver_state: _SolverState,
        made_jump: Bool,
    ) -> Tuple[PyTree, _ErrorEstimate, DenseInfo, _SolverState, RESULTS]:
        del made_jump
        tm1, ym1 = solver_state
        control = terms.contr(tm1, t1)
        y1 = (ym1**ω + terms.vf_prod(t0, y0, args, control) ** ω).ω
        dense_info = dict(y0=y0, y1=y1)
        solver_state = (t0, y0)
        return y1, None, dense_info, solver_state, RESULTS.successful

    def func(
        self,
        terms: AbstractTerm,
        t0: Scalar,
        y0: PyTree,
        args: PyTree,
    ) -> PyTree:
        return terms.vf(t0, y0, args)
