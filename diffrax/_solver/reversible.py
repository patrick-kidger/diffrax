from collections.abc import Callable
from typing import cast, Optional, TypeAlias, TypeVar

from equinox.internal import ω
from jaxtyping import PyTree

from .._custom_types import Args, BoolScalarLike, DenseInfo, RealScalarLike, VF, Y
from .._solution import RESULTS, update_result
from .._solver.base import (
    AbstractReversibleSolver,
    AbstractWrappedSolver,
)
from .._term import AbstractTerm
from .runge_kutta import AbstractERK


ω = cast(Callable, ω)

_BaseSolverState = TypeVar("_BaseSolverState")
_SolverState: TypeAlias = tuple[_BaseSolverState, Y]


class Reversible(
    AbstractReversibleSolver[_SolverState], AbstractWrappedSolver[_SolverState]
):
    """
    Reversible solver method.

    Allows any explicit Runge-Kutta solver ([`diffrax.AbstractERK`][]) to be made
    algebraically reversible.

    **Arguments:**

    - `solver`: base solver to be made reversible
    - `coupling_parameter`: determines coupling between the two evolving solutions.
    Must be within the range `0 < coupling_parameter < 1`. Unless you need finer control
    over stability, the default value of `0.999` should be sufficient.

    !!! note

        When solving SDEs, the base `solver` must converge to the Statonovich solution.

    ??? cite "References"

        This method was developed in:

        ```bibtex
        @article{mccallum2024efficient,
            title={Efficient, Accurate and Stable Gradients for Neural ODEs},
            author={McCallum, Sam and Foster, James},
            journal={arXiv preprint arXiv:2410.11648},
            year={2024}
        }
        ```

        And built on previous work by:

        ```bibtex
        @article{kidger2021efficient,
            title={Efficient and accurate gradients for neural sdes},
            author={Kidger, Patrick and Foster, James and Li, Xuechen Chen and Lyons,
                    Terry},
            journal={Advances in Neural Information Processing Systems},
            volume={34},
            pages={18747--18761},
            year={2021}
        }

        @article{zhuang2021mali,
            title={Mali: A memory efficient and reverse accurate integrator for neural
                    odes},
            author={Zhuang, Juntang and Dvornek, Nicha C and Tatikonda, Sekhar and
            Duncan, James S},
            journal={arXiv preprint arXiv:2102.04668},
            year={2021}
        }
        ```
    """

    solver: AbstractERK
    coupling_parameter: float = 0.999

    @property
    def interpolation_cls(self):  # pyright: ignore
        return self.solver.interpolation_cls

    @property
    def term_structure(self):
        return self.solver.term_structure

    @property
    def term_compatible_contr_kwargs(self):
        return self.solver.term_compatible_contr_kwargs

    @property
    def root_finder(self):
        return self.solver.root_finder  # pyright: ignore

    @property
    def root_find_max_steps(self):
        return self.solver.root_find_max_steps  # pyright: ignore

    def order(self, terms: PyTree[AbstractTerm]) -> Optional[int]:
        return self.solver.order(terms)

    def strong_order(self, terms: PyTree[AbstractTerm]) -> Optional[RealScalarLike]:
        return self.solver.strong_order(terms)

    def init(
        self,
        terms: PyTree[AbstractTerm],
        t0: RealScalarLike,
        t1: RealScalarLike,
        y0: Y,
        args: Args,
    ) -> _SolverState:
        if not isinstance(self.solver, AbstractERK):
            raise ValueError(
                "`Reversible` is only compatible with `AbstractERK` base solvers."
            )
        original_solver_init = self.solver.init(terms, t0, t1, y0, args)
        return (original_solver_init, y0)

    def step(
        self,
        terms: PyTree[AbstractTerm],
        t0: RealScalarLike,
        t1: RealScalarLike,
        y0: Y,
        args: Args,
        solver_state: _SolverState,
        made_jump: BoolScalarLike,
    ) -> tuple[Y, Optional[Y], DenseInfo, _SolverState, RESULTS]:
        original_solver_state, z0 = solver_state

        step_z0, _, dense_info, original_solver_state, result1 = self.solver.step(
            terms, t0, t1, z0, args, original_solver_state, True
        )
        y1 = (self.coupling_parameter * (ω(y0) - ω(z0)) + ω(step_z0)).ω

        step_y1, y_error, _, _, result2 = self.solver.step(
            terms, t1, t0, y1, args, original_solver_state, True
        )
        z1 = (ω(y1) + ω(z0) - ω(step_y1)).ω

        solver_state = (original_solver_state, z1)
        result = update_result(result1, result2)

        return y1, y_error, dense_info, solver_state, result

    def backward_step(
        self,
        terms: PyTree[AbstractTerm],
        t0: RealScalarLike,
        t1: RealScalarLike,
        y1: Y,
        args: Args,
        solver_state: _SolverState,
        made_jump: BoolScalarLike,
    ) -> tuple[Y, DenseInfo, _SolverState, RESULTS]:
        original_solver_state, z1 = solver_state
        step_y1, _, _, original_solver_state, result1 = self.solver.step(
            terms, t1, t0, y1, args, original_solver_state, True
        )
        z0 = (ω(z1) - ω(y1) + ω(step_y1)).ω
        step_z0, _, dense_info, _, result2 = self.solver.step(
            terms, t0, t1, z0, args, original_solver_state, True
        )
        y0 = ((1 / self.coupling_parameter) * (ω(y1) - ω(step_z0)) + ω(z0)).ω

        solver_state = (original_solver_state, z0)
        result = update_result(result1, result2)

        return y0, dense_info, solver_state, result

    def func(
        self, terms: PyTree[AbstractTerm], t0: RealScalarLike, y0: Y, args: Args
    ) -> VF:
        return self.solver.func(terms, t0, y0, args)
