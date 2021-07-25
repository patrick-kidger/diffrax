import abc
import functools as ft
import jax
from typing import Tuple, TypeVar

from ..custom_types import Array, PyTree, Scalar
from ..interpolation import AbstractInterpolation


T = TypeVar('T', bound=PyTree)
T2_co = TypeVar('T2_co', bound=PyTree, covariant=True)


class AbstractSolver(metaclass=abc.ABCMeta):
    recommended_interpolation: AbstractInterpolation

    @abc.abstractmethod
    def init(self, t0: Scalar, y0: Array["state": ...]) -> T:
        pass

    @abc.abstractmethod
    def step(self, t0: Scalar, t1: Scalar, y0: Array["state": ...], solver_state: T) -> Tuple[Array["state": ...], T]:
        pass


@ft.partial(jax.jit, static_argnums=0)
def _splitting_method_step(solvers: list[AbstractSolver], t0: Scalar, t1: Scalar, y0: Array["state": ...], solver_state: list[T2_co]) -> Tuple[Array["state": ...], list[T2_co]]:
    new_solver_state = []
    for solver, solver_state_i in zip(solvers, solver_state):
        y0, solver_state_i = solver.step(t0, t1, y0, solver_state_i)
        new_solver_state.append(solver_state_i)
    return y0, new_solver_state



class SplittingMethod(AbstractSolver):
    def __init__(self, *, solvers: list[AbstractSolver], **kwargs):
        assert len(solvers) > 0
        super().__init__(**kwargs)
        self.solvers = solvers
        self.recommended_interpolation = self.solvers[0].recommended_interpolation

    def init(self, t0: Scalar, y0: Array["state": ...]) -> list[T2_co]:
        return [solver.init(t0, y0) for solver in self.solvers]

    def step(self, t0: Scalar, t1: Scalar, y0: Array["state": ...], solver_state: list[T2_co]) -> Tuple[Array["state": ...], list[T2_co]]:
        return _splitting_method_step(self.solvers, t0, t1, solver_state)

