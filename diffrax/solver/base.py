import abc
import equinox as eqx
from typing import Optional, Tuple, Type, TypeVar

from ..custom_types import Array, DenseInfo, PyTree, Scalar, SquashTreeDef
from ..local_interpolation import AbstractLocalInterpolation


_SolverState = TypeVar('_SolverState', bound=Optional[PyTree])


class AbstractSolver(eqx.Module):
    @property
    @abc.abstractmethod
    def order(self) -> int:
        pass

    @property
    @abc.abstractmethod
    def interpolation_cls(self) -> Type[AbstractLocalInterpolation]:
        pass

    def init(
        self, t0: Scalar, t1: Scalar, y0: Array["state"],  # noqa: F821
        args: PyTree, y_treedef: SquashTreeDef,
    ) -> _SolverState:
        return None

    @abc.abstractmethod
    def step(
        self,
        t0: Scalar,
        t1: Scalar,
        y0: Array["state"],  # noqa: F821
        args: PyTree,
        y_treedef: SquashTreeDef,
        solver_state: _SolverState,
    ) -> Tuple[Array["state"], Optional[Array["state"]], DenseInfo, _SolverState]:  # noqa: F821
        pass

    def func_for_init(self, t: Scalar, y_: Array["state"], args: PyTree,  # noqa: F821
                      y_treedef: SquashTreeDef) -> Array["state"]:  # noqa: F821
        raise ValueError("func_for_init does not exist.")
