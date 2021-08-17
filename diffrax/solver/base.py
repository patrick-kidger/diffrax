import abc
from typing import Optional, Tuple, Type, TypeVar

import equinox as eqx

from ..custom_types import Array, DenseInfo, PyTree, Scalar
from ..local_interpolation import AbstractLocalInterpolation


_SolverState = TypeVar("_SolverState", bound=Optional[PyTree])


class AbstractSolver(eqx.Module):
    @property
    @abc.abstractmethod
    def interpolation_cls(self) -> Type[AbstractLocalInterpolation]:
        pass

    @property
    @abc.abstractmethod
    def order(self) -> int:
        pass

    @abc.abstractmethod
    def wrap(self, t0: Scalar, y0: PyTree, args: PyTree, direction: Scalar):
        pass

    def init(
        self,
        t0: Scalar,
        t1: Scalar,
        y0: Array["state"],  # noqa: F821
        args: PyTree,
    ) -> _SolverState:
        return None

    @abc.abstractmethod
    def step(
        self,
        t0: Scalar,
        t1: Scalar,
        y0: Array["state"],  # noqa: F821
        args: PyTree,
        solver_state: _SolverState,
        made_jump: Array[(), bool],
    ) -> Tuple[
        Array["state"], Optional[Array["state"]], DenseInfo, _SolverState  # noqa: F821
    ]:
        pass

    def func_for_init(
        self,
        t0: Scalar,
        y0: Array["state"],  # noqa: F821
        args: PyTree,
    ) -> Array["state"]:  # noqa: F821
        raise ValueError(
            "An initial step size cannot be selected automatically. The most common "
            "scenario for this error to occur is when trying to use adaptive step "
            "size solvers with SDEs. Please specify an initial `dt0` instead."
        )
