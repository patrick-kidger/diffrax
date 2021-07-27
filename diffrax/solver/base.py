import abc
from typing import Tuple, TypeVar

from ..autojit import autojit
from ..custom_types import Array, PyTree, Scalar, SquashTreeDef
from ..tree import tree_dataclass


T = TypeVar('T', bound=PyTree)


@tree_dataclass
class AbstractSolver(metaclass=abc.ABCMeta):
    # Subclasses must define:
    # (cannot put them here because of default-before-non-default problems with dataclasses)
    # recommended_interpolation: Type[AbstractInterpolation]
    # jit: bool

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.jit_step = autojit(cls.step)

    def init(
        self, y_treedef: SquashTreeDef, t0: Scalar, t1: Scalar, y0: Array["state"], args: PyTree  # noqa: F821
    ) -> T:  # noqa: F821
        return None

    @abc.abstractmethod
    def step(
        self,
        y_treedef: SquashTreeDef,
        t0: Scalar,
        t1: Scalar,
        y0: Array["state"],  # noqa: F821
        args: PyTree,
        solver_state: T
    ) -> Tuple[Array["state"], T]:  # noqa: F821
        pass

    def step_maybe_jit(self, *args, **kwargs):
        if self.jit:
            return self.jit_step(*args, **kwargs)
        else:
            return self.step(*args, **kwargs)
