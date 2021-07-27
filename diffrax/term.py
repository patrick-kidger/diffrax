import abc
import jax
import jax.numpy as jnp
from typing import Callable, Tuple

from .custom_types import Array, PyTree, Scalar, SquashTreeDef
from .path import AbstractPath
from .tree import tree_dataclass, tree_squash, tree_unsquash


def _prod(vf: Array["state":...,  # noqa: F821
                    "control":...],  # noqa: F821
          control: Array["control":...]) -> Array["state":...]:  # noqa: F821
    return jnp.tensordot(vf, control, axes=control.ndim)


@tree_dataclass
class AbstractTerm(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def vf(self, t: Scalar, y: PyTree, args: PyTree) -> PyTree:
        pass

    @abc.abstractmethod
    def contr(self, t0: Scalar, t1: Scalar) -> PyTree:
        pass

    @abc.abstractmethod
    def prod(self, vf: PyTree, control: PyTree) -> PyTree:
        pass

    def vf_prod(self, t: Scalar, y: PyTree, args: PyTree, control: PyTree) -> PyTree:
        return self.prod(self.vector_field(t, y, args), control)

    def vf_(self, y_treedef: SquashTreeDef, t: Scalar, y_: Array["state"],  # noqa: F821
            args: PyTree) -> Tuple[Array["state*control"], SquashTreeDef]:  # noqa: F821
        y = tree_unsquash(y_treedef, y_)
        vf = self.vector_field(t, y, args)
        return tree_squash(vf)

    def contr_(self, t0: Scalar, t1: Scalar) -> Tuple[Array["control"], SquashTreeDef]:  # noqa: F821
        return tree_squash(self.contr(t0, t1))

    def prod_(
        self,
        vf_treedef: SquashTreeDef,
        control_treedef: SquashTreeDef,
        vf_: Array["state*control"],  # noqa: F821
        control_: Array["control"]  # noqa: F821
    ) -> Array["state"]:  # noqa: F821
        vf = tree_unsquash(vf_treedef, vf_)
        control = tree_unsquash(control_treedef, control_)
        prod = self.prod(vf, control)
        prod_, _ = tree_squash(prod)
        return prod_

    def vf_prod_(
        self,
        y_treedef: SquashTreeDef,
        control_treedef: SquashTreeDef,
        t: Scalar,
        y_: Array["state"],  # noqa: F821
        args: PyTree,
        control_: Array["control"]  # noqa: F821
    ) -> Array["state"]:  # noqa: F821
        y = tree_unsquash(y_treedef, y_)
        vf = self.vector_field(t, y, args)
        control = tree_unsquash(control_treedef, control_)
        prod = self.prod(vf, control)
        prod_, _ = tree_squash(prod)
        return prod_

    # This exists just to get out of an annoying catch-22, and shouldn't be used much in general.
    def func_for_init(self, y_treedef: SquashTreeDef, t: Scalar, y_: Array["state"],  # noqa: F821
                      args: PyTree) -> Array["state"]:  # noqa: F821
        raise ValueError(f"func_for_init does not exist for term of type {type(self)}")


@tree_dataclass
class ODETerm(AbstractTerm):
    vector_field: Callable[[Scalar, PyTree, PyTree], PyTree]

    def vf(self, t: Scalar, y: PyTree, args: PyTree) -> PyTree:
        return self.vector_field(t, y, args)

    @staticmethod
    def contr(t0: Scalar, t1: Scalar) -> Scalar:
        return t1 - t0

    @staticmethod
    def prod(vf: PyTree, control: Scalar) -> PyTree:
        return jax.tree_map(lambda v: control * v, vf)

    def func_for_init(self, y_treedef: SquashTreeDef, t: Scalar, y_: Array["state"],  # noqa: F821
                      args: PyTree) -> Array["state"]:  # noqa: F821
        y = tree_unsquash(y_treedef, y_)
        vf = self.vf(t, y, args)
        vf, _ = tree_squash(vf)
        return vf


@tree_dataclass
class ControlTerm(AbstractTerm):
    vector_field: Callable[[Scalar, PyTree, PyTree], PyTree]
    control: AbstractPath

    def vf(self, t: Scalar, y: PyTree, args: PyTree) -> PyTree:
        return self.vector_field(t, y, args)

    def contr(self, t0: Scalar, t1: Scalar) -> PyTree:
        return self.control.evaluate(t0, t1)

    @staticmethod
    def prod(vf: PyTree, control: PyTree) -> PyTree:
        return jax.tree_map(_prod, vf, control)

    def to_ode(self):
        vector_field = _ControlToODE(self)
        return ODETerm(vector_field=vector_field)


@tree_dataclass
class _ControlToODE:
    control_term: ControlTerm

    def __call__(self, t: Scalar, y: PyTree, args: PyTree) -> PyTree:
        control = self.control_term.control.derivative(t)
        return self.control_term.vf_prod(t, y, args, control)
