import abc
import jax
import jax.numpy as jnp
from typing import Callable

from .custom_types import Array, PyTree, Scalar, SquashTreeDef
from .path import AbstractPath
from .tree import tree_squash, tree_unsquash


def _prod(vf: Array["state":...,  # noqa: F821
                    "control":...],  # noqa: F821
          control: Array["control":...]) -> Array["state":...]:  # noqa: F821
    return jnp.tensordot(vf, control, axes=control.ndim)


class AbstractTerm(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def vector_field(self, t: Scalar, y: PyTree) -> PyTree:
        pass

    @abc.abstractmethod
    def diff_control(self, t: Scalar) -> PyTree:
        pass

    @abc.abstractmethod
    def eval_control(self, t0: Scalar, t1: Scalar) -> PyTree:
        pass

    def prod(self, vf: PyTree, control: PyTree) -> PyTree:
        return jax.tree_map(_prod, vf, control)

    def vector_field_prod(self, t: Scalar, y: PyTree, control: PyTree) -> PyTree:
        return self.prod(self.vector_field(t, y), control)

    def vector_field_(self, y_treedef: SquashTreeDef, t: Scalar,
                      y_: Array["state"]) -> Array["state*control"]:  # noqa: F821
        y = tree_unsquash(y_treedef, y_)
        vf = self.vector_field(t, y)
        vf_, _ = tree_squash(vf)
        return vf_

    def diff_control_(self, t: Scalar) -> Array["control"]:  # noqa: F821
        control = self.diff_control(t)
        control_, _ = tree_squash(control)
        return control_

    def eval_control_(self, t0: Scalar, t1: Scalar) -> Array["control"]:  # noqa: F821
        control = self.eval_control(t0, t1)
        control_, _ = tree_squash(control)
        return control_

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

    def vector_field_prod_(
        self,
        y_treedef: SquashTreeDef,
        control_treedef: SquashTreeDef,
        t: Scalar,
        y_: Array["state"],  # noqa: F821
        control_: Array["control"]  # noqa: F821
    ) -> Array["state"]:  # noqa: F821
        y = tree_unsquash(y_treedef, y_)
        vf = self.vector_field(t, y)
        control = tree_unsquash(control_treedef, control_)
        prod = self.prod(vf, control)
        prod_, _ = tree_squash(prod)
        return prod_


class ControlTerm(AbstractTerm):
    def __init__(self, *, vector_field: Callable[[Scalar, PyTree], PyTree], control: AbstractPath, **kwargs):
        super().__init__(**kwargs)
        self.vector_field = vector_field
        self.control = control

    # To avoid abstractmethod errors
    def vector_field(self, t: Scalar, y: PyTree) -> PyTree:
        pass

    def diff_control(self, t: Scalar) -> PyTree:
        return self.control.derivative(t)

    def eval_control(self, t0: Scalar, t1: Scalar) -> PyTree:
        return self.control.evaluate(t0, t1)


class ODETerm(AbstractTerm):
    def __init__(self, *, vector_field: Callable[[Scalar, PyTree], PyTree], **kwargs):
        super().__init__(**kwargs)
        self.vector_field = vector_field

    # To avoid abstractmethod errors
    def vector_field(self, t: Scalar, y: PyTree) -> PyTree:
        pass

    def diff_control(self, t: Scalar) -> Scalar:
        return 1

    def eval_control(self, t0: Scalar, t1: Scalar) -> Scalar:
        return t1 - t0

    def prod(self, vf: PyTree, control: Scalar) -> PyTree:
        # control assumed to be trivial
        return jax.tree_map(lambda v: control * v, vf)
