import abc
from dataclasses import dataclass
import jax
import jax.numpy as jnp
from typing import Callable, Tuple

from .custom_types import Array, PyTree, Scalar, SquashTreeDef
from .misc import stable_method_hash
from .path import AbstractPath
from .tree import tree_squash, tree_unsquash


def _prod(vf: Array["state":...,  # noqa: F821
                    "control":...],  # noqa: F821
          control: Array["control":...]) -> Array["state":...]:  # noqa: F821
    return jnp.tensordot(vf, control, axes=control.ndim)


class AbstractTerm(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def vector_field(self, t: Scalar, y: PyTree, args: PyTree) -> PyTree:
        pass

    @abc.abstractmethod
    def diff_control(self, t: Scalar) -> PyTree:
        pass

    @abc.abstractmethod
    def eval_control(self, t0: Scalar, t1: Scalar) -> PyTree:
        pass

    @abc.abstractmethod
    def prod(self, vf: PyTree, control: PyTree) -> PyTree:
        pass

    @stable_method_hash
    def vector_field_prod(self, t: Scalar, y: PyTree, args: PyTree, control: PyTree) -> PyTree:
        return self.prod(self.vector_field(t, y, args), control)

    @stable_method_hash
    def vector_field_(self, y_treedef: SquashTreeDef, t: Scalar, y_: Array["state"],  # noqa: F821
                      args: PyTree) -> Tuple[Array["state*control"], SquashTreeDef]:  # noqa: F821
        y = tree_unsquash(y_treedef, y_)
        vf = self.vector_field(t, y, args)
        return tree_squash(vf)

    @stable_method_hash
    def diff_control_(self, t: Scalar) -> Tuple[Array["control"], SquashTreeDef]:  # noqa: F821
        control = self.diff_control(t)
        return tree_squash(control)

    @stable_method_hash
    def eval_control_(self, t0: Scalar, t1: Scalar) -> Tuple[Array["control"], SquashTreeDef]:  # noqa: F821
        control = self.eval_control(t0, t1)
        return tree_squash(control)

    @stable_method_hash
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

    @stable_method_hash
    def vector_field_prod_(
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


@dataclass(frozen=True)
class ControlTerm(AbstractTerm):
    vector_field: Callable[[Scalar, PyTree, PyTree], PyTree]
    control: AbstractPath

    # To avoid abstractmethod errors
    def vector_field(self, t: Scalar, y: PyTree, args: PyTree) -> PyTree:
        pass

    @stable_method_hash
    def diff_control(self, t: Scalar) -> PyTree:
        return self.control.derivative(t)

    @stable_method_hash
    def eval_control(self, t0: Scalar, t1: Scalar) -> PyTree:
        return self.control.evaluate(t0, t1)

    @stable_method_hash
    def prod(self, vf: PyTree, control: PyTree) -> PyTree:
        return jax.tree_map(_prod, vf, control)


@dataclass(frozen=True)
class ODETerm(AbstractTerm):
    vector_field: Callable[[Scalar, PyTree, PyTree], PyTree]

    # To avoid abstractmethod errors
    def vector_field(self, t: Scalar, y: PyTree, args: PyTree) -> PyTree:
        pass

    @stable_method_hash
    def diff_control(self, t: Scalar) -> Scalar:
        return 1

    @stable_method_hash
    def eval_control(self, t0: Scalar, t1: Scalar) -> Scalar:
        return t1 - t0

    @stable_method_hash
    def prod(self, vf: PyTree, control: Scalar) -> PyTree:
        return jax.tree_map(lambda v: control * v, vf)
