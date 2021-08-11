import abc
from typing import Callable, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp

from .custom_types import Array, PyTree, Scalar, SquashTreeDef
from .misc import tree_squash, tree_unsquash
from .path import AbstractPath


def _prod(
    vf: Array["state":..., "control":...], control: Array["control":...]  # noqa: F821
) -> Array["state":...]:  # noqa: F821
    return jnp.tensordot(vf, control, axes=control.ndim)


class AbstractTerm(eqx.Module):
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

    def vf_(
        self,
        t: Scalar,
        y_: Array["state"],  # noqa: F821
        args: PyTree,
        y_treedef: SquashTreeDef,
    ) -> Tuple[Array["state*control"], SquashTreeDef]:  # noqa: F821
        y = tree_unsquash(y_treedef, y_)
        vf = self.vector_field(t, y, args)
        return tree_squash(vf)

    def contr_(
        self, t0: Scalar, t1: Scalar
    ) -> Tuple[Array["control"], SquashTreeDef]:  # noqa: F821
        return tree_squash(self.contr(t0, t1))

    def prod_(
        self,
        vf_: Array["state*control"],  # noqa: F821
        control_: Array["control"],  # noqa: F821
        vf_treedef: SquashTreeDef,
        control_treedef: SquashTreeDef,
    ) -> Array["state"]:  # noqa: F821
        vf = tree_unsquash(vf_treedef, vf_)
        control = tree_unsquash(control_treedef, control_)
        prod = self.prod(vf, control)
        prod_, _ = tree_squash(prod)
        return prod_

    def vf_prod_(
        self,
        t: Scalar,
        y_: Array["state"],  # noqa: F821
        args: PyTree,
        control_: Array["control"],  # noqa: F821
        y_treedef: SquashTreeDef,
        control_treedef: SquashTreeDef,
    ) -> Array["state"]:  # noqa: F821
        y = tree_unsquash(y_treedef, y_)
        vf = self.vector_field(t, y, args)
        control = tree_unsquash(control_treedef, control_)
        prod = self.prod(vf, control)
        prod_, _ = tree_squash(prod)
        return prod_

    # This is a pinhole break in our vector-field/control abstraction.
    # Everywhere else we get to evaluate over some interval, which allows us to
    # evaluate our control over that interval. However to select the initial point in
    # an adapative step size scheme, the standard heuristic is to start by making
    # evaluations at just the initial point -- no intervals involved.
    def func_for_init(
        self,
        t: Scalar,
        y_: Array["state"],  # noqa: F821
        args: PyTree,
        y_treedef: SquashTreeDef,
    ) -> Array["state"]:  # noqa: F821
        raise ValueError(
            "An initial step size cannot be selected automatically. The most common "
            "scenario for this error to occur is when trying to use adaptive step "
            "size solvers with SDEs. Please specify an initial `dt0` instead."
        )


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

    def func_for_init(
        self,
        t: Scalar,
        y_: Array["state"],  # noqa: F821
        args: PyTree,
        y_treedef: SquashTreeDef,
    ) -> Array["state"]:  # noqa: F821
        y = tree_unsquash(y_treedef, y_)
        vf = self.vf(t, y, args)
        vf, _ = tree_squash(vf)
        return vf


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


class _ControlToODE(eqx.Module):
    control_term: ControlTerm

    def __call__(self, t: Scalar, y: PyTree, args: PyTree) -> PyTree:
        control = self.control_term.control.derivative(t)
        return self.control_term.vf_prod(t, y, args, control)
