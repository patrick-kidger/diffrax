import abc
from typing import Callable, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree

from .custom_types import Array, PyTree, Scalar
from .path import AbstractPath


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

    # This is a pinhole break in our vector-field/control abstraction.
    # Everywhere else we get to evaluate over some interval, which allows us to
    # evaluate our control over that interval. However to select the initial point in
    # an adapative step size scheme, the standard heuristic is to start by making
    # evaluations at just the initial point -- no intervals involved.
    def func_for_init(self, t: Scalar, y: PyTree, args: PyTree) -> PyTree:
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

    func_for_init = vf


def _prod(
    vf: Array["state":..., "control":...], control: Array["control":...]  # noqa: F821
) -> Array["state":...]:  # noqa: F821
    return jnp.tensordot(vf, control, axes=control.ndim)


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


class MultiTerm(AbstractTerm):
    terms: Tuple[AbstractTerm, ...]

    def vf(self, t: Scalar, y: PyTree, args: PyTree) -> Tuple[PyTree, ...]:
        return tuple(term.vf(t, y, args) for term in self.terms)

    def contr(self, t0: Scalar, t1: Scalar) -> Tuple[PyTree, ...]:
        return tuple(term.contr(t0, t1) for term in self.terms)

    def prod(self, vf: Tuple[PyTree, ...], control: Tuple[PyTree, ...]) -> PyTree:
        return sum(
            term.prod(vf_, control_)
            for term, vf_, control_ in zip(self.terms, vf, control)
        )

    func_for_init = vf


class WrapTerm(AbstractTerm):
    term: AbstractTerm
    unravel_y: callable
    unravel_control: callable
    unravel_vf: callable

    def __init__(
        self, *, term: AbstractTerm, t: Scalar, y: PyTree, args: PyTree, **kwargs
    ):
        super().__init__(**kwargs)

        control = term.contr(t, t + 1e-6)
        vf = term.vf(t, y, args)

        _, unravel_y = ravel_pytree(y)
        _, unravel_control = ravel_pytree(control)
        _, unravel_vf = ravel_pytree(vf)

        self.term = term
        self.unravel_y = unravel_y
        self.unravel_control = unravel_control
        self.unravel_vf = unravel_vf

    def vf(
        self,
        t: Scalar,
        y: Array["state"],  # noqa: F821
        args: PyTree,
    ) -> Array["state*control"]:  # noqa: F821
        y = self.unravel_y(y)
        vf = self.term.vf(t, y, args)
        vf, _ = ravel_pytree(vf)
        return vf

    def contr(self, t0: Scalar, t1: Scalar) -> Array["control"]:  # noqa: F821
        control, _ = ravel_pytree(self.term.contr(t0, t1))
        return control

    def prod(
        self,
        vf: Array["state*control"],  # noqa: F821
        control: Array["control"],  # noqa: F821
    ) -> Array["state"]:  # noqa: F821
        vf = self.unravel_vf(vf)
        control = self.unravel_control(control)
        prod = self.term.prod(vf, control)
        prod, _ = ravel_pytree(prod)
        return prod

    # Define this to skip the extra ravel/unravelling that prod(vf(...), ...) does
    def vf_prod(
        self,
        t: Scalar,
        y: Array["state"],  # noqa: F821
        args: PyTree,
        control: Array["control"],  # noqa: F821
    ) -> Array["state"]:  # noqa: F821
        y = self.unravel_y(y)
        control = self.unravel_control(control)
        vf = self.term.vf(t, y, args)
        prod = self.term.prod(vf, control)
        prod, _ = ravel_pytree(prod)
        return prod

    func_for_init = vf
