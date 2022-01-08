import abc
from typing import Callable, Tuple

import equinox as eqx
import jax
import jax.flatten_util as fu
import jax.numpy as jnp

from .custom_types import Array, PyTree, Scalar
from .path import AbstractPath


class AbstractTerm(eqx.Module):
    r"""Abstract base class for all terms.

    Let $y$ solve some differential equation with vector field $f$ and control $x$.

    Let $y$ have PyTree structure $T$, let the output of the vector field have
    PyTree structure $S$, and let $x$ have PyTree structure $U$, Then
    $f : T \to S$ whilst the interaction $(f, x) \mapsto f \mathrm{d}x$ is a function
    $(S, U) \to T$.
    """

    @abc.abstractmethod
    def vf(self, t: Scalar, y: PyTree, args: PyTree) -> PyTree:
        """The vector field.

        Represents a function $f(t, y(t), args)$.

        **Arguments:**

        - `t`: the integration time.
        - `y`: the evolving state; a PyTree of structure $T$.
        - `args`: any static arguments as passed to [`diffrax.diffeqsolve`][].

        **Returns:**

        A PyTree of structure $S$.
        """
        pass

    @abc.abstractmethod
    def contr(self, t0: Scalar, t1: Scalar) -> PyTree:
        r"""The control.

        Represents the $\mathrm{d}t$ in an ODE, or the $\mathrm{d}w(t)$ in an SDE, etc.

        Most numerical ODE solvers work by making a step of length
        $\Delta t = t_1 - t_0$. Likewise most numerical SDE solvers work by sampling
        some Brownian motion $\Delta w \sim \mathcal{N}(0, t_1 - t_0)$.

        Correspondingly a control is *not* defined at a point. Instead it is defined
        over an interval $[t_0, t_1]$.

        **Arguments:**

        - `t0`: the start of the interval.
        - `t1`: the end of the interval.

        **Returns:**

        A PyTree of structure $U$. For a control $x$ then the result should
        represent $x(t_1) - x(t_0)$.
        """
        pass

    @abc.abstractmethod
    def prod(self, vf: PyTree, control: PyTree) -> PyTree:
        r"""Determines the interaction between vector field and control.

        With a solution $y$ to a differential equation with vector field $f$ and
        control $x$, this computes $f(t, y(t), args) \Delta x(t)$ given $f(t, y(t))$
        and $\Delta x(t)$.

        **Arguments:**

        - `vf`: The vector field evaluation; a PyTree of structure $S$.
        - `control`: The control evaluated over an interval; a PyTree of structure $U$.

        **Returns:**

        The interaction between the vector field and control; a PyTree of structure
        $T$.
        """
        pass

    def vf_prod(self, t: Scalar, y: PyTree, args: PyTree, control: PyTree) -> PyTree:
        r"""The composition of [`diffrax.AbstractTerm.vf`][] and
        [`diffrax.AbstractTerm.prod`][].

        With a solution $y$ to a differential equation with vector field $f$ and
        control $x$, this computes $f(t, y(t), args) \Delta x(t)$ given $t$, $y(t)$,
        $args$, and $\Delta x(t)$.

        This is offered as a special case that can be overridden when it is more
        efficient to do so.

        !!! example

            Consider when `vf` computes a matrix-matrix product, and `prod` computes a
            matrix-vector product. Then doing a naive composition corresponds to a
            (matrix-matrix)-vector product, which is less efficient than the
            corresponding matrix-(matrix-vector) product. Overriding this method offers
            a way to reclaim that efficiency.

        **Arguments:**

        - `t`: the integration time.
        - `y`: the evolving state; a PyTree of structure $T$.
        - `args`: any static arguments as passed to [`diffrax.diffeqsolve`][].
        - `control`: The control evaluated over an interval; a PyTree of structure $U$.

        **Returns:**

        A PyTree of structure $T$.
        """
        return self.prod(self.vector_field(t, y, args), control)

    # This is a pinhole break in our vector-field/control abstraction.
    # Everywhere else we get to evaluate over some interval, which allows us to
    # evaluate our control over that interval. However to select the initial point in
    # an adapative step size scheme, the standard heuristic is to start by making
    # evaluations at just the initial point -- no intervals involved.
    def func_for_init(self, t: Scalar, y: PyTree, args: PyTree) -> PyTree:
        """This is a special-cased version of [`diffrax.AbstractTerm.vf`][].

        If it so happens that the PyTree structures $T$ and $S$ are the same, then a
        subclass of `AbstractTerm` shoud set `func_for_init = vf`.

        This case is used when selecting the initial step size of an ODE solve
        automatically.
        """

        raise ValueError(
            "An initial step size cannot be selected automatically. The most common "
            "scenario for this error to occur is when trying to use adaptive step "
            "size solvers with SDEs. Please specify an initial `dt0` instead."
        )


class ODETerm(AbstractTerm):
    r"""A term representing $f(t, y(t), args) \mathrm{d}t$. That is to say, the term
    appearing on the right hand side of an ODE, in which the control is time.

    !!! example

        ```python
        vector_field = lambda t, y, args: -y
        ode_term = ODETerm(vector_field)
        solver = Euler(ode_term)
        ```
    """
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


ODETerm.__init__.__doc__ = """**Arguments:**

- `vector_field`: A callable representing the vector field. This callable takes three
    arguments `(t, y, args)`. `t` is a scalar representing the integration time. `y` is
    the evolving state of the system. `args` are any static arguments as passed to
    [`diffrax.diffeqsolve`][].
"""


def _prod(
    vf: Array["state":..., "control":...], control: Array["control":...]  # noqa: F821
) -> Array["state":...]:  # noqa: F821
    return jnp.tensordot(vf, control, axes=jnp.asarray(control).ndim)


class ControlTerm(AbstractTerm):
    r"""A term representing the general case of $f(t, y(t), args) \mathrm{d}x(t)$, in
    which the vector field - control interaction is a matrix-vector product.

    !!! example

        ```python
        control = UnsafeBrownianPath(shape=(2,), key=...)
        vector_field = lambda t, y, args: jnp.stack([y, y], axis=-1)
        diffusion_term = ControlTerm(vector_field, control)
        solver = Euler(diffusion_term)
        ```

    !!! example

        ```python
        ts = jnp.array([1., 2., 2.5, 3.])
        data = jnp.array([[0.1, 2.0],
                          [0.3, 1.5],
                          [1.0, 1.6],
                          [0.2, 1.1]])
        control = LinearInterpolation(ts, data)
        vector_field = lambda t, y, args: jnp.stack([y, y], axis=-1)
        cde_term = ControlTerm(vector_field, control)
        solver = Euler(cde_term)
        ```
    """
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
        r"""If the control is differentiable then $f(t, y(t), args) \mathrm{d}x(t)$
        may be thought of as an ODE as
        $f(t, y(t), args) \frac{\mathrm{d}x}{\mathrm{d}t}\mathrm{d}t$.

        This method converts this `ControlTerm` into the corresponding
        [`diffrax.ODETerm`][] in this way.
        """
        vector_field = _ControlToODE(self)
        return ODETerm(vector_field=vector_field)

    # func_for_init deliberately not set.


ControlTerm.__init__.__doc__ = """**Arguments:**

- `vector_field`: A callable representing the vector field. This callable takes three
    arguments `(t, y, args)`. `t` is a scalar representing the integration time. `y` is
    the evolving state of the system. `args` are any static arguments as passed to
    [`diffrax.diffeqsolve`][].
- `control`: A callable representing the control. Should have an `evaluate(t0, t1)`
    method. If using [`diffrax.ControlTerm.to_ode`][] then it should have a
    `derivative(t)` method.
"""


class _ControlToODE(eqx.Module):
    control_term: ControlTerm

    def __call__(self, t: Scalar, y: PyTree, args: PyTree) -> PyTree:
        control = self.control_term.control.derivative(t)
        return self.control_term.vf_prod(t, y, args, control)


def _sum(*x):
    return sum(x[1:], x[0])


class MultiTerm(AbstractTerm):
    r"""Accumulates multiple terms into a single term.

    Consider the SDE

    $\mathrm{d}y(t) = f(t, y(t))\mathrm{d}t + g(t, y(t))\mathrm{d}w(t)$

    This has two terms on the right hand side. It may be represented with a single
    term as

    $\mathrm{d}y(t) = [f(t, y(t)), g(t, y(t))] \cdot [\mathrm{d}t, \mathrm{d}w(t)]$

    whose vector field - control interaction is a dot product.

    `MultiTerm` performs this transform. For simplicitly most differential equation
    solvers (at least those built-in to Diffrax) accept just a single term, so this
    transform is a necessary part of e.g. solving an SDE with both drift and diffusion.
    """

    terms: Tuple[AbstractTerm, ...]

    def __init__(self, *terms):
        """**Arguments:**

        - `*terms`: Any number of [`diffrax.AbstractTerm`][]s to combine.
        """
        self.terms = terms

    def vf(self, t: Scalar, y: PyTree, args: PyTree) -> Tuple[PyTree, ...]:
        return tuple(term.vf(t, y, args) for term in self.terms)

    def contr(self, t0: Scalar, t1: Scalar) -> Tuple[PyTree, ...]:
        return tuple(term.contr(t0, t1) for term in self.terms)

    def prod(self, vf: Tuple[PyTree, ...], control: Tuple[PyTree, ...]) -> PyTree:
        out = [
            term.prod(vf_, control_)
            for term, vf_, control_ in zip(self.terms, vf, control)
        ]
        return jax.tree_map(_sum, *out)

    def func_for_init(self, t: Scalar, y: PyTree, args: PyTree) -> Tuple[PyTree, ...]:
        return tuple(term.func_for_init(t, y, args) for term in self.terms)


class WrapTerm(AbstractTerm):
    term: AbstractTerm
    direction: Scalar
    unravel_y: callable
    unravel_control: callable
    unravel_vf: callable

    def __init__(
        self,
        *,
        term: AbstractTerm,
        t: Scalar,
        y: PyTree,
        args: PyTree,
        direction,
        **kwargs
    ):
        super().__init__(**kwargs)

        control = term.contr(t, t + 1e-6)
        vf = term.vf(t, y, args)

        _, unravel_y = fu.ravel_pytree(y)
        _, unravel_control = fu.ravel_pytree(control)
        _, unravel_vf = fu.ravel_pytree(vf)

        self.term = term
        self.direction = direction
        self.unravel_y = unravel_y
        self.unravel_control = unravel_control
        self.unravel_vf = unravel_vf

    def vf(
        self,
        t: Scalar,
        y: Array["state"],  # noqa: F821
        args: PyTree,
    ) -> Array["state*control"]:  # noqa: F821
        t = t * self.direction
        y = self.unravel_y(y)
        vf = self.term.vf(t, y, args)
        vf, _ = fu.ravel_pytree(vf)
        return vf

    def contr(self, t0: Scalar, t1: Scalar) -> Array["control"]:  # noqa: F821
        t0, t1 = jnp.where(self.direction == 1, t0, -t1), jnp.where(
            self.direction == 1, t1, -t0
        )
        control, _ = fu.ravel_pytree(self.term.contr(t0, t1))
        control = control * self.direction
        return control

    def prod(
        self,
        vf: Array["state*control"],  # noqa: F821
        control: Array["control"],  # noqa: F821
    ) -> Array["state"]:  # noqa: F821
        vf = self.unravel_vf(vf)
        control = self.unravel_control(control)
        prod = self.term.prod(vf, control)
        prod, _ = fu.ravel_pytree(prod)
        return prod

    # Define this to skip the extra ravel/unravelling that prod(vf(...), ...) does
    def vf_prod(
        self,
        t: Scalar,
        y: Array["state"],  # noqa: F821
        args: PyTree,
        control: Array["control"],  # noqa: F821
    ) -> Array["state"]:  # noqa: F821
        t = t * self.direction
        y = self.unravel_y(y)
        control = self.unravel_control(control)
        vf = self.term.vf(t, y, args)
        prod = self.term.prod(vf, control)
        prod, _ = fu.ravel_pytree(prod)
        return prod

    def func_for_init(
        self,
        t: Scalar,
        y: Array["state"],  # noqa: F821
        args: PyTree,
    ) -> Array["state*control"]:  # noqa: F821
        t = t * self.direction
        y = self.unravel_y(y)
        vf = self.term.func_for_init(t, y, args)
        vf, _ = fu.ravel_pytree(vf)
        return vf
