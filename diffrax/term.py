import abc
from typing import Callable, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from .custom_types import Array, PyTree, Scalar
from .misc import ω
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
        control $x$, this computes $f(t, y(t), args) \Delta x(t)$ given
        $f(t, y(t), args)$ and $\Delta x(t)$.

        **Arguments:**

        - `vf`: The vector field evaluation; a PyTree of structure $S$.
        - `control`: The control evaluated over an interval; a PyTree of structure $U$.

        **Returns:**

        The interaction between the vector field and control; a PyTree of structure
        $T$.

        !!! note

            This function must be bilinear.
        """
        pass

    def vf_prod(self, t: Scalar, y: PyTree, args: PyTree, control: PyTree) -> PyTree:
        r"""The composition of [`diffrax.AbstractTerm.vf`][] and
        [`diffrax.AbstractTerm.prod`][].

        With a solution $y$ to a differential equation with vector field $f$ and
        control $x$, this computes $f(t, y(t), args) \Delta x(t)$ given $t$, $y(t)$,
        $args$, and $\Delta x(t)$.

        Its default implementation is simply
        ```python
        self.prod(self.vf(t, y, args), control)
        ```

        This is offered as a special case that can be overridden when it is more
        efficient to do so.

        !!! example

            Consider when `vf` computes a matrix-matrix product, and `prod` computes a
            matrix-vector product. Then doing a naive composition corresponds to a
            (matrix-matrix)-vector product, which is less efficient than the
            corresponding matrix-(matrix-vector) product. Overriding this method offers
            a way to reclaim that efficiency.

        !!! example

            This is used extensively for efficiency when backpropagating via
            [`diffrax.BacksolveAdjoint`][].

        **Arguments:**

        - `t`: the integration time.
        - `y`: the evolving state; a PyTree of structure $T$.
        - `args`: any static arguments as passed to [`diffrax.diffeqsolve`][].
        - `control`: The control evaluated over an interval; a PyTree of structure $U$.

        **Returns:**

        A PyTree of structure $T$.

        !!! note

            This function must be linear in `control`.
        """
        return self.prod(self.vf(t, y, args), control)

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

        See [`diffrax.AbstractSolver.func_for_init`][].
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
        diffeqsolve(ode_term, ...)
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
        diffeqsolve(diffusion_term, ...)
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
        diffeqsolve(cde_term, ...)
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

    whose vector field -- control interaction is a dot product.

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

    def vf(self, t: Scalar, y: PyTree, args: PyTree) -> PyTree:
        t = t * self.direction
        return self.term.vf(t, y, args)

    def contr(self, t0: Scalar, t1: Scalar) -> PyTree:
        t0, t1 = jnp.where(self.direction == 1, t0, -t1), jnp.where(
            self.direction == 1, t1, -t0
        )
        return (self.direction * self.term.contr(t0, t1) ** ω).ω

    def prod(self, vf: PyTree, control: PyTree) -> PyTree:
        return self.term.prod(vf, control)

    def func_for_init(self, t: Scalar, y: PyTree, args: PyTree) -> PyTree:
        t = t * self.direction
        return self.term.func_for_init(t, y, args)


class AdjointTerm(AbstractTerm):
    term: AbstractTerm

    def vf(
        self, t: Scalar, y: Tuple[PyTree, PyTree, PyTree], args: PyTree
    ) -> Tuple[PyTree, PyTree, PyTree]:
        # We compute the vector field via `self.vf_prod`. We could also do it manually,
        # but this is relatively painless.#
        #
        # This can be done because `self.vf_prod` is linear in `control`. As such we
        # can obtain just the vector field component by representing this linear
        # operation as a matrix. Which in turn is simply computing the Jacobian.
        #
        # Notes:
        # - Whilst `self.vf_prod` also involves autodifferentiation, we don't
        #   actually compute a second derivative anywhere. (The derivatives are of
        #   different quantities.)
        # - Because the operation is linear, then in some sense this Jacobian isn't
        #   really doing any autodifferentiation at all.
        # - If we wanted we could manually perform the operations that this Jacobian is
        #   doing; in particular this requires `jax.linear_transpose`-ing
        #   `self.term.prod` to get something `control`-shaped.

        # The value of `control` is never actually used -- just its shape, dtype, and
        # PyTree structure. (This is because `self.vf_prod` is linear in `control`.)
        control = self.contr(t, t)

        y_size = sum(np.size(yi) for yi in jax.tree_leaves(y))
        control_size = sum(np.size(ci) for ci in jax.tree_leaves(control))
        if y_size > control_size:
            make_jac = jax.jacfwd
        else:
            make_jac = jax.jacrev

        # Find the tree structure of vf_prod by smuggling it out as an additional
        # result from the Jacobian calculation.
        vf_prod_tree = None
        control_tree = jax.tree_structure(control)

        def _fn(_control):
            _out = self.vf_prod(t, y, args, _control)
            nonlocal vf_prod_tree
            vf_prod_tree = jax.tree_structure(_out)
            return _out

        jac = make_jac(_fn)(control)
        if jax.tree_structure(None) in (vf_prod_tree, control_tree):
            # Very much a faffy, and highly unusual/not-useful edge case to handle.
            raise NotImplementedError(
                "`AdjointTerm` not implemented for `None` controls or states."
            )
        return jax.tree_transpose(vf_prod_tree, control_tree, jac)

    def contr(self, t0: Scalar, t1: Scalar) -> PyTree:
        return (-self.term.contr(t0, t1) ** ω).ω

    def prod(
        self, vf: Tuple[PyTree, PyTree, PyTree], control: PyTree
    ) -> Tuple[PyTree, PyTree, PyTree]:
        # As per what is returnd from `self.vf`, then `vf` has a PyTree structure of
        # (control_tree, vf_prod_tree)

        # Calculate vf_prod_tree by smuggling it out.
        sentinel = vf_prod_tree = object()
        control_tree = jax.tree_structure(control)

        def _get_vf_tree(_, tree):
            nonlocal vf_prod_tree
            structure = jax.tree_structure(tree)
            if vf_prod_tree is sentinel:
                vf_prod_tree = structure
            else:
                assert vf_prod_tree == structure

        jax.tree_map(_get_vf_tree, control, vf)
        assert vf_prod_tree is not sentinel

        vf = jax.tree_transpose(control_tree, vf_prod_tree, vf)

        example_vf_prod = jax.tree_unflatten(
            vf_prod_tree, [0 for _ in range(vf_prod_tree.num_leaves)]
        )

        def _contract(_, vf_piece):
            assert jax.tree_structure(vf_piece) == control_tree
            _contracted = jax.tree_map(_prod, vf_piece, control)
            return sum(jax.tree_leaves(_contracted), 0)

        return jax.tree_map(_contract, example_vf_prod, vf)

    def vf_prod(
        self, t: Scalar, y: Tuple[PyTree, PyTree, PyTree], args: PyTree, control: PyTree
    ) -> Tuple[PyTree, PyTree, PyTree]:
        # Note the inclusion of "implicit" parameters (as `term` might be a callable
        # PyTree a la Equinox) and "explicit" parameters (`args`)
        y, a_y, _, _ = y
        diff_term, nondiff_term = eqx.partition(self.term, eqx.is_inexact_array)

        def _to_vjp(_y, _args, _diff_term):
            _term = eqx.combine(_diff_term, nondiff_term)
            return _term.vf_prod(t, _y, _args, control)

        dy, vjp = jax.vjp(_to_vjp, y, args, diff_term)
        da_y, da_args, da_diff_term = vjp(a_y)
        return dy, da_y, da_args, da_diff_term
