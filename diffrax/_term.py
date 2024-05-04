import abc
import operator
from collections.abc import Callable
from typing import cast, Generic, Optional, TypeVar, Union

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np
from equinox.internal import ω
from jaxtyping import ArrayLike, PyTree, PyTreeDef

from ._custom_types import Args, Control, IntScalarLike, RealScalarLike, VF, Y
from ._misc import upcast_or_raise
from ._path import AbstractPath


_VF = TypeVar("_VF", bound=VF)
_Control = TypeVar("_Control", bound=Control)


class AbstractTerm(eqx.Module, Generic[_VF, _Control]):
    r"""Abstract base class for all terms.

    Let $y$ solve some differential equation with vector field $f$ and control $x$.

    Let $y$ have PyTree structure $T$, let the output of the vector field have
    PyTree structure $S$, and let $x$ have PyTree structure $U$, Then
    $f : T \to S$ whilst the interaction $(f, x) \mapsto f \mathrm{d}x$ is a function
    $(S, U) \to T$.
    """

    @abc.abstractmethod
    def vf(self, t: RealScalarLike, y: Y, args: Args) -> _VF:
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
    def contr(self, t0: RealScalarLike, t1: RealScalarLike, **kwargs) -> _Control:
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
    def prod(self, vf: _VF, control: _Control) -> Y:
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

    def vf_prod(self, t: RealScalarLike, y: Y, args: Args, control: _Control) -> Y:
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

    def is_vf_expensive(
        self,
        t0: RealScalarLike,
        t1: RealScalarLike,
        y: Y,
        args: Args,
    ) -> bool:
        """Specifies whether evaluating the vector field is "expensive", in the
        specific sense that it is cheaper to evaluate `vf_prod` twice than `vf` once.

        Some solvers use this to change their behaviour, so as to act more efficiently.
        """
        return False


class ODETerm(AbstractTerm[_VF, RealScalarLike]):
    r"""A term representing $f(t, y(t), args) \mathrm{d}t$. That is to say, the term
    appearing on the right hand side of an ODE, in which the control is time.

    `vector_field` should return some PyTree, with the same structure as the initial
    state `y0`, and with every leaf shape-broadcastable and dtype-upcastable to the
    equivalent leaf in `y0`.

    !!! example

        ```python
        vector_field = lambda t, y, args: -y
        ode_term = ODETerm(vector_field)
        diffeqsolve(ode_term, ...)
        ```
    """

    vector_field: Callable[[RealScalarLike, Y, Args], _VF]

    def vf(self, t: RealScalarLike, y: Y, args: Args) -> _VF:
        out = self.vector_field(t, y, args)
        if jtu.tree_structure(out) != jtu.tree_structure(y):
            raise ValueError(
                "The vector field inside `ODETerm` must return a pytree with the "
                "same structure as `y0`."
            )

        def _broadcast_and_upcast(oi, yi):
            oi = jnp.broadcast_to(oi, jnp.shape(yi))
            oi = upcast_or_raise(
                oi,
                yi,
                "the vector field passed to `ODETerm`",
                "the corresponding leaf of `y`",
            )
            return oi

        return jtu.tree_map(_broadcast_and_upcast, out, y)

    def contr(self, t0: RealScalarLike, t1: RealScalarLike, **kwargs) -> RealScalarLike:
        return t1 - t0

    def prod(self, vf: _VF, control: RealScalarLike) -> Y:
        def _mul(v):
            c = upcast_or_raise(
                control,
                v,
                "the output of `ODETerm.contr(...)`",
                "the output of `ODETerm.vf(...)`",
            )
            return c * v

        return jtu.tree_map(_mul, vf)


ODETerm.__init__.__doc__ = """**Arguments:**

- `vector_field`: A callable representing the vector field. This callable takes three
    arguments `(t, y, args)`. `t` is a scalar representing the integration time. `y` is
    the evolving state of the system. `args` are any static arguments as passed to
    [`diffrax.diffeqsolve`][].
"""


class _CallableToPath(AbstractPath[_Control]):
    fn: Callable

    @property
    def t0(self):
        return -jnp.inf

    @property
    def t1(self):
        return jnp.inf

    def evaluate(
        self, t0: RealScalarLike, t1: Optional[RealScalarLike] = None, left: bool = True
    ) -> _Control:
        return self.fn(t0, t1)


def _callable_to_path(
    x: Union[
        AbstractPath[_Control], Callable[[RealScalarLike, RealScalarLike], _Control]
    ],
) -> AbstractPath[_Control]:
    if isinstance(x, AbstractPath):
        return x
    else:
        return _CallableToPath(x)


# vf: Shaped[Array, "*state *control"]
# control: Shaped[Array, "*control"]
# return: Shaped[Array, "*state"]
def _prod(vf, control):
    return jnp.tensordot(vf, control, axes=jnp.ndim(control))


class _AbstractControlTerm(AbstractTerm[_VF, _Control]):
    vector_field: Callable[[RealScalarLike, Y, Args], _VF]
    control: Union[
        AbstractPath[_Control], Callable[[RealScalarLike, RealScalarLike], _Control]
    ] = eqx.field(converter=_callable_to_path)  # pyright: ignore

    def vf(self, t: RealScalarLike, y: Y, args: Args) -> VF:
        return self.vector_field(t, y, args)

    def contr(self, t0: RealScalarLike, t1: RealScalarLike, **kwargs) -> _Control:
        return self.control.evaluate(t0, t1, **kwargs)  # pyright: ignore

    def to_ode(self) -> ODETerm:
        r"""If the control is differentiable then $f(t, y(t), args) \mathrm{d}x(t)$
        may be thought of as an ODE as

        $f(t, y(t), args) \frac{\mathrm{d}x}{\mathrm{d}t}\mathrm{d}t$.

        This method converts this `ControlTerm` into the corresponding
        [`diffrax.ODETerm`][] in this way.
        """
        vector_field = _ControlToODE(self)
        return ODETerm(vector_field=vector_field)


_AbstractControlTerm.__init__.__doc__ = """**Arguments:**

- `vector_field`: A callable representing the vector field. This callable takes three
    arguments `(t, y, args)`. `t` is a scalar representing the integration time. `y` is
    the evolving state of the system. `args` are any static arguments as passed to
    [`diffrax.diffeqsolve`][].
- `control`: The control. Should either be (A) a [`diffrax.AbstractPath`][], in which
    case its `evaluate(t0, t1)` method will be used to give the increment of the control
    over a time interval `[t0, t1]`, or (B) a callable `(t0, t1) -> increment`, which
    returns the increment directly.
"""


class ControlTerm(_AbstractControlTerm[_VF, _Control]):
    r"""A term representing the general case of $f(t, y(t), args) \mathrm{d}x(t)$, in
    which the vector field - control interaction is a matrix-vector product.

    `vector_field` and `control` should both return PyTrees, both with the same
    structure as the initial state `y0`. Every dimension of `control` is then
    contracted against the last dimensions of `vector_field`; that is to say if each
    leaf of `y0` has shape `(y1, ..., yN)`, and the corresponding leaf of `control`
    has shape `(c1, ..., cM)`, then the corresponding leaf of `vector_field` should
    have shape `(y1, ..., yN, c1, ..., cM)`.

    A common special case is when `y0` and `control` are vector-valued, and
    `vector_field` is matrix-valued.

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

    def prod(self, vf: _VF, control: _Control) -> Y:
        return jtu.tree_map(_prod, vf, control)


class WeaklyDiagonalControlTerm(_AbstractControlTerm[_VF, _Control]):
    r"""A term representing the case of $f(t, y(t), args) \mathrm{d}x(t)$, in
    which the vector field - control interaction is a matrix-vector product, and the
    matrix is square and diagonal. In this case we may represent the matrix as a vector
    of just its diagonal elements. The matrix-vector product may be calculated by
    pointwise multiplying this vector with the control; this is more computationally
    efficient than writing out the full matrix and then doing a full matrix-vector
    product.

    Correspondingly, `vector_field` and `control` should both return PyTrees, and both
    should have the same structure and leaf shape as the initial state `y0`. These are
    multiplied together pointwise.

    !!! info

        Why "weakly" diagonal? Consider the matrix representation of the vector field,
        as a square diagonal matrix. In general, the (i,i)-th element may depending
        upon any of the values of `y`. It is only if the (i,i)-th element only depends
        upon the i-th element of `y` that the vector field is said to be "diagonal",
        without the "weak". (This stronger property is useful in some SDE solvers.)
    """

    def prod(self, vf: _VF, control: _Control) -> Y:
        with jax.numpy_dtype_promotion("standard"):
            return jtu.tree_map(operator.mul, vf, control)


class _ControlToODE(eqx.Module):
    control_term: _AbstractControlTerm

    def __call__(self, t: RealScalarLike, y: Y, args: Args) -> Y:
        control = self.control_term.control.derivative(t)  # pyright: ignore
        return self.control_term.vf_prod(t, y, args, control)


def _sum(*x):
    return sum(x[1:], x[0])


_Terms = TypeVar("_Terms", bound=tuple[AbstractTerm, ...])


class MultiTerm(AbstractTerm, Generic[_Terms]):
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

    terms: _Terms

    def __init__(self, *terms: AbstractTerm):
        """**Arguments:**

        - `*terms`: Any number of [`diffrax.AbstractTerm`][]s to combine.
        """
        self.terms = terms  # pyright: ignore

    def vf(self, t: RealScalarLike, y: Y, args: Args) -> tuple[PyTree[ArrayLike], ...]:
        return tuple(term.vf(t, y, args) for term in self.terms)

    def contr(
        self, t0: RealScalarLike, t1: RealScalarLike, **kwargs
    ) -> tuple[PyTree[ArrayLike], ...]:
        return tuple(term.contr(t0, t1, **kwargs) for term in self.terms)

    def prod(
        self, vf: tuple[PyTree[ArrayLike], ...], control: tuple[PyTree[ArrayLike], ...]
    ) -> Y:
        out = [
            term.prod(vf_, control_)
            for term, vf_, control_ in zip(self.terms, vf, control)
        ]
        return jtu.tree_map(_sum, *out)

    def vf_prod(
        self,
        t: RealScalarLike,
        y: Y,
        args: Args,
        control: tuple[PyTree[ArrayLike], ...],
    ) -> Y:
        out = [
            term.vf_prod(t, y, args, control_)
            for term, control_ in zip(self.terms, control)
        ]
        return jtu.tree_map(_sum, *out)

    def is_vf_expensive(
        self,
        t0: RealScalarLike,
        t1: RealScalarLike,
        y: Y,
        args: Args,
    ) -> bool:
        return any(term.is_vf_expensive(t0, t1, y, args) for term in self.terms)


class WrapTerm(AbstractTerm[_VF, _Control]):
    term: AbstractTerm[_VF, _Control]
    direction: IntScalarLike

    def vf(self, t: RealScalarLike, y: Y, args: Args) -> _VF:
        t = t * self.direction
        return self.term.vf(t, y, args)

    def contr(self, t0: RealScalarLike, t1: RealScalarLike, **kwargs) -> _Control:
        _t0 = jnp.where(self.direction == 1, t0, -t1)
        _t1 = jnp.where(self.direction == 1, t1, -t0)
        return (self.direction * self.term.contr(_t0, _t1, **kwargs) ** ω).ω

    def prod(self, vf: _VF, control: _Control) -> Y:
        with jax.numpy_dtype_promotion("standard"):
            return self.term.prod(vf, control)

    def vf_prod(self, t: RealScalarLike, y: Y, args: Args, control: _Control) -> Y:
        t = t * self.direction
        return self.term.vf_prod(t, y, args, control)

    def is_vf_expensive(
        self,
        t0: RealScalarLike,
        t1: RealScalarLike,
        y: Y,
        args: Args,
    ) -> bool:
        _t0 = jnp.where(self.direction == 1, t0, -t1)
        _t1 = jnp.where(self.direction == 1, t1, -t0)
        return self.term.is_vf_expensive(_t0, _t1, y, args)


class AdjointTerm(AbstractTerm[_VF, _Control]):
    term: AbstractTerm[_VF, _Control]

    def is_vf_expensive(
        self,
        t0: RealScalarLike,
        t1: RealScalarLike,
        y: tuple[
            PyTree[ArrayLike], PyTree[ArrayLike], PyTree[ArrayLike], PyTree[ArrayLike]
        ],
        args: Args,
    ) -> bool:
        control_struct = jax.eval_shape(self.contr, t0, t1)
        if sum(c.size for c in jtu.tree_leaves(control_struct)) in (0, 1):
            return False
        else:
            return True

    def vf(
        self,
        t: RealScalarLike,
        y: tuple[
            PyTree[ArrayLike], PyTree[ArrayLike], PyTree[ArrayLike], PyTree[ArrayLike]
        ],
        args: Args,
    ) -> PyTree[ArrayLike]:
        # We compute the vector field via `self.vf_prod`. We could also do it manually,
        # but this is relatively painless.
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

        y_size = sum(np.size(yi) for yi in jtu.tree_leaves(y))
        control_size = sum(np.size(ci) for ci in jtu.tree_leaves(control))
        if y_size > control_size:
            make_jac = jax.jacfwd
        else:
            make_jac = jax.jacrev

        # Find the tree structure of vf_prod by smuggling it out as an additional
        # result from the Jacobian calculation.
        sentinel = vf_prod_tree = object()
        control_tree = jtu.tree_structure(control)

        def _fn(_control):
            _out = self.vf_prod(t, y, args, _control)
            nonlocal vf_prod_tree
            structure = jtu.tree_structure(_out)
            if vf_prod_tree is sentinel:
                vf_prod_tree = structure
            else:
                assert vf_prod_tree == structure
            return _out

        jac = make_jac(_fn)(control)
        assert vf_prod_tree is not sentinel
        vf_prod_tree = cast(PyTreeDef, vf_prod_tree)
        if jtu.tree_structure(None) in (vf_prod_tree, control_tree):
            # An unusual/not-useful edge case to handle.
            raise NotImplementedError(
                "`AdjointTerm.vf` not implemented for `None` controls or states."
            )
        return jtu.tree_transpose(vf_prod_tree, control_tree, jac)

    def contr(self, t0: RealScalarLike, t1: RealScalarLike, **kwargs) -> _Control:
        return self.term.contr(t0, t1, **kwargs)

    def prod(
        self, vf: PyTree[ArrayLike], control: _Control
    ) -> tuple[
        PyTree[ArrayLike], PyTree[ArrayLike], PyTree[ArrayLike], PyTree[ArrayLike]
    ]:
        # As per what is returned from `self.vf`, then `vf` has a PyTree structure of
        # (control_tree, vf_prod_tree)

        # Calculate vf_prod_tree by smuggling it out.
        sentinel = vf_prod_tree = object()
        control_tree = jtu.tree_structure(control)

        def _get_vf_tree(_, tree):
            nonlocal vf_prod_tree
            structure = jtu.tree_structure(tree)
            if vf_prod_tree is sentinel:
                vf_prod_tree = structure
            else:
                assert vf_prod_tree == structure

        jtu.tree_map(_get_vf_tree, control, vf)
        assert vf_prod_tree is not sentinel
        vf_prod_tree = cast(PyTreeDef, vf_prod_tree)

        vf = jtu.tree_transpose(control_tree, vf_prod_tree, vf)

        example_vf_prod = jtu.tree_unflatten(
            vf_prod_tree, [0 for _ in range(vf_prod_tree.num_leaves)]
        )

        def _contract(_, vf_piece):
            assert jtu.tree_structure(vf_piece) == control_tree
            _contracted = jtu.tree_map(_prod, vf_piece, control)
            return sum(jtu.tree_leaves(_contracted), 0)

        return jtu.tree_map(_contract, example_vf_prod, vf)

    def vf_prod(
        self,
        t: RealScalarLike,
        y: tuple[
            PyTree[ArrayLike], PyTree[ArrayLike], PyTree[ArrayLike], PyTree[ArrayLike]
        ],
        args: Args,
        control: _Control,
    ) -> tuple[
        PyTree[ArrayLike], PyTree[ArrayLike], PyTree[ArrayLike], PyTree[ArrayLike]
    ]:
        # Note the inclusion of "implicit" parameters (as `term` might be a callable
        # PyTree a la Equinox) and "explicit" parameters (`args`)
        y, a_y, _, _ = y
        diff_args, nondiff_args = eqx.partition(args, eqx.is_inexact_array)
        diff_term, nondiff_term = eqx.partition(self.term, eqx.is_inexact_array)

        def _to_vjp(_y, _diff_args, _diff_term):
            _args = eqx.combine(_diff_args, nondiff_args)
            _term = eqx.combine(_diff_term, nondiff_term)
            return _term.vf_prod(t, _y, _args, control)

        dy, vjp = jax.vjp(_to_vjp, y, diff_args, diff_term)
        da_y, da_diff_args, da_diff_term = vjp((-(a_y**ω)).ω)
        return dy, da_y, da_diff_args, da_diff_term
