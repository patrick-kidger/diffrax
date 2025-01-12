import abc
import operator
import warnings
from collections.abc import Callable
from typing import cast, Generic, Optional, TypeVar, Union
from typing_extensions import TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import lineax as lx
import numpy as np
from equinox.internal import ω
from jaxtyping import Array, ArrayLike, PyTree, PyTreeDef, Shaped

from ._brownian import AbstractBrownianPath
from ._custom_types import (
    AbstractBrownianIncrement,
    Args,
    Control,
    IntScalarLike,
    RealScalarLike,
    VF,
    Y,
)
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

        !!! note

            This function must be bilinear.

        **Arguments:**

        - `vf`: The vector field evaluation; a PyTree of structure $S$.
        - `control`: The control evaluated over an interval; a PyTree of structure $U$.

        **Returns:**

        The interaction between the vector field and control; a PyTree of structure
        $T$.
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
) -> AbstractPath:
    if isinstance(x, AbstractPath):
        return x
    else:
        return _CallableToPath(x)


# vf: Shaped[Array, "*state *control"]
# control: Shaped[Array, "*control"]
# return: Shaped[Array, "*state"]
def _prod(vf, control):
    return jnp.tensordot(jnp.conj(vf), control, axes=jnp.ndim(control))


class ControlTerm(AbstractTerm[_VF, _Control]):
    r"""A term representing the general case of $f(t, y(t), args) \mathrm{d}x(t)$, in
    which the vector field ($f$) - control ($\mathrm{d}x$) interaction is a
    matrix-vector product.

    This is typically used for either stochastic differential equations or for
    controlled differential equations.

    `ControlTerm` can be used in two different ways.

    1. Simple way: directly return JAX arrays.

        `vector_field` and `control` should both return PyTrees, both with the same
        structure as the initial state `y0`. All leaves should be JAX arrays.

        If each leaf of `y0` has shape `(y1, ..., yN)`, and the corresponding leaf of
        `control` has shape `(c1, ..., cM)`, then the corresponding leaf of
        `vector_field` should have shape `(y1, ..., yN, c1, ..., cM)`. Leaf-by-leaf, the
        corresponding dimensions of `vector_field` and control are contracted against
        each other.

        This includes normal matrix-vector products as a special case: when `y0` is an
        array with shape `(m,)`, the control is an array with shape `(n,)`, and the
        vector field is an array with shape `(m, n)`.

    2. Advanced way: have the vector field return a [Lineax linear operator](https://docs.kidger.site/lineax/api/operators).

        This is suitable for use cases in which you know that the vector field has
        special structure -- e.g. it is diagonal -- and you would like to use that
        structure for a more efficient implementation.

        In this case, then `vector_field` should return a
        [Lineax linear operator](https://docs.kidger.site/lineax/api/operators), the
        control can return anything compatible with the
        [`.mv`](https://docs.kidger.site/lineax/api/operators/#lineax.AbstractLinearOperator.mv)
        method of that operator, and the interaction is defined as
        `vector_field(t0, y, arg).mv(control(t0, t1))`.

        In this case no special PyTree handling is done -- perform this inside the
        operator's `.mv` if required. (As you can see, this approach is basically about
        deferring the whole linear operation to Lineax.)

    !!! Example

        In this example we consider an SDE with `m`-dimensional state
        $y \in \mathbb{R}^m$, an `n`-dimensional Brownian motion
        $W(t) \in \mathbb{R}^n$, and a constant diffusion of shape `(m, n)`.

        $\mathrm{d}y(t) = \begin{bmatrix} 1 & ... & 1 \\ & ... & \\ 1 & ... & 1 \end{bmatrix} \mathrm{d}W(t)$

        ```python
        from diffrax import ControlTerm, diffeqsolve, UnsafeBrownianPath

        y0 = jnp.ones((m,))
        control = UnsafeBrownianPath(shape=(n,), key=...)

        def vector_field(t, y, args):
            return jnp.ones((m, n))

        diffusion_term = ControlTerm(vector_field, control)
        diffeqsolve(terms=diffusion_term, y0=y0, ...)
        ```

    !!! Example

        In this example we consider an SDE with a one-dimensional state
        $y(t) \in \mathbb{R}$ and a two-dimensional Brownian motion
        $W(t) \in \mathbb{R}^2$, given by:

        $\mathrm{d}y(t) = \begin{bmatrix} y(t) \\ y(t) + 1 \end{bmatrix} \mathrm{d}W(t)$

        We use the simple matrix-vector product way of combining things.

        ```python
        from diffrax import ControlTerm, diffeqsolve, UnsafeBrownianPath

        control = UnsafeBrownianPath(shape=(2,), key=...)

        def vector_field(t, y, args):
            return jnp.stack([y, y + 1], axis=-1)

        diffusion_term = ControlTerm(vector_field, control)
        diffeqsolve(diffusion_term, ...)
        ```

    !!! Example

        In this example we consider an SDE with two-dimensional state
        $(y_1(t), y_2(t)) \in \mathbb{R}^2$ and a two-dimensional Brownian motion
        $W(t) \in \mathbb{R}^2$ -- and for which the diffusion matrix is
        diagonal.

        $\mathrm{d}\begin{bmatrix} y_1 \\ y_2 \end{bmatrix}(t) = \begin{bmatrix} y_2(t) & 0 \\ 0 & y_1(t) \end{bmatrix} \mathrm{d}W(t)$

        As such we use the more-advanced approach of using
        [Lineax](https://github.com/patrick-kidger/lineax/)'s linear operators to
        represent the diffusion matrix.

        ```python
        from diffrax import ControlTerm, diffeqsolve, UnsafeBrownianPath

        control = UnsafeBrownianPath(shape=(2,), key=...)

        def vector_field(t, y, args):
            # y is a JAX array of shape (2,)
            y1, y2 = y
            diagonal = jnp.array([y2, y1])
            return lineax.DiagonalLinearOperator(diagonal)

        diffusion_term = ControlTerm(vector_field, control)
        diffeqsolve(diffusion_term, ...)
        ```

    !!! Example

        In this example we consider a controlled differnetial equation, for which the
        control is given by an interpolation of some data. (See also the
        [neural controlled differential equation](../examples/neural_cde/) example.)

        ```python
        from diffrax import ControlTerm, diffeqsolve, LinearInterpolation, UnsafeBrownianPath

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
    """  # noqa: E501

    vector_field: Callable[[RealScalarLike, Y, Args], _VF]
    control: AbstractPath[_Control]

    def __init__(
        self,
        vector_field: Callable[[RealScalarLike, Y, Args], _VF],
        control: Union[
            AbstractPath[_Control], Callable[[RealScalarLike, RealScalarLike], _Control]
        ],
    ):
        self.vector_field = vector_field
        self.control = _callable_to_path(control)

    def vf(self, t: RealScalarLike, y: Y, args: Args) -> VF:
        return self.vector_field(t, y, args)

    def contr(self, t0: RealScalarLike, t1: RealScalarLike, **kwargs) -> _Control:
        return self.control.evaluate(t0, t1, **kwargs)

    def prod(self, vf: _VF, control: _Control) -> Y:
        if isinstance(vf, lx.AbstractLinearOperator):
            return vf.mv(control)
        else:
            return jtu.tree_map(_prod, vf, control)

    def vf_prod(self, t: RealScalarLike, y: Y, args: Args, control: _Control) -> Y:
        vf = self.vf(t, y, args)
        out = self.prod(vf, control)

        def _raise():
            # SDEs are a common special case; try to make the error message a little
            # easier to understand in this case!
            if isinstance(self.control, AbstractBrownianPath):
                diffusion_word = "diffusion"
                control_word = "Brownian motion"
                diffusion_phrase = "diffusion matrix"
            else:
                diffusion_word = "vector field"
                control_word = "control"
                diffusion_phrase = "vector field in a control term"
            if isinstance(vf, lx.AbstractLinearOperator):
                dot_phrase = (
                    f"combined with `{type(vf).__module__}.{type(vf).__qualname__}.mv`"
                )
            else:
                dot_phrase = "dotted together"
            vf_str = eqx.tree_pformat(vf)
            control_str = eqx.tree_pformat(control)
            out_str = eqx.tree_pformat(out)
            y_str = eqx.tree_pformat(y)
            if "\n" in vf_str:
                vf_str = f"\n```\n{vf_str}\n```\n"
            else:
                vf_str = f" `{vf_str}` "
            if "\n" in control_str:
                control_str = f"\n```\n{control_str}\n```\n"
            else:
                control_str = f" `{control_str}`, "
            if "\n" in out_str:
                out_str = f"\n```\n{out_str}\n```\n"
            else:
                out_str = f" `{out_str}`, "
            if "\n" in y_str:
                y_str = f"\n```\n{y_str}\n```\n"
            else:
                y_str = f" `{y_str}`.\n"
            raise ValueError(
                "The `ControlTerm` returned arrays whose output structure did not "
                "match the structure of the evolving state `y`. Specifically, the "
                f"{diffusion_word} had structure{vf_str}and the {control_word} "
                f"had structure{control_str}which when {dot_phrase} produced an "
                f"output of structure{out_str}which is different to the evolving "
                f"state `y` which had structure{y_str}"
                "\n"
                "This became an error in Diffrax 0.7.0. In previous versions of "
                "Diffrax then the output was broadcast to the shape of `y`. This "
                "has been removed as it was a common source of bugs.\n"
                "\n"
                "To walk you through what is going on, here is a sample program "
                "that now raises an error:\n"
                "```\n"
                "import diffrax as dfx\n"
                "import jax.numpy as jnp\n"
                "import jax.random as jr\n"
                "\n"
                "def drift(t, y, args):\n"
                "    return -y\n"
                "\n"
                "def diffusion(t, y, args):\n"
                "    return jnp.array([1., 0.5])\n"
                "\n"
                "key = jr.key(0)\n"
                "bm = dfx.VirtualBrownianTree(t0=0, t1=1, tol=1e-3, shape=(2,), key=key)\n"  # noqa: E501
                "terms = dfx.MultiTerm(dfx.ODETerm(drift), dfx.ControlTerm(diffusion, bm))\n"  # noqa: E501
                "solver = dfx.Euler()\n"
                "y0 = jnp.array([1., 1.])\n"
                "dfx.diffeqsolve(terms, solver, t0=0, t1=1, dt0=0.1, y0=y0)\n"
                "```\n"
                "In this case, the diffusion returns an array of shape `(2,)` and "
                "the Brownian motion is of shape `(2,)`. By the rules of "
                "`ControlTerm`, they are then dotted together so that the "
                "diffusion term returns a scalar. Under previous versions of "
                "Diffrax, this would then be broadcast out to both elements of the "
                "evolving state `y`, corresponding to the SDE:\n"
                "```\n"
                "dy₁(t) = -y₁(t) dt + dW₁ + 0.5 dW₂\n"
                "dy₂(t) = -y₂(t) dt + dW₁ + 0.5 dW₂\n"
                "```\n"
                "or the equivalent in vector notation, with `y(t), W(t) ⋹ R²`\n"
                "```\n"
                "dy(t) = -y(t) dt + [[1, 0.5], [1, 0.5]] dW\n"
                "```\n"
                "Which may have been unexpected! Quite possibly what was actually "
                "intended was an SDE with diagonal noise:\n"
                "```\n"
                "dy(t) = -y(t) dt + [[1, 0], [0, 0.5]] dW\n"
                "```\n"
                "\n"
                "As of Diffrax 0.7.0, the recommended way to express the "
                f"{diffusion_phrase} is to use a Lineax linear operator. "
                "(https://docs.kidger.site/lineax/api/operators/) For example, to "
                "represent diagonal noise in the example above:\n"
                "```python\n"
                "import lineax as lx\n"
                "\n"
                "def diffusion(t, y, args):\n"
                "    diagonal = jnp.array([1., 0.5])\n"
                "    return lx.DiagonalLinearOperator(diagonal)\n"
                "```\n"
            )

        if jtu.tree_structure(y) != jtu.tree_structure(out):
            _raise()

        def _check_shape(yi, out_i):
            if jnp.shape(yi) != jnp.shape(out_i):
                _raise()

        jtu.tree_map(_check_shape, y, out)
        return out

    def to_ode(self) -> ODETerm:
        r"""If the control is differentiable then $f(t, y(t), args) \mathrm{d}x(t)$
        may be thought of as an ODE as

        $f(t, y(t), args) \frac{\mathrm{d}x}{\mathrm{d}t}\mathrm{d}t$.

        This method converts this `ControlTerm` into the corresponding
        [`diffrax.ODETerm`][] in this way.
        """
        vector_field = _ControlToODE(self)
        return ODETerm(vector_field=vector_field)


ControlTerm.__init__.__doc__ = """**Arguments:**

- `vector_field`: A callable representing the vector field. This callable takes three
    arguments `(t, y, args)`. `t` is a scalar representing the integration time. `y` is
    the evolving state of the system. `args` are any static arguments as passed to
    [`diffrax.diffeqsolve`][]. This `vector_field` can either be

    1. a function that returns a PyTree of JAX arrays, or
    2. it can return a
        [Lineax linear operator](https://docs.kidger.site/lineax/api/operators),
        as described above.

- `control`: The control. Should either be

    1. a [`diffrax.AbstractPath`][], in which case its `.evaluate(t0, t1)` method
        will be used to give the increment of the control over a time interval
        `[t0, t1]`, or
    2. a callable `(t0, t1) -> increment`, which returns the increment directly.
"""


def WeaklyDiagonalControlTerm(vector_field, control):
    r"""
    DEPRECATED. Prefer:

    ```python
    def vector_field(t, y, args):
        return lineax.DiagonalLinearOperator(...)

    diffrax.ControlTerm(vector_field, ...)
    ```

    The current implementation is a backward-compatible shim that returns something like
    the code snippet the above.

    ---

    A term representing the case of $f(t, y(t), args) \mathrm{d}x(t)$, in
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

    warnings.warn(
        "`WeaklyDiagonalControlTerm` is now deprecated, in favour combining "
        "`ControlTerm` with a `lineax.AbstractLinearOperator`. This offers a way "
        "to define a vector field with any kind of structure -- diagonal or "
        "otherwise.\n"
        "For a diagonal linear operator, then this can be easily converted as "
        "follows. What was previously:\n"
        "```\n"
        "def vector_field(t, y, args):\n"
        "    ...\n"
        "    return some_vector\n"
        "\n"
        "diffrax.WeaklyDiagonalControlTerm(vector_field)\n"
        "```\n"
        "is now:\n"
        "```\n"
        "import lineax\n"
        "\n"
        "def vector_field(t, y, args):\n"
        "    ...\n"
        "    return lineax.DiagonalLinearOperator(some_vector)\n"
        "\n"
        "diffrax.ControlTerm(vector_field)\n"
        "```\n"
        "Lineax is available at `https://github.com/patrick-kidger/lineax`.\n",
        stacklevel=2,
    )

    def new_vector_field(t, y, args):
        vf = vector_field(t, y, args)
        return lx.DiagonalLinearOperator(vf)

    return ControlTerm(new_vector_field, control)


class _ControlToODE(eqx.Module):
    control_term: ControlTerm

    def __call__(self, t: RealScalarLike, y: Y, args: Args) -> Y:
        control = self.control_term.control.derivative(t)
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
        control_struct = eqx.filter_eval_shape(self.contr, t0, t1)
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


# The Underdamped Langevin SDE trajectory consists of two components: the position
# `x` and the velocity `v`. Both of these have the same shape.
# So, by UnderdampedLangevinX we denote the shape of the x component, and by
# UnderdampedLangevinTuple we denote the shape of the tuple (x, v).
UnderdampedLangevinLeaf: TypeAlias = Shaped[Array, " *underdamped_langevin"]
UnderdampedLangevinX: TypeAlias = PyTree[
    Shaped[Array, "?*underdamped_langevin"], "UnderdampedLangevinX"
]
UnderdampedLangevinTuple: TypeAlias = tuple[UnderdampedLangevinX, UnderdampedLangevinX]


def _broadcast_pytree(source, target_tree):
    # Broadcasts the source PyTree to the shape and PyTree structure of
    # target_tree_shape. Requires that source is a prefix tree of target_tree
    # This is used to broadcast gamma and u to the shape of x0 and v0
    def inner_broadcast(_src_arr, _inner_target_tree):
        _arr = jnp.asarray(_src_arr)

        def fun(_leaf):
            return jnp.asarray(
                jnp.broadcast_to(_arr, _leaf.shape), dtype=jnp.result_type(_leaf)
            )

        return jtu.tree_map(fun, _inner_target_tree)

    return jtu.tree_map(inner_broadcast, source, target_tree)


def broadcast_underdamped_langevin_arg(
    arg: PyTree[ArrayLike], x: UnderdampedLangevinX, arg_name: str
) -> UnderdampedLangevinX:
    """Broadcasts the argument `arg` to the same structure as the position `x`."""
    try:
        return _broadcast_pytree(arg, x)
    except ValueError:
        raise RuntimeError(
            "The PyTree structure and shapes of the arguments `gamma` and `u`"
            "in the Underdamped Langevin term must be the same as the structure"
            "and shapes of the position `x`."
        )


class UnderdampedLangevinDiffusionTerm(
    AbstractTerm[
        UnderdampedLangevinX, Union[UnderdampedLangevinX, AbstractBrownianIncrement]
    ]
):
    r"""Represents the diffusion term in the Underdamped Langevin Diffusion (ULD).
    The ULD SDE takes the form:

    \begin{align*}
        \mathrm{d} x(t) &= v(t) \, \mathrm{d}t \\
        \mathrm{d} v(t) &= - \gamma \, v(t) \, \mathrm{d}t - u \,
        \nabla \! f( x(t) ) \, \mathrm{d}t + \sqrt{2 \gamma u} \, \mathrm{d} w(t),
    \end{align*}

    where $x(t), v(t) \in \mathbb{R}^d$ represent the position
    and velocity, $w$ is a Brownian motion in $\mathbb{R}^d$,
    $f: \mathbb{R}^d \rightarrow \mathbb{R}$ is a potential function, and
    $\gamma , u \in \mathbb{R}^{d \times d}$ are diagonal matrices governing
    the friction and the damping of the system.
    """

    gamma: PyTree[ArrayLike]
    u: PyTree[ArrayLike]
    control: AbstractBrownianPath

    def __init__(
        self,
        gamma: PyTree[ArrayLike],
        u: PyTree[ArrayLike],
        bm: AbstractBrownianPath,
    ):
        r"""
        **Arguments:**

        - `gamma`: A vector containing the diagonal entries of the friction matrix;
            a scalar or a PyTree of the same shape as the position vector $x$.
        - `u`: A vector containing the diagonal entries of the damping matrix;
            a scalar or a PyTree of the same shape as the position vector $x$.
        - `bm`: A Brownian path representing the Brownian motion $w$.
        """
        self.gamma = gamma
        self.u = u
        self.control = bm

    def vf(
        self, t: RealScalarLike, y: UnderdampedLangevinTuple, args: Args
    ) -> UnderdampedLangevinX:
        x, v = y
        # gamma, u and v can all have different pytree structures, we only know that
        # gamma and u are prefixes of v

        gamma = broadcast_underdamped_langevin_arg(self.gamma, v, "gamma")
        u = broadcast_underdamped_langevin_arg(self.u, v, "u")

        def _fun(_gamma, _u):
            return jnp.sqrt(2 * _gamma * _u)

        vf_v = jtu.tree_map(_fun, gamma, u)
        return vf_v

    def contr(
        self, t0: RealScalarLike, t1: RealScalarLike, **kwargs
    ) -> Union[UnderdampedLangevinX, AbstractBrownianIncrement]:
        return self.control.evaluate(t0, t1, **kwargs)

    def prod(
        self, vf: UnderdampedLangevinX, control: UnderdampedLangevinX
    ) -> UnderdampedLangevinTuple:
        # The vf is only for the velocity component. The position component is
        # unaffected by the diffusion.
        dw = control
        v_out = jtu.tree_map(operator.mul, vf, dw)
        x_out = jtu.tree_map(jnp.zeros_like, v_out)
        return x_out, v_out


class UnderdampedLangevinDriftTerm(AbstractTerm):
    r"""Represents the drift term in the Underdamped Langevin Diffusion (ULD).
    The ULD SDE takes the form:

    \begin{align*}
        \mathrm{d} x(t) &= v(t) \, \mathrm{d}t \\
        \mathrm{d} v(t) &= - \gamma \, v(t) \, \mathrm{d}t - u \,
        \nabla \! f( x(t) ) \, \mathrm{d}t + \sqrt{2 \gamma u} \, \mathrm{d} w(t),
    \end{align*}

    where $x(t), v(t) \in \mathbb{R}^d$ represent the position
    and velocity, $w$ is a Brownian motion in $\mathbb{R}^d$,
    $f: \mathbb{R}^d \rightarrow \mathbb{R}$ is a potential function, and
    $\gamma , u \in \mathbb{R}^{d \times d}$ are diagonal matrices governing
    the friction and the damping of the system.
    """

    gamma: PyTree[ArrayLike]
    u: PyTree[ArrayLike]
    grad_f: Callable[[UnderdampedLangevinX, Args], UnderdampedLangevinX]

    def __init__(
        self,
        gamma: PyTree[ArrayLike],
        u: PyTree[ArrayLike],
        grad_f: Callable[[UnderdampedLangevinX, Args], UnderdampedLangevinX],
    ):
        r"""
        **Arguments:**

        - `gamma`: A vector containing the diagonal entries of the friction matrix;
            a scalar or a PyTree of the same shape as the position vector $x$.
        - `u`: A vector containing the diagonal entries of the damping matrix;
            a scalar or a PyTree of the same shape as the position vector $x$.
        - `grad_f`: A callable representing the gradient of the potential function $f$.
            This callable should take a PyTree of the same shape as $x$ and
            an optional `args` argument, returning a PyTree of the same shape.
        """
        self.gamma = gamma
        self.u = u
        self.grad_f = grad_f

    def vf(
        self, t: RealScalarLike, y: UnderdampedLangevinTuple, args: Args
    ) -> UnderdampedLangevinTuple:
        x, v = y
        # gamma, u and v can all have different pytree structures, we only know that
        # gamma and u are prefixes of v (which is the same as x)

        gamma = broadcast_underdamped_langevin_arg(self.gamma, v, "gamma")
        u = broadcast_underdamped_langevin_arg(self.u, v, "u")

        def fun(_gamma, _u, _v, _f_x):
            return -_gamma * _v - _u * _f_x

        vf_x = v
        try:
            f_x = self.grad_f(x, args)  # Pass args to grad_f
            vf_v = jtu.tree_map(fun, gamma, u, v, f_x)
        except ValueError:
            raise RuntimeError(
                "The function `grad_f` in the Underdamped Langevin term must be"
                " a callable, whose input and output have the same PyTree structure"
                " and shapes as the position `x`."
            )
        vf_y = (vf_x, vf_v)
        return vf_y

    def contr(self, t0: RealScalarLike, t1: RealScalarLike, **kwargs) -> RealScalarLike:
        return t1 - t0

    def prod(
        self, vf: UnderdampedLangevinTuple, control: RealScalarLike
    ) -> UnderdampedLangevinTuple:
        return jtu.tree_map(lambda _vf: control * _vf, vf)


AbstractTerm.__module__ = "diffrax"
ODETerm.__module__ = "diffrax"
ControlTerm.__module__ = "diffrax"
WeaklyDiagonalControlTerm.__module__ = "diffrax"
MultiTerm.__module__ = "diffrax"
UnderdampedLangevinDriftTerm.__module__ = "diffrax"
UnderdampedLangevinDiffusionTerm.__module__ = "diffrax"
