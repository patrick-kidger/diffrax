import functools as ft
from collections.abc import Callable
from typing import Any, ClassVar, Literal, Union

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, PyTree, Scalar

from .._custom_types import Aux, Fn
from .._root_find import AbstractRootFinder
from .._solution import RESULTS


class _BisectionState(eqx.Module, strict=True):
    lower: Scalar
    upper: Scalar
    flip: Bool[Array, ""]
    error: Float[Array, ""]


class _ExpansionCarry(eqx.Module, strict=True):
    lower: Scalar
    upper: Scalar


def _interval_contains_root(
    carry: _ExpansionCarry,
    /,
    *,
    need_positive: Bool[Array, ""],
    expand_upper: Bool[Array, ""],
    fn: Fn[Scalar, Scalar, Aux],
    args: Any,
) -> Bool[Array, ""]:
    new_boundary = jnp.where(expand_upper, carry.upper, carry.lower)
    carry_val, _ = fn(new_boundary, args)
    return need_positive ^ (carry_val > 0.0)


def _expand_interval(
    carry: _ExpansionCarry,
    /,
    *,
    expand_upper: Bool[Array, ""],
) -> _ExpansionCarry:
    new_domain = 2.0 * (carry.upper - carry.lower)
    new_lower = jnp.where(expand_upper, carry.upper, carry.lower - new_domain)
    new_upper = jnp.where(expand_upper, carry.upper + new_domain, carry.lower)
    return _ExpansionCarry(new_lower, new_upper)


def _expand_interval_repeatedly(
    lower: Scalar,
    upper: Scalar,
    *,
    upper_val: Scalar,
    lower_val: Scalar,
    need_positive: Bool[Array, ""],
    fn: Fn[Scalar, Scalar, Any],
    args: PyTree,
) -> tuple[Scalar, Scalar]:
    initial_interval = _ExpansionCarry(lower, upper)
    expand_upper = need_positive ^ (upper_val < lower_val)
    cond_fun = ft.partial(
        _interval_contains_root,
        need_positive=need_positive,
        expand_upper=expand_upper,
        fn=fn,
        args=args,
    )
    body_fun = ft.partial(_expand_interval, expand_upper=expand_upper)
    final_interval = jax.lax.while_loop(cond_fun, body_fun, initial_interval)
    lower = final_interval.lower
    upper = final_interval.upper
    return lower, upper


class Bisection(AbstractRootFinder[Scalar, Scalar, Aux, _BisectionState], strict=True):
    """The bisection method of root finding. This may only be used with functions
    `R->R`, i.e. functions with scalar input and scalar output.

    This requires the following `options`:

    - `lower`: The lower bound on the interval which contains the root.
    - `upper`: The upper bound on the interval which contains the root.

    Which are passed as, for example,
    `optimistix.root_find(..., options=dict(lower=0, upper=1))`

    This algorithm works by considering the interval `[lower, upper]`, checking the
    sign of the evaluated function at the midpoint of the interval, and then keeping
    whichever half contains the root. This is then repeated. The iteration stops once
    the interval is sufficiently small.

    If `expand_if_necessary` and `detect` are true, the initial interval will be
    expanded if it doesn't contain the the root.  This expansion assumes that the
    function is monotonic.
    """

    rtol: float
    atol: float
    flip: Union[bool, Literal["detect"]] = "detect"
    expand_if_necessary: bool = False
    # All norms are the same for scalars.
    norm: ClassVar[Callable[[PyTree], Scalar]] = jnp.abs

    def init(
        self,
        fn: Fn[Scalar, Scalar, Aux],
        y: Scalar,
        args: PyTree,
        options: dict[str, Any],
        f_struct: jax.ShapeDtypeStruct,
        aux_struct: PyTree[jax.ShapeDtypeStruct],
        tags: frozenset[object],
    ) -> _BisectionState:
        lower = jnp.asarray(options["lower"], f_struct.dtype)
        upper = jnp.asarray(options["upper"], f_struct.dtype)
        del options, aux_struct
        if jnp.shape(y) != () or jnp.shape(lower) != () or jnp.shape(upper) != ():
            raise ValueError(
                "Bisection can only be used to find the roots of a function taking a "
                "scalar input."
            )
        if not isinstance(f_struct, jax.ShapeDtypeStruct) or f_struct.shape != ():
            raise ValueError(
                "Bisection can only be used to find the roots of a function producing "
                "a scalar output."
            )
        if isinstance(self.flip, bool):
            # Make it possible to avoid the extra two function compilations.
            flip = jnp.array(self.flip)
        elif self.flip == "detect":
            lower_val, _ = fn(lower, args)
            upper_val, _ = fn(upper, args)
            flip = lower_val > upper_val
            if self.expand_if_necessary:
                lower, upper = _expand_interval_repeatedly(
                    lower,
                    upper,
                    upper_val=upper_val,
                    lower_val=lower_val,
                    need_positive=lower_val < 0.0,
                    fn=fn,
                    args=args,
                )
            else:
                lower_neg = lower_val < 0
                upper_neg = upper_val < 0
                root_not_contained = lower_neg == upper_neg
                flip = eqx.error_if(
                    flip,
                    root_not_contained,
                    msg="The root is not contained in [lower, upper]",
                )
        else:
            raise ValueError("`flip` may only be True, False, or 'detect'.")
        return _BisectionState(
            lower=lower,
            upper=upper,
            flip=flip,
            error=jnp.array(jnp.inf, f_struct.dtype),
        )

    def step(
        self,
        fn: Fn[Scalar, Scalar, Aux],
        y: Scalar,
        args: PyTree,
        options: dict[str, Any],
        state: _BisectionState,
        tags: frozenset[object],
    ) -> tuple[Scalar, _BisectionState, Aux]:
        del options
        error, aux = fn(y, args)
        negative = state.flip ^ (error < 0)
        new_lower = jnp.where(negative, y, state.lower)
        new_upper = jnp.where(negative, state.upper, y)
        new_y = new_lower + 0.5 * (new_upper - new_lower)
        new_state = _BisectionState(
            lower=new_lower, upper=new_upper, flip=state.flip, error=error
        )
        return new_y, new_state, aux

    def terminate(
        self,
        fn: Fn[Scalar, Scalar, Aux],
        y: Scalar,
        args: PyTree,
        options: dict[str, Any],
        state: _BisectionState,
        tags: frozenset[object],
    ) -> tuple[Bool[Array, ""], RESULTS]:
        del fn, args, options
        scale = self.atol + self.rtol * jnp.abs(y)
        y_small = jnp.abs(state.lower - state.upper) < scale
        f_small = jnp.abs(state.error) < self.atol
        return y_small & f_small, RESULTS.successful

    def postprocess(
        self,
        fn: Fn[Scalar, Scalar, Aux],
        y: Scalar,
        aux: Aux,
        args: PyTree,
        options: dict[str, Any],
        state: _BisectionState,
        tags: frozenset[object],
        result: RESULTS,
    ) -> tuple[Scalar, Aux, dict[str, Any]]:
        return y, aux, {}


Bisection.__init__.__doc__ = """**Arguments:**

- `rtol`: Relative tolerance for terminating solve.
- `atol`: Absolute tolerance for terminating solve.
- `flip`: Can be set to any of:
    - `False`: specify that `fn(lower, args) < 0 < fn(upper, args)`.
    - `True`: specify that `fn(lower, args) > 0 > fn(upper, args)`.
    - `"detect"`: automatically check `fn(lower, args)` and `fn(upper, args)`. Note that
        this option may increase both runtime and compilation time.
- `expand_if_necessary`: If `True` then the `lower` and `upper` passed as options will
    be made larger or smaller if the root is not found within the interval
    `[lower, upper]`. To locate the root correctly then this assumes that the function
    is monotonic.
"""
