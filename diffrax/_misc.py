from collections.abc import Callable
from typing import Any, cast, Optional

import jax
import jax.core
import jax.lax as lax
import jax.numpy as jnp
import jax.tree_util as jtu
import optimistix as optx
from jaxtyping import Array, ArrayLike, PyTree, Shaped

from ._custom_types import BoolScalarLike, RealScalarLike


_itemsize_kind_type: dict[tuple[int, str], Any] = {
    (1, "i"): jnp.int8,
    (2, "i"): jnp.int16,
    (4, "i"): jnp.int32,
    (8, "i"): jnp.int64,
    (2, "f"): jnp.float16,
    (4, "f"): jnp.float32,
    (8, "f"): jnp.float64,
}


def force_bitcast_convert_type(val, new_type):
    val = jnp.asarray(val)
    intermediate_type = _itemsize_kind_type[new_type.dtype.itemsize, val.dtype.kind]
    val = val.astype(intermediate_type)
    return lax.bitcast_convert_type(val, new_type)


def _fill_forward(
    last_observed_yi: Shaped[Array, " *channels"], yi: Shaped[Array, " *channels"]
) -> tuple[Shaped[Array, " *channels"], Shaped[Array, " *channels"]]:
    yi = jnp.where(jnp.isnan(yi), last_observed_yi, yi)
    return yi, yi


@jax.jit
def fill_forward(
    ys: Shaped[Array, " times *channels"],
    replace_nans_at_start: Optional[Shaped[Array, " *channels"]] = None,
) -> Shaped[Array, " times *channels"]:
    """Fill-forwards over missing data (represented as NaN).

    By default it works its was along the "times" axis, filling in NaNs with the most
    recent non-NaN observation.

    The "channels" dimension is just for convenience, and the operation is essentially
    vmap'd over this dimension.

    Any NaNs at the start (with no previous non-NaN observation) may be left alone, or
    filled in, depending on `replace_nans_at_start`.

    **Arguments:**

    - `ys`: The data, which should use NaN to represent missing data.
    - `replace_nans_at_start`: Optional. If passed, used to fill-forward NaNs occuring
        at the start, prior to any non-NaN observations being made.

    **Returns:**

    The fill-forwarded data.
    """

    if replace_nans_at_start is None:
        y0 = ys[0]
    else:
        y0 = jnp.broadcast_to(replace_nans_at_start, ys[0].shape)
    _, ys = lax.scan(_fill_forward, y0, ys)
    return ys


def linear_rescale(t0, t, t1) -> Array:
    """Calculates (t - t0) / (t1 - t0), assuming t0 <= t <= t1.

    Specially handles the edge case t0 == t1:
        - zero is returned;
        - gradients through all three arguments are zero.
    """

    cond = t0 == t1
    numerator = cast(Array, jnp.where(cond, 0, t - t0))
    denominator = cast(Array, jnp.where(cond, 1, t1 - t0))
    return numerator / denominator


def adjoint_rms_seminorm(x: tuple[PyTree, PyTree, PyTree, PyTree]) -> RealScalarLike:
    """Defines an adjoint seminorm. This can frequently be used to increase the
    efficiency of backpropagation via [`diffrax.BacksolveAdjoint`][], as follows:

    ```python
    adjoint_controller = diffrax.PIDController(norm=diffrax.adjoint_rms_seminorm)
    adjoint = diffrax.BacksolveAdjoint(stepsize_controller=adjoint_controller)
    diffrax.diffeqsolve(..., adjoint=adjoint)
    ```

    Note that this means that any `stepsize_controller` specified for the forward pass
    will not be automatically used for the backward pass (as `adjoint_controller`
    overrides it), so you should specify any custom `rtol`, `atol` etc. for the
    backward pass as well.

    ??? cite "Reference"

        ```bibtex
        @article{kidger2021hey,
            author={Kidger, Patrick and Chen, Ricky T. Q. and Lyons, Terry},
            title={``{H}ey, that's not an {ODE}'': {F}aster {ODE} {A}djoints via
                   {S}eminorms},
            year={2021},
            journal={International Conference on Machine Learning}
        }
        ```
    """
    assert isinstance(x, tuple)
    assert len(x) == 4
    y, a_y, a_args, a_terms = x
    del a_args, a_terms  # whole point
    return optx.rms_norm((y, a_y))


def left_broadcast_to(arr, shape):
    """As `jax.numpy.broadcast_to`, except that `arr` is lined up with the left-hand
    edge of `shape`, rather than the right-hand edge.
    """

    indices = tuple(slice(None) if i < arr.ndim else None for i in range(len(shape)))
    return jnp.broadcast_to(arr[indices], shape)


def split_by_tree(key, tree, is_leaf: Optional[Callable[[PyTree], bool]] = None):
    """Like jax.random.split but accepts tree as a second argument and produces
    a tree of keys with the same structure.
    """
    treedef = jtu.tree_structure(tree, is_leaf=is_leaf)
    return jtu.tree_unflatten(treedef, jax.random.split(key, treedef.num_leaves))


def is_tuple_of_ints(obj):
    return isinstance(obj, tuple) and all(isinstance(x, int) for x in obj)


def static_select(pred: BoolScalarLike, a: ArrayLike, b: ArrayLike) -> ArrayLike:
    # This is mostly useful in that it doesn't promote `a` or `b` to Arrays when the
    # predicate is statically known.
    # This in turn allows us to perform some trace-time optimisations that XLA isn't
    # smart enough to do on its own.
    if (
        type(pred) is not bool
        and type(jax.core.get_aval(pred)) is jax.core.ConcreteArray
    ):
        with jax.ensure_compile_time_eval():
            pred = pred.item()
    if pred is True:
        return a
    elif pred is False:
        return b
    elif a is b:
        return a
    else:
        return lax.select(pred, a, b)


def upcast_or_raise(
    x: ArrayLike, array_for_dtype: ArrayLike, x_name: str, dtype_name: str
):
    """If `JAX_NUMPY_DTYPE_PROMOTION=strict`, then this will raise an error if
    `jnp.result_type(x, array_for_dtype)` is not the same as `array_for_dtype.dtype`.
    It will then cast `x` to `jnp.result_type(x, array_for_dtype)`.

    Thus if `JAX_NUMPY_DTYPE_PROMOTION=standard`, then the usual anything-goes behaviour
    will apply. If `JAX_NUMPY_DTYPE_PROMOTION=strict` then we loosen from prohibiting
    all dtype casting, to still allowing upcasting.
    """
    x_dtype = jnp.result_type(x)
    target_dtype = jnp.result_type(array_for_dtype)
    with jax.numpy_dtype_promotion("standard"):
        promote_dtype = jnp.result_type(x_dtype, target_dtype)
    config_value = jax.config.jax_numpy_dtype_promotion  # pyright: ignore
    if config_value == "strict":
        if target_dtype != promote_dtype:
            raise ValueError(
                f"When `JAX_NUMPY_DTYPE_PROMOTION=strict`, then {x_name} must have "
                f"a dtype that can be promoted to the dtype of {dtype_name}. "
                f"However {x_name} had dtype {x_dtype} and {dtype_name} had dtype "
                f"{target_dtype}."
            )
    elif config_value != "standard":
        assert False, f"Unrecognised `JAX_NUMPY_DTYPE_PROMOTION={config_value}`"
    return jnp.astype(x, promote_dtype)
