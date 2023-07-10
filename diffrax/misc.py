from typing import Callable, Optional, Tuple, Union

import jax
import jax.core
import jax.flatten_util as fu
import jax.lax as lax
import jax.numpy as jnp
import jax.tree_util as jtu
from jax.typing import ArrayLike

from .custom_types import Array, PyTree, Scalar


_itemsize_kind_type = {
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
    last_observed_yi: Array["channels":...], yi: Array["channels":...]  # noqa: F821
) -> Tuple[Array["channels":...], Array["channels":...]]:  # noqa: F821
    yi = jnp.where(jnp.isnan(yi), last_observed_yi, yi)
    return yi, yi


@jax.jit
def fill_forward(
    ys: Array["times", ...],  # noqa: F821
    replace_nans_at_start: Optional[Array[...]] = None,  # noqa: F821
) -> Array["times", ...]:  # noqa: F821
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


def linear_rescale(t0, t, t1):
    """Calculates (t - t0) / (t1 - t0), assuming t0 <= t <= t1.

    Specially handles the edge case t0 == t1:
        - zero is returned;
        - gradients through all three arguments are zero.
    """

    cond = t0 == t1
    numerator = jnp.where(cond, 0, t - t0)
    denominator = jnp.where(cond, 1, t1 - t0)
    return numerator / denominator


def rms_norm(x: PyTree) -> Scalar:
    x, _ = fu.ravel_pytree(x)
    if x.size == 0:
        return 0
    return _rms_norm(x)


@jax.custom_jvp
def _rms_norm(x):
    x_sq = jnp.real(x * jnp.conj(x))
    return jnp.sqrt(jnp.mean(x_sq))


@_rms_norm.defjvp
def _rms_norm_jvp(x, tx):
    (x,) = x
    (tx,) = tx
    out = _rms_norm(x)
    # Get zero gradient, rather than NaN gradient, in these cases
    pred = (out == 0) | jnp.isinf(out)
    numerator = jnp.where(pred, 0, x)
    denominator = jnp.where(pred, 1, out * x.size)
    t_out = jnp.dot(numerator / denominator, tx)
    return out, t_out


def adjoint_rms_seminorm(x: Tuple[PyTree, PyTree, PyTree, PyTree]) -> Scalar:
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
    return rms_norm((y, a_y))


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


def static_select(pred: Union[bool, Array], a: ArrayLike, b: ArrayLike) -> ArrayLike:
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
    else:
        return lax.select(pred, a, b)
