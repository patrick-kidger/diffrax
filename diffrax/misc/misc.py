from typing import Optional, Tuple

import jax
import jax.flatten_util as fu
import jax.lax as lax
import jax.numpy as jnp

from ..custom_types import Array, PyTree, Scalar


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


class ContainerMeta(type):
    def __new__(cls, name, bases, dict):
        assert "reverse_lookup" not in dict
        _dict = {}
        reverse_lookup = []
        i = 0
        for key, value in dict.items():
            if key.startswith("__") and key.endswith("__"):
                _dict[key] = value
            else:
                _dict[key] = i
                reverse_lookup.append(value)
                i += 1
        _dict["reverse_lookup"] = reverse_lookup
        return super().__new__(cls, name, bases, _dict)

    def __instancecheck__(cls, instance):
        return isinstance(instance, int) or super().__instancecheck__(instance)

    def __getitem__(cls, item):
        return cls.reverse_lookup[item]

    def __len__(cls):
        return len(cls.reverse_lookup)


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
    sqnorm = jnp.mean(x**2)
    cond = sqnorm == 0
    # Double-where trick to avoid NaN gradients.
    # See JAX issues #5039 and #1052.
    _sqnorm = jnp.where(cond, 1.0, sqnorm)
    return jnp.where(cond, 0.0, jnp.sqrt(_sqnorm))


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
