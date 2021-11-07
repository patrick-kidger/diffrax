from typing import Any, List, Optional, Tuple

import jax
import jax.lax as lax
import jax.numpy as jnp

from ..custom_types import Array, PyTree, Scalar
from .ravel import ravel_pytree


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


def _stack_pytrees(*arrays):
    return jnp.stack(arrays)


def stack_pytrees(pytrees: List[PyTree]) -> PyTree:
    return jax.tree_map(_stack_pytrees, *pytrees)


def is_perturbed(x: Any) -> bool:
    if isinstance(x, jax.ad.JVPTracer):
        return True
    elif isinstance(x, jax.core.Tracer):
        return any(is_perturbed(attr) for name, attr in x._contents())
    else:
        return False


def check_no_derivative(x: Array, name: str) -> None:
    if is_perturbed(x):
        raise ValueError(f"Cannot differentiate {name}.")


class ContainerMeta(type):
    def __new__(cls, name, bases, dict):
        assert "_reverse_lookup" not in dict
        _dict = {}
        _reverse_lookup = {}
        i = 0
        for key, value in dict.items():
            if key.startswith("__") and key.endswith("__"):
                _dict[key] = value
            else:
                _dict[key] = i
                _reverse_lookup[i] = value
                i += 1
        _dict["_reverse_lookup"] = _reverse_lookup
        return super().__new__(cls, name, bases, _dict)

    def __getitem__(cls, item):
        return cls._reverse_lookup[item]

    def __len__(cls):
        return len(cls._reverse_lookup)


def _fill_forward(
    last_observed_yi: Array["channels"], yi: Array["channels"]  # noqa: F821
) -> Tuple[Array["channels"], Array["channels"]]:  # noqa: F821
    yi = jnp.where(jnp.isnan(yi), last_observed_yi, yi)
    return yi, yi


@jax.jit
def fill_forward(
    ys: Array["times", "channels"],  # noqa: F821
    replace_nans_at_start: Optional[Array["channels"]] = None,  # noqa: F821
) -> Array["times, channels"]:  # noqa: F821
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


@jax.custom_jvp
def nextafter(x: Array) -> Array:
    y = jnp.nextafter(x, jnp.inf)
    # Flush denormal to normal.
    # Our use for these is to handle jumps in the vector field. Typically that means
    # there will be an "if x > cond" condition somewhere. However JAX uses DAZ
    # (denormals-are-zero), which will cause this check to fail near zero:
    # `jnp.nextafter(0, jnp.inf) > 0` gives `False`.
    return jnp.where(x == 0, jnp.finfo(x.dtype).tiny, y)


nextafter.defjvps(lambda x_dot, _, __: x_dot)


@jax.custom_jvp
def nextbefore(x: Array) -> Array:
    y = jnp.nextafter(x, jnp.NINF)
    return jnp.where(x == 0, -jnp.finfo(x.dtype).tiny, y)


nextbefore.defjvps(lambda x_dot, _, __: x_dot)


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
    x, _ = ravel_pytree(x)
    if x.size == 0:
        return 0
    sqnorm = jnp.mean(x ** 2)
    cond = sqnorm == 0
    # Double-where trick to avoid NaN gradients.
    # See JAX issues #5039 and #1052.
    _sqnorm = jnp.where(cond, 1.0, sqnorm)
    return jnp.where(cond, 0.0, jnp.sqrt(_sqnorm))
