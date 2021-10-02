from typing import List, Tuple

import jax
import jax.lax as lax
import jax.numpy as jnp

from ..custom_types import Array, PyTree, Scalar
from .ravel import ravel_pytree


def _stack_pytrees(*arrays):
    return jnp.stack(arrays)


def stack_pytrees(pytrees: List[PyTree]) -> PyTree:
    return jax.tree_map(_stack_pytrees, *pytrees)


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


def _fill_forward(
    last_observed_yi: Array["channels"], yi: Array["channels"]  # noqa: F821
) -> Tuple[Array["channels"], Array["channels"]]:  # noqa: F821
    yi = jnp.where(jnp.isnan(yi), last_observed_yi, yi)
    return yi, yi


@jax.jit
def fill_forward(
    ys: Array["times", "channels"]  # noqa: F821
) -> Array["times, channels"]:  # noqa: F821
    _, ys = lax.scan(_fill_forward, ys[0], ys)
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
    # Assumes t0 <= t <= t1

    cond = t0 == t1
    # `where` to avoid NaN gradients
    div = jnp.where(cond, 1, t1 - t0)
    out = (t - t0) / div
    # `where` to get correct gradient if `cond` is True.
    out = jnp.where(cond, t, out)
    return out


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


def copy_docstring_from(source):
    def _copy_docstring_from(target):
        target.__doc__ = source.__doc__
        return target

    return _copy_docstring_from
