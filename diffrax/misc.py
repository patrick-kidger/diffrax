import jax
import jax.numpy as jnp

from .custom_types import Array, PyTree


def _stack_pytrees(*arrays):
    return jnp.stack(arrays)


def stack_pytrees(pytrees: list[PyTree]) -> PyTree:
    return jax.tree_map(_stack_pytrees, *pytrees)


def safe_concatenate(arrays: list[Array]) -> Array:
    if len(arrays) == 0:
        return jnp.array([])
    return jnp.concatenate(arrays)


class _stable_method_hash:
    def __init__(self, __self__, __func__):
        super().__init__()
        self.__self__ = __self__
        self.__func__ = __func__

    def __call__(self, *args, **kwargs):
        return self.__func__(self.__self__, *args, **kwargs)

    def __hash__(self):
        return hash((self.__self__, self.__func__))

    def __eq__(self, other):
        try:
            other_self = other.__self__
            other_func = other.__func__
        except AttributeError:
            return False
        else:
            return self.__self__ == other_self and self.__func__ == other_func


class stable_method_hash:
    def __init__(self, func):
        super().__init__()
        self.func = func

    def __get__(self, instance, owner=None):
        if owner is None:
            return self
        return _stable_method_hash(instance, self)

    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)
