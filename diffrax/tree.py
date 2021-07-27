from dataclasses import dataclass, fields
import jax
import jax.numpy as jnp
import math
import numpy as np
from typing import Tuple

from .custom_types import Array, PyTree, SquashTreeDef
from .misc import safe_concatenate


def tree_squash(tree: PyTree) -> Tuple[Array, SquashTreeDef]:
    flat, treedef = jax.tree_flatten(tree)
    if len(flat) == 1:
        # Optimised no-copying case
        shapes = None
        splits = None
        treedef = (treedef, shapes, splits)
        flat = flat[0]
    else:
        shapes = tuple(flat_i.shape for flat_i in flat)
        splits = tuple(np.array([math.prod(shape) for shape in shapes[:-1]]).cumsum())
        flat = [flat_i.flatten() for flat_i in flat]
        flat = safe_concatenate(flat)
        treedef = SquashTreeDef(treedef, shapes, splits)
    return flat, treedef


def tree_unsquash(treedef: SquashTreeDef, flat: Array) -> PyTree:
    treedef, shapes, splits = treedef
    if shapes is None:
        flat = [flat]
    else:
        flat = jnp.split(flat, splits)
        flat = [flat_i.reshape(shape) for flat_i, shape in zip(flat, shapes)]
    return jax.tree_unflatten(treedef, flat)


# dataclasses.astuple operates recursively, which destroys information about
# nested tree_dataclasses. This is just a shallow tuplification.
def _dataclass_astuple(datacls):
    return tuple(getattr(datacls, field.name) for field in fields(datacls))


def tree_dataclass(cls: type):
    datacls = dataclass(frozen=True)(cls)

    def flatten(self):
        return _dataclass_astuple(self), None

    def unflatten(_, fields):
        return cls(*fields)

    jax.tree_util.register_pytree_node(datacls, flatten, unflatten)

    return datacls


class tree_method:
    def __init__(self, method):
        super().__init__()
        self.method = method

    def __get__(self, instance, owner=None):
        if instance is None:
            return self
        return _bound_tree_method(instance, self)

    def __call__(self, *args, **kwargs):
        return self.method(*args, **kwargs)


class _bound_tree_method:
    def __init__(self, __self__, tree_method):
        super().__init__()
        # Match the bound method API
        self.__self__ = __self__
        self.__func__ = tree_method.method

        # Usually just equivalent to __func__.__name__
        # However this guards against funny behaviour like
        #
        # @tree_method
        # def f(...): ...
        #
        # class X:
        #     name_that_isnt_f = f
        names = (
            name for cls in self.__self__.__class__.__mro__ for name,
            value in cls.__dict__.items() if value is tree_method
        )
        self._name = next(names)  # any name will do

    def __call__(self, *args, **kwargs):
        return self.__func__(self.__self__, *args, **kwargs)


def _flatten_btm(self):
    return (self.__self__,), self._name


def _unflatten_btm(name, instance):
    (instance,) = instance
    return getattr(instance, name)


jax.tree_util.register_pytree_node(_bound_tree_method, _flatten_btm, _unflatten_btm)
