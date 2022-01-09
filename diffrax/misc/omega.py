import operator
from typing import Optional

import jax
import jax.numpy as jnp


class _Metaω(type):
    def __rpow__(cls, value):
        return cls(value)


class ω(metaclass=_Metaω):
    """Provides friendlier syntax for mapping with `jax.tree_map`.

    !!! example

        ```python
        (ω(a) + ω(b)).ω == jax.tree_map(operator.add, a, b)
        ```

    !!! tip

        To minimise the number of brackets, the following `__rpow__` syntax can be
        used:

        ```python
        (a**ω + b**ω).ω == jax.tree_map(operator.add, a, b)
        ```

        This is entirely equivalent to the above.
    """

    def __init__(self, value, is_leaf=None):
        """
        **Arguments:**

        - `value`: The PyTree to wrap.
        - `is_leaf`: An optional value for the `is_leaf` argument to `jax.tree_map`.

        !!! note

            The `is_leaf` argument cannot be set when using the `__rpow__` syntax for
            initialisation.
        """
        self.ω = value
        self.is_leaf = is_leaf

    def __getitem__(self, item):
        return ω(
            jax.tree_map(lambda x: x[item], self.ω, is_leaf=self.is_leaf),
            is_leaf=self.is_leaf,
        )

    def call(self, fn):
        return ω(jax.tree_map(fn, self.ω, is_leaf=self.is_leaf), is_leaf=self.is_leaf)

    @property
    def at(self):
        return _ωUpdateHelper(self.ω, self.is_leaf)


def _equal_code(fn1: Optional[callable], fn2: Optional[callable]):
    """Checks whether fn1 and fn2 both have the same code.

    It's essentially impossible to see if two functions are equivalent, so this won't,
    and isn't intended, to catch every possible difference between fn1 and fn2. But it
    should at least catch the common case that `is_leaf` is specified for one input and
    not specified for the other.
    """
    sentinel1 = object()
    sentinel2 = object()
    code1 = getattr(getattr(fn1, "__code__", sentinel1), "co_code", sentinel2)
    code2 = getattr(getattr(fn2, "__code__", sentinel1), "co_code", sentinel2)
    return type(code1) == type(code2) and code1 == code2


def _set_op_main(base, name: str, op: callable) -> callable:
    def fn(self, other):
        if isinstance(other, ω):
            if jax.tree_structure(self.ω) != jax.tree_structure(other.ω):
                raise ValueError("PyTree structures must match.")
            if not _equal_code(self.is_leaf, other.is_leaf):
                raise ValueError("`is_leaf` must match.")
            return ω(
                jax.tree_map(op, self.ω, other.ω, is_leaf=self.is_leaf),
                is_leaf=self.is_leaf,
            )
        elif isinstance(other, (bool, complex, float, int, jnp.ndarray)):
            return ω(
                jax.tree_map(lambda x: op(x, other), self.ω, is_leaf=self.is_leaf),
                is_leaf=self.is_leaf,
            )
        else:
            raise RuntimeError("Type of `other` not understood.")

    fn.__name__ = name
    fn.__qualname__ = base.__qualname__ + "." + name
    setattr(base, name, fn)


for (name, op) in [
    ("__add__", operator.add),
    ("__radd__", lambda x, y: y + x),
    ("__sub__", operator.sub),
    ("__rsub__", lambda x, y: y - x),
    ("__mul__", operator.mul),
    ("__rmul__", lambda x, y: y * x),
    ("__truediv__", operator.truediv),
    ("__rtruediv__", lambda x, y: y / x),
    ("__floordiv__", operator.floordiv),
    ("__rfloordiv__", lambda x, y: y // x),
    ("__matmul__", operator.matmul),
    ("__rmatmul__", lambda x, y: y @ x),
    ("__eq__", operator.eq),
    ("__ne__", operator.ne),
]:
    _set_op_main(ω, name, op)


class _ωUpdateHelper:
    def __init__(self, value, is_leaf):
        self.value = value
        self.is_leaf = is_leaf

    def __getitem__(self, item):
        return _ωUpdateRef(self.value, item, self.is_leaf)


class _ωUpdateRef:
    def __init__(self, value, item, is_leaf):
        self.value = value
        self.item = item
        self.is_leaf = is_leaf

    def get(self, **kwargs):
        value, item = self.ω
        return value.at[item].get(**kwargs)


def _set_op_at(base, name: str, op: callable) -> callable:
    def fn(self, other):
        if isinstance(other, ω):
            if jax.tree_structure(self.value) != jax.tree_structure(other.ω):
                raise ValueError("PyTree structures must match.")
            if not _equal_code(self.is_leaf, other.is_leaf):
                raise ValueError("is_leaf specifications must match.")
            return ω(
                jax.tree_map(
                    lambda x, y: op(x, self.item, y),
                    self.value,
                    other.ω,
                    is_leaf=self.is_leaf,
                ),
                is_leaf=self.is_leaf,
            )
        elif isinstance(other, (bool, complex, float, int, jnp.ndarray)):
            return ω(
                jax.tree_map(
                    lambda x: op(x, self.item, other), self.value, is_leaf=self.is_leaf
                ),
                is_leaf=self.is_leaf,
            )
        else:
            raise RuntimeError("Type of `other` not understood.")

    fn.__name__ = name
    fn.__qualname__ = base.__qualname__ + "." + name
    setattr(base, name, fn)


for (name, op) in [
    ("set", lambda x, y, z, **kwargs: x.at[y].set(z, **kwargs)),
    ("add", lambda x, y, z, **kwargs: x.at[y].add(z, **kwargs)),
    ("multiply", lambda x, y, z, **kwargs: x.at[y].multiply(z, **kwargs)),
    ("divide", lambda x, y, z, **kwargs: x.at[y].divide(z, **kwargs)),
    ("power", lambda x, y, z, **kwargs: x.at[y].power(z, **kwargs)),
    ("min", lambda x, y, z, **kwargs: x.at[y].min(z, **kwargs)),
    ("max", lambda x, y, z, **kwargs: x.at[y].max(z, **kwargs)),
]:
    _set_op_at(_ωUpdateRef, name, op)
