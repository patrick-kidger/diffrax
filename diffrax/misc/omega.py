import operator

import jax
import jax.numpy as jnp


class _Metaω(type):
    def __rpow__(cls, value):
        return cls(value)


class ω(metaclass=_Metaω):
    """Wraps jax.tree_map; e.g.

    ```python
    (ω(a) + ω(b)).ω == jax.tree_map(operator.add, a, b)
    ```

    Can also be initialised using the following __rpow__ syntax, which helps to
    minimise the number of brackets:

    ```python
    (a**ω + b**ω).ω == jax.tree_map(operator.add, a, b)
    ```
    """

    def __init__(self, value):
        self.ω = value

    def __getitem__(self, item):
        return jax.tree_map(lambda x: x[item], self.ω) ** ω

    def call(self, fn):
        return jax.tree_map(fn, self.ω) ** ω

    @property
    def at(self):
        return _ωUpdateHelper(self.ω)


def _set_op_main(base, name: str, op: callable) -> callable:
    def fn(self, other):
        if isinstance(other, ω):
            if jax.tree_structure(self.ω) != jax.tree_structure(other.ω):
                raise ValueError("PyTree structures must match.")
            return jax.tree_map(op, self.ω, other.ω) ** ω
        elif isinstance(other, (bool, complex, float, int, jnp.ndarray)):
            return jax.tree_map(lambda x: op(x, other), self.ω) ** ω
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
    def __init__(self, value):
        self.value = value

    def __getitem__(self, item):
        return _ωUpdateRef(self.value, item)


class _ωUpdateRef:
    def __init__(self, value, item):
        self.value = value
        self.item = item

    def get(self, **kwargs):
        value, item = self.ω
        return value.at[item].get(**kwargs)


def _set_op_at(base, name: str, op: callable) -> callable:
    def fn(self, other):
        if isinstance(other, ω):
            if jax.tree_structure(self.value) != jax.tree_structure(other.ω):
                raise ValueError("PyTree structures must match.")
            return (
                jax.tree_map(lambda x, y: op(x, self.item, y), self.value, other.ω) ** ω
            )
        elif isinstance(other, (bool, complex, float, int, jnp.ndarray)):
            return jax.tree_map(lambda x: op(x, self.item, other), self.value) ** ω
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
