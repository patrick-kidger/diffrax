from typing import Generic, TypeVar

import jax


# RefHolder can be used for two things.
#
# (a) Outputing non-JAX types from a vmap'd operation. See JAX issue #7603.
# (b) Ensuring that something will be treated as a static argnum when using eqx.jitf.

_T = TypeVar("_T")


class RefHolder(Generic[_T]):
    def __init__(self, value: _T):
        self.value = value


def _refholder_flatten(self):
    return (), self.value


def _refholder_unflatten(value, _):
    return RefHolder(value)


jax.tree_util.register_pytree_node(RefHolder, _refholder_flatten, _refholder_unflatten)
