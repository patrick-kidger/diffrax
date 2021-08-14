from typing import List

import jax
import jax.numpy as jnp

from ..custom_types import PyTree


def _stack_pytrees(*arrays):
    return jnp.stack(arrays)


def stack_pytrees(pytrees: List[PyTree]) -> PyTree:
    return jax.tree_map(_stack_pytrees, *pytrees)


class ContainerMeta(type):
    def __new__(cls, name, bases, dict):
        assert "_reverse_lookup" not in dict
        dict["_reverse_lookup"] = {value: key for key, value in dict.items()}
        # Check that all values are unique
        assert (
            len(dict) == len(dict["_reverse_lookup"]) + 1
        )  # +1 for dict['_reverse_lookup'] itself.
        return super().__new__(cls, name, bases, dict)

    def __getitem__(cls, item):
        return cls._reverse_lookup[item]
