from dataclasses import dataclass, fields
import jax


# dataclasses.astuple operates recursively, which destroys information about
# nested tree_dataclasses. This is just a shallow tuplification.
def _dataclass_astuple(datacls):
    return tuple(getattr(datacls, field.name) for field in fields(datacls))


def tree_dataclass(cls: type):
    cls = dataclass(frozen=True)(cls)

    def flatten(self):
        return _dataclass_astuple(self), None

    def unflatten(_, fields):
        return cls(*fields)

    jax.tree_util.register_pytree_node(cls, flatten, unflatten)

    return cls
