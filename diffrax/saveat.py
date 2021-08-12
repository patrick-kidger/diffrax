from typing import Optional

import equinox as eqx
import jax.numpy as jnp

from .custom_types import Array


class _SaveAt(eqx.Module):
    t0: bool = False
    t1: bool = False
    steps: bool = False
    controller_state: bool = False
    solver_state: bool = False
    dense: bool = False


class SaveAt(_SaveAt):
    t: Optional[Array["times"]] = None  # noqa: F821

    def __init__(self, t=None, **kwargs):
        super().__init__(**kwargs)
        if t is not None:
            t = jnp.asarray(t)
        self.t = t

    def __post_init__(self):
        if (
            not self.t0
            and not self.t1
            and self.t is None
            and not self.steps
            and not self.dense
        ):
            raise ValueError("Empty saveat -- nothing will be saved.")
