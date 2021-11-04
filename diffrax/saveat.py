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
    ts: Optional[Array["times"]] = None  # noqa: F821

    def __init__(self, *, ts=None, **kwargs):
        super().__init__(**kwargs)
        if ts is not None:
            ts = jnp.asarray(ts)
        self.ts = ts

        if (
            not self.t0
            and not self.t1
            and self.ts is None
            and not self.steps
            and not self.dense
        ):
            raise ValueError("Empty saveat -- nothing will be saved.")
