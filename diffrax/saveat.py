from typing import Optional

import equinox as eqx

from .custom_types import Array
from .misc import vmap_any


class SaveAt(eqx.Module):
    t0: bool = False
    t1: bool = False
    t: Optional[Array["times"]] = None  # noqa: F821
    steps: bool = False
    controller_state: bool = False
    solver_state: bool = False
    dense: bool = False

    def __post_init__(self):
        if self.t is not None and vmap_any(self.t[1:] >= self.t[:-1]):
            raise ValueError("saveat.t must be strictly increasing.")
        if (
            not self.t0
            and not self.t1
            and self.t is None
            and not self.steps
            and not self.dense
        ):
            raise ValueError("Empty saveat -- nothing will be saved.")
