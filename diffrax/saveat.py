import equinox as eqx
from typing import Optional

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
