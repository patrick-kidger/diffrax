from dataclasses import dataclass
from typing import Optional

from .custom_types import Array


@dataclass(frozen=True)
class SaveAt:
    t0: bool = False
    t1: bool = False
    t: Optional[Array["times"]] = None  # noqa: F821
    steps: bool = False
    controller_state: bool = False
    solver_state: bool = False
