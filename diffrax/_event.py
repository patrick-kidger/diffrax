from collections.abc import Callable
from typing import Optional, Union

import equinox as eqx
import optimistix as optx
from jaxtyping import PyTree

from ._custom_types import BoolScalarLike, RealScalarLike


class Event(eqx.Module):
    cond_fn: PyTree[Callable[..., Union[BoolScalarLike, RealScalarLike]]]
    root_finder: Optional[optx.AbstractRootFinder] = None
