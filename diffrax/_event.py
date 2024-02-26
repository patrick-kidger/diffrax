from collections.abc import Callable
from typing import Optional, Union

import equinox as eqx
import optimistix as optx
from jaxtyping import ArrayLike, PyTree

from ._custom_types import BoolScalarLike, RealScalarLike


class EventFn(eqx.Module):
    cond_fn: Callable[..., Union[BoolScalarLike, RealScalarLike]]
    transition_fn: Optional[Callable[[PyTree[ArrayLike]], PyTree[ArrayLike]]] = (
        lambda x: x
    )


class Event(eqx.Module):
    event_fn: PyTree[EventFn]
    root_finder: Optional[optx.AbstractRootFinder] = None
