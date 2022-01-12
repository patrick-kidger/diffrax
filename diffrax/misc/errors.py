from typing import Sequence, Type, Union

import jax.experimental.host_callback as hcb
import jax.numpy as jnp

from ..custom_types import Array, Int
from .unvmap import unvmap_any


def error_if(
    pred: Union[bool, Array[..., bool]],
    msg: str,
    error_cls: Type[Exception] = ValueError,
) -> bool:
    """For use as part of validating inputs.

    Example:
        def f(x):
            cond = cond_fn(x)
            error_if(cond)
    """
    branched_error_if(pred, 0, [msg], error_cls)


def branched_error_if(
    pred: Union[bool, Array[..., bool]],
    index: Int,
    msgs: Sequence[str],
    error_cls: Type[Exception] = ValueError,
) -> bool:
    def raises(_arg):
        _pred, _index = _arg
        if _pred:
            if isinstance(_index, jnp.ndarray):
                _index = _index.item()
            raise error_cls(msgs[_index])

    if isinstance(pred, jnp.ndarray):
        pred = unvmap_any(pred)
        hcb.call(raises, (pred, index))
    elif isinstance(pred, bool):
        raises((pred, index))
    else:
        assert False, "`pred` must either be a `bool` or a JAX array."
