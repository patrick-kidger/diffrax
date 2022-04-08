from typing import Sequence, Union

import jax.experimental.host_callback as hcb
import jax.lax as lax

from ..custom_types import Array, Int
from .unvmap import unvmap_any


def error_if(
    pred: Union[bool, Array[..., bool]],
    msg: str,
) -> bool:
    """For use as part of validating inputs.
    Works even under JIT.

    Example:
        @jax.jit
        def f(x):
            error_if(x < 0, "x must be >= 0")

        f(jax.numpy.array(-1))
    """
    branched_error_if(pred, 0, [msg])


def branched_error_if(
    pred: Union[bool, Array[..., bool]],
    index: Int,
    msgs: Sequence[str],
) -> bool:
    def raises(_index):
        raise RuntimeError(msgs[_index.item()])

    pred = unvmap_any(pred)
    lax.cond(pred, lambda: hcb.call(raises, index), lambda: None)
