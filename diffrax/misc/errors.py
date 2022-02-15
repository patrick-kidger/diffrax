from typing import Callable, Sequence, Type, Union

import jax
import jax.experimental.host_callback as hcb
import jax.numpy as jnp
import numpy as np

from ..custom_types import Array, Int
from .unvmap import unvmap_any


_Bool = Union[bool, Array[..., bool]]


def error_if(
    pred: Union[_Bool, Callable[[], _Bool]],
    msg: str,
    error_cls: Type[Exception] = ValueError,
) -> bool:
    """For use as part of validating inputs.
    Works even under JIT.

    Example:
        @jax.jit
        def f(x):
            error_if(x < 0, "x must be >= 0")

        f(jax.numpy.array(-1))
    """
    branched_error_if(pred, 0, [msg], error_cls)


def branched_error_if(
    pred: Union[_Bool, Callable[[], _Bool]],
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

    if callable(pred):
        with jax.ensure_compile_time_eval():
            pred = pred()

    if isinstance(pred, jnp.ndarray):
        with jax.ensure_compile_time_eval():
            pred = unvmap_any(pred)

    if isinstance(pred, jax.core.Tracer):
        hcb.call(raises, (pred, index))
    elif isinstance(pred, (bool, np.ndarray, jnp.ndarray)):
        raises((pred, index))
    else:
        msg = (
            "`pred` must either be a `bool`, a JAX array, or a zero-argument callable "
            "that returns a `bool` or JAX array, instead we got "
            f"value: {pred} of type: {type(pred)}."
        )
        assert False, msg
