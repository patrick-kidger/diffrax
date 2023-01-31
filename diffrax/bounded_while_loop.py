import functools as ft
import math
from typing import Any, Callable, Optional, Union

import equinox as eqx
import equinox.internal as eqxi
import jax
import jax.lax as lax
import jax.numpy as jnp
import jax.tree_util as jtu


def bounded_while_loop(
    cond_fun,
    body_fun,
    init_val,
    max_steps: Optional[int],
    *,
    buffers: Optional[Callable] = None,
    base: int = 16
):
    """Reverse-mode autodifferentiable while loop.

    This only exists to support a few edge cases:
    - forward-mode autodiff;
    - reading from `buffers`.
    You should almost always prefer to use `equinox.internal.checkpointed_while_loop`
    instead.

    Once 'bloops' land in JAX core then this function will be removed.

    **Arguments:**

    - cond_fun: function `a -> bool`.
    - body_fun: function `a -> a`.
    - init_val: pytree of type `a`.
    - max_steps: integer or `None`.
    - buffers: function `a -> node or nodes`.
    - base: integer.

    Note the extra `max_steps` argument. If this is `None` then `bounded_while_loop`
    will fall back to `lax.while_loop` (which is not reverse-mode autodifferentiable).
    If it is a non-negative integer then this is the maximum number of steps which may
    be taken in the loop, after which the loop will exit unconditionally.

    Note the extra `buffers` argument. This behaves similarly to the same argument for
    `equinox.internal.checkpointed_while_loop`: these support efficient in-place updates
    but no operation. (Unlike `checkpointed_while_loop`, however, this supports being
    read from.)

    Note the extra `base` argument.
    - Run time will increase slightly as `base` increases.
    - Compilation time will decrease substantially as
      `math.ceil(math.log(max_steps, base))` decreases. (Which happens as `base`
      increases.)
    """

    init_val = jtu.tree_map(jnp.asarray, init_val)

    if max_steps is None:
        return lax.while_loop(cond_fun, body_fun, init_val)

    if not isinstance(max_steps, int) or max_steps < 0:
        raise ValueError("max_steps must be a non-negative integer")
    if max_steps == 0:
        return init_val

    def _cond_fun(val, step):
        return cond_fun(val) & (step < max_steps)

    init_data = (cond_fun(init_val), init_val, 0)
    rounded_max_steps = base ** int(math.ceil(math.log(max_steps, base)))
    if buffers is None:
        buffers = lambda _: ()
    _, val, _ = _while_loop(
        _cond_fun, body_fun, init_data, rounded_max_steps, buffers, base
    )
    return val


def _while_loop(cond_fun, body_fun, data, max_steps, buffers, base):
    if max_steps == 1:
        pred, val, step = data

        tag = object()

        def _buffers(v):
            nodes = buffers(v)
            tree = jtu.tree_map(_unwrap_buffers, nodes, is_leaf=_is_buffer)
            return jtu.tree_leaves(tree)

        val = eqx.tree_at(
            _buffers, val, replace_fn=ft.partial(_Buffer, _pred=pred, _tag=tag)
        )
        new_val = body_fun(val)
        if jax.eval_shape(lambda: val) != jax.eval_shape(lambda: new_val):
            raise ValueError("body_fun must have matching input and output structures")

        def _is_our_buffer(x):
            return isinstance(x, _Buffer) and x._tag is tag

        def _unwrap_or_select(new_v, v):
            if _is_our_buffer(new_v):
                assert _is_our_buffer(v)
                assert eqx.is_array(new_v._array)
                assert eqx.is_array(v._array)
                return new_v._array
            else:
                return lax.select(pred, new_v, v)

        new_val = jtu.tree_map(_unwrap_or_select, new_val, val, is_leaf=_is_our_buffer)
        new_step = step + 1
        return cond_fun(new_val, new_step), new_val, new_step
    else:

        def _call(_data):
            return _while_loop(
                cond_fun, body_fun, _data, max_steps // base, buffers, base
            )

        def _scan_fn(_data, _):
            _pred, _, _ = _data
            _unvmap_pred = eqxi.unvmap_any(_pred)
            return lax.cond(_unvmap_pred, _call, lambda x: x, _data), None

        # Don't put checkpointing on the lowest level
        if max_steps != base:
            _scan_fn = jax.checkpoint(_scan_fn, prevent_cse=False)

        return lax.scan(_scan_fn, data, xs=None, length=base)[0]


def _is_buffer(x):
    return isinstance(x, _Buffer)


def _unwrap_buffers(x):
    while _is_buffer(x):
        x = x._array
    return x


class _Buffer(eqx.Module):
    _array: Union[jnp.ndarray, "_Buffer"]
    _pred: jnp.ndarray
    _tag: object = eqx.static_field()

    def __getitem__(self, item):
        return self._array[item]

    def _set(self, pred, item, x):
        pred = pred & self._pred
        if isinstance(self._array, _Buffer):
            array = self._array._set(pred, item, x)
        else:
            old_x = self._array[item]
            x = jnp.where(pred, x, old_x)
            array = self._array.at[item].set(x)
        return _Buffer(array, self._pred, self._tag)

    @property
    def at(self):
        return _BufferAt(self)

    @property
    def shape(self):
        return self._array.shape

    @property
    def dtype(self):
        return self._array.dtype

    @property
    def size(self):
        return self._array.size


class _BufferAt(eqx.Module):
    _buffer: _Buffer

    def __getitem__(self, item):
        return _BufferItem(self._buffer, item)


class _BufferItem(eqx.Module):
    _buffer: _Buffer
    _item: Any

    def set(self, x):
        return self._buffer._set(True, self._item, x)
