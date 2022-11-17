import functools as ft
import math

import equinox.internal as eqxi
import jax
import jax.lax as lax
import jax.numpy as jnp
import jax.tree_util as jtu


def bounded_while_loop(cond_fun, body_fun, init_val, max_steps, base=16):
    """Reverse-mode autodifferentiable while loop.

    Mostly as `lax.while_loop`, with a few small changes.

    Arguments:
        cond_fun: function `a -> bool`
        body_fun: function `a -> a`.
        init_val: pytree of type `a`.
        max_steps: integer or `None`.
        base: integer.

    Note the extra `max_steps` argument. If this is `None` then `bounded_while_loop`
    will fall back to `lax.while_loop` (which is not reverse-mode autodifferentiable).
    If it is a non-negative integer then this is the maximum number of steps which may
    be taken in the loop, after which the loop will exit unconditionally.

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
    _, val, _ = _while_loop(_cond_fun, body_fun, init_data, rounded_max_steps, base)
    return val


def _while_loop(cond_fun, body_fun, data, max_steps, base):
    if max_steps == 1:
        pred, val, step = data
        new_val = body_fun(val)
        new_val = jtu.tree_map(ft.partial(lax.select, pred), new_val, val)
        new_step = step + 1
        return cond_fun(new_val, new_step), new_val, new_step
    else:

        def _call(_data):
            return _while_loop(cond_fun, body_fun, _data, max_steps // base, base)

        def _scan_fn(_data, _):
            _pred, _, _ = _data
            _unvmap_pred = eqxi.unvmap_any(_pred)
            return lax.cond(_unvmap_pred, _call, lambda x: x, _data), None

        # Don't put checkpointing on the lowest level
        if max_steps != base:
            _scan_fn = jax.checkpoint(_scan_fn, prevent_cse=False)

        return lax.scan(_scan_fn, data, xs=None, length=base)[0]
