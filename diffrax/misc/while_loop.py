import functools as ft

from .cond import cond


def _identity(val):
    return val


def while_loop(cond_fun, body_fun, init_val, max_steps):
    """Reverse-mode autodifferentiable while loop.

    As `lax.while_loop`. Additionally requires a `max_steps` argument, which bounds the
    maximum number of steps in the while loop; after this it will exit unconditionally.

    If the typical number of steps is smaller than `max_steps` then this should be more
    efficient than simply using `lax.scan`.
    """
    if not isinstance(max_steps, int):
        raise ValueError("max_steps must be an integer")
    if max_steps < 0:
        raise ValueError("max_steps must be a positive integer")
    elif max_steps == 0:
        return init_val
    elif max_steps == 1:
        return cond(cond_fun(init_val), body_fun, _identity, init_val)
    else:
        left_steps = max_steps // 2
        right_steps = max_steps - left_steps
        val = while_loop(cond_fun, body_fun, init_val, left_steps)
        _while_loop = ft.partial(while_loop, cond_fun, body_fun, max_steps=right_steps)
        return cond(cond_fun(val), _while_loop, _identity, val)
