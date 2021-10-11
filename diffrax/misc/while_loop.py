import jax.lax as lax

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
    if max_steps == 0 or (max_steps & (max_steps - 1) != 0):
        raise ValueError("max_steps must be a power of two")  # TODO; relax this
    return _while_loop(cond_fun, body_fun, init_val, max_steps)


def _while_loop(cond_fun, body_fun, init_val, max_steps):
    if max_steps == 0:
        return init_val
    elif max_steps == 1:
        return cond(cond_fun(init_val), body_fun, _identity, init_val)
    else:
        half_steps = max_steps // 2

        def _while(val):
            return _while_loop(cond_fun, body_fun, val, half_steps)

        def _scan_fn(val, _):
            pred = cond_fun(val)
            return cond(pred, _while, _identity, val), None

        return lax.scan(_scan_fn, init_val, xs=None, length=2)[0]
