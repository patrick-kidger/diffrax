import jax.lax as lax

from .cond import maybe


def _fun(cond_fun, body_fun):
    def __fun(data):
        _, val = data
        val = body_fun(val)
        return cond_fun(val), val

    return __fun


def while_loop(cond_fun, body_fun, init_val, max_steps):
    """Reverse-mode autodifferentiable while loop.

    As `lax.while_loop`. Additionally requires a `max_steps` argument, which bounds the
    maximum number of steps in the while loop; after this it will exit unconditionally.

    If the typical number of steps is smaller than `max_steps` then this should be more
    efficient than simply using `lax.scan`.
    """

    if not isinstance(max_steps, int) or max_steps < 0:
        raise ValueError("max_steps must be a non-negative integer")
    if max_steps == 0:
        return init_val
    if max_steps == 1:
        return maybe(cond_fun(init_val), body_fun, init_val)
    if max_steps & (max_steps - 1) != 0:
        raise ValueError("max_steps must be a power of two")

    fun = _fun(cond_fun, body_fun)
    init_data = (cond_fun(init_val), init_val)
    _, init_val = _while_loop(fun, init_data, max_steps)
    return init_val


def _while_loop(fun, data, max_steps):
    assert max_steps > 1

    half_steps = max_steps // 2

    if half_steps == 1:
        _call = fun
    else:

        def _call(_data):
            return _while_loop(fun, _data, half_steps)

    def _scan_fn(_data, _):
        _pred, _val = _data
        return maybe(_pred, _call, _data), None

    return lax.scan(_scan_fn, data, xs=None, length=2)[0]
