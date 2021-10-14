import jax.lax as lax

from .unvmap import unvmap_any


def _identity(x):
    return x


def _maybe(pred, fun, operand):
    """Possibly executes an automorphic function.

    Morally speaking, this is a `lax.cond` with its falsey branch taken to be the
    identity. This version has several optimisations to work around limtiations of
    `vmap` and in-place updates, though, so as to get more efficient behaviour.

    Arguments:
        pred: boolean array.
        fun: Function with type signature `a -> a`..
        operand: PyTree with structure `a`.

    Warning:
        If `fun` makes any in-place updates to its argument then this will not be
        picked up on when running the CPU. (See JAX issue #8192, which affects
        `lax.cond` as well.)

    Return value:
        As `lax.cond(pred, fun, lambda x: x, operand)`.

    Unlike the above simple implementation, then in addition if every batch element of
    `pred` is `False` then `fun` will not be executed at all. This makes it more
    efficient when `jax.vmap`-ing. (Which normally executes both branches
    unconditionally when vmap'ing.)
    """

    unvmap_pred = unvmap_any(pred)

    def _call(x):
        return lax.cond(pred, fun, _identity, x)

    return lax.cond(unvmap_pred, _call, _identity, operand)


def _fun(cond_fun, body_fun):
    def __fun(data):
        pred, val = data
        new_val = body_fun(val)
        return cond_fun(new_val), new_val

    return __fun


def bounded_while_loop(cond_fun, body_fun, init_val, max_steps):
    """Reverse-mode autodifferentiable while loop.

    As `lax.while_loop`, except that it also has a `max_steps` argument bounding the
    maximum number of steps; after this the loop will exit unconditionally.
    """

    if max_steps is None:
        return lax.while_loop(cond_fun, body_fun, init_val)
    if not isinstance(max_steps, int) or max_steps < 0:
        raise ValueError("max_steps must be a non-negative integer")
    if max_steps == 0:
        return init_val
    if max_steps == 1:
        return _maybe(cond_fun(init_val), body_fun, init_val)
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
        _pred, _ = _data
        return _maybe(_pred, _call, _data), None

    return lax.scan(_scan_fn, data, xs=None, length=2)[0]
