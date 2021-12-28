import jax
import jax.lax as lax

from .unvmap import unvmap_any


def bounded_while_loop(cond_fun, body_fun, init_val, max_steps):
    """Reverse-mode autodifferentiable while loop.

    Mostly as `lax.while_loop`, with a few small changes.

    Arguments:
        cond_fun: function `a -> a`
        body_fun: function `a -> (a, b)`, where `b` is a pytree prefix of `a`.
        init_val: pytree with structure `a`.
        max_steps: integer or `None`.

    Limitation:
        `body_fun` should not make any in-place updates to its argument. When running
        on the CPU then the XLA compiler will fail to treat these as in-place and will
        make a copy every time. (See JAX issue #8192.)

        It is for this reason that `body_fun` returns a second argument. In the simple
        case that `a` is a single JAX array, it is semantically equivalent to using
        ```
        def _body_fun(val):
            new_val, index = body_fun(val)
            if index is None:
                return new_val
            else:
                return new_val.at[index].set(val)
        ```
        as the `body_fun` of a `lax.while_loop`. (And in the general case things are
        tree-map'd in the obvious way.)

        Internally, `bounded_while_loop` will treat things so as to work around this
        limitation of XLA.

        If this extra `index` returned needs to be a tuple, then it should be wrapped
        into a `diffrax.utils.Index`: i.e. `return new_val, Index(index)`. This is
        because a tuple is a pytree, and then `b` will not be a pytree prefix of `a`.

    Note the extra `max_steps` argument. If this is `None` then `bounded_while_loop`
    will fall back to `lax.while_loop` (which is not reverse-mode autodifferentiable).
    If it is a non-negative integer then this is the maximum number of steps which may
    be taken in the loop, after which the loop will exit unconditionally.
    """

    if max_steps is None:

        def _make_update(_index, _val, _new_val):
            if _index is None:
                return _new_val
            else:
                if isinstance(_index, Index):
                    _index = _index.value
                return _val.at[_index].set(_new_val)

        def _body_fun(_val):
            _new_val, _index = body_fun(_val)
            return jax.tree_map(
                _make_update, _index, _val, _new_val, is_leaf=lambda x: x is None
            )

        return lax.while_loop(cond_fun, _body_fun, init_val)

    if not isinstance(max_steps, int) or max_steps < 0:
        raise ValueError("max_steps must be a non-negative integer")
    if max_steps == 0:
        return init_val
    if max_steps & (max_steps - 1) != 0:
        raise ValueError("max_steps must be a power of two")

    init_data = (cond_fun(init_val), init_val)
    _, val = _while_loop(cond_fun, body_fun, init_data, max_steps)
    return val


class Index:
    """Used with diffrax.utils.bounded_while_loop."""

    def __init__(self, value):
        self.value = value


def _identity(x):
    return x


def _while_loop(cond_fun, body_fun, data, max_steps):
    if max_steps == 1:
        pred, val = data

        def _make_update(_index, _val, _new_val):
            if _index is None:
                _keep = lambda a, b: lax.select(pred, a, b)
                return jax.tree_map(_keep, _new_val, _val)
            else:
                # This is the reason for the in-place update being returned from
                # body_fun, rather than happening within it. We need to `lax.select`
                # the update-to-make, not the updated buffer. (The latter results in
                # XLA:CPU failing to determine that the buffer can be updated in-place;
                # instead it makes a copy. c.f. JAX issue #8192.)
                if isinstance(_index, Index):
                    _index = _index.value
                _new_val = lax.select(pred, _new_val, _val[_index])
                return _val.at[_index].set(_new_val)

        new_val, index = body_fun(val)
        new_val = jax.tree_map(
            _make_update, index, val, new_val, is_leaf=lambda x: x is None
        )
        return cond_fun(new_val), new_val
    else:

        def _call(_data):
            return _while_loop(cond_fun, body_fun, _data, max_steps // 2)

        def _scan_fn(_data, _):
            _pred, _ = _data
            _unvmap_pred = unvmap_any(_pred)
            return lax.cond(_unvmap_pred, _call, _identity, _data), None

        return lax.scan(_scan_fn, data, xs=None, length=2)[0]