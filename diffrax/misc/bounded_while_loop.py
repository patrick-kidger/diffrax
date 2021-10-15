import jax
import jax.lax as lax

from .unvmap import unvmap_any


def bounded_while_loop(cond_fun, body_fun, init_val, max_steps):
    """Reverse-mode autodifferentiable while loop.

    As `lax.while_loop`, except that it also has a `max_steps` argument bounding the
    maximum number of steps; after this the loop will exit unconditionally.
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
    def __init__(self, value):
        self.value = value


def _identity(x):
    return x


def _maybe(pred, fun, operand):
    def _make_update(_index, _operand, _new_operand):
        if _index is None:
            _keep = lambda a, b: lax.select(pred, a, b)
            return jax.tree_map(_keep, _new_operand, _operand)
        else:
            if isinstance(_index, Index):
                _index = _index.value
            _new_operand = lax.select(pred, _new_operand, _operand[_index])
            return _operand.at[_index].set(_new_operand)

    def _call(_operand):
        _new_operand, _index = fun(_operand)
        return jax.tree_map(
            _make_update, _index, _operand, _new_operand, is_leaf=lambda x: x is None
        )

    return _call(operand)


def _while_loop(cond_fun, body_fun, data, max_steps):
    if max_steps == 1:
        pred, val = data
        new_val = _maybe(pred, body_fun, val)
        return cond_fun(new_val), new_val
    else:

        def _call(_data):
            return _while_loop(cond_fun, body_fun, _data, max_steps // 2)

        def _scan_fn(_data, _):
            _pred, _ = _data
            _unvmap_pred = unvmap_any(_pred)
            return lax.cond(_unvmap_pred, _call, _identity, _data), None

        return lax.scan(_scan_fn, data, xs=None, length=2)[0]
