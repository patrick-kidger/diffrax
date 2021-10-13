import jax
import jax.lax as lax

from .unvmap import unvmap_any


def _maybe(pred, fun1, fun2, operand):
    """Possibly executes an automorphic function.

    Morally speaking, this is a `lax.cond` with its falsey branch taken to be the
    identity. This version has several optimisations to work around limtiations of
    `vmap` and in-place updates, though, so as to get more efficient behaviour.

    Arguments:
        pred: boolean array.
        fun1: Function with type signature `a -> (a, b)` where `b` is a pytree prefix
              of `a`.
        fun2: Function with type signature `a -> a`.
        operand: PyTree with structure `a`.

    Warning:
        Neither `fun1` nor `fun2` are allowed to make in-place updates to their
        argument.

    Return value:
        In the simple case that `operand` and `update` are a single JAX array, then the
        result of this function is essentially that of:
        ```
        def _maybe(pred, fun1, fun2, operand):
            def _fun(o):
                update, index = fun1(o)
                if index is None:
                    p = update
                else:
                    p = o.at[index].set(update)
                return fun2(p)
            return lax.cond(pred, _fun, lambda x: x, operand)
        ```
        For general PyTrees then things are broadcast in the obvious way.

    Unlike the above simple implementation, this much more efficient when
    `jax.vmap`-ing:
    - If every batch element of `pred` is False then `fun` will not be executed at all.
      (The above simple implementation would run it and then throw everything away, as
      `vmap` of `lax.cond` is converted to a `lax.select`, which executes both branches
      unconditionally.)
    - In-place updates are handled specially to work around a limitation of the XLA
      compiler. (See JAX issue #8192.)
    """

    # Various important optimisations happening here:
    # - The `lax.cond` never gets turned into a `lax.select` under `vmap`, because its
    #   condition is always a scalar, thanks to unvmap'ing. This is what ensures `fun`
    #   is not executed when the whole batch of `pred` is `False`.
    # - `fun1` and `fun2` are only used once each. If they appeared multiple times then
    #   tracing the jaxpr for this function would involve tracing `fun` twice, which
    #   would imply exponential work when using _maybe recursively, in
    #   bounded_while_loop.
    # - The in-place updates never occur inside a `lax.select`, or inside a vmap'able
    #   `lax.cond` (which under `vmap` get turned into a `lax.select`). See JAX issue
    #   #8192: the XLA compiler is able to elide the copy (and make a true in-place
    #   update) when using `lax.cond`, but not `lax.select`.

    unvmap_pred = unvmap_any(pred)

    def _fun(_):
        update, index = fun1(operand)
        if jax.tree_structure(update) != jax.tree_structure(operand):
            raise RuntimeError(
                "The tree structures for `update` and `operand` must be identical."
            )
            # Moreover the tree structure for `index` must be a prefix, but that's
            # harder to check.
        get_index = lambda i, o: o if i is None else o[i]
        no_update = jax.tree_map(get_index, index, operand, is_leaf=lambda x: x is None)
        keep = lambda a, b: lax.select(pred, a, b)
        update = jax.tree_map(keep, update, no_update)
        make_update = lambda i, u, o: u if i is None else o.at[i].set(u)
        out = jax.tree_map(
            make_update, index, update, operand, is_leaf=lambda x: x is None
        )
        return fun2(out)

    return lax.cond(unvmap_pred, _fun, lambda _: operand, None)


def _fun(cond_fun, body_fun):
    def _fun1(data):
        pred, val = data
        update, index = body_fun(val)
        new_update = (pred, update)
        new_index = (None, index)
        return new_update, new_index

    def _fun2(data):
        _, val = data
        return cond_fun(val), val

    return _fun1, _fun2


def bounded_while_loop(cond_fun, body_fun, init_val, max_steps):
    """Reverse-mode autodifferentiable while loop.

    Roughly equivalent to `lax.while_loop`, although it has a slightly different type
    signature.

    Arguments:
        cond_fun: A function of type `a -> Bool`.
        body_fun: A function of type `a -> (a, b)`, where `b` is a prefix PyTree of
                  `a`. The implementation of `body_fun` must not make any in-place
                  updates to its argument.
        init_val: A value with PyTree structure `a`.
        max_steps: int.

    Note the return type for `body_fun`, which different to that used in
    `lax.while_loop`. In the simple case that `a` and `b` are both JAX arrays, then
    this is essentially equivalent to using
    ```
    def _body_fun(val):
        update, index, new_val = body_fun(val)
        new_val = new_val.at[index].set(update)
        return new_val
    ```
    as the `body_fun` of a `lax.while_loop`.

    This handling of in-place updates is needed for efficiency reasons. The
    implementation of `body_fun` must not make any in-place updates to its argument,
    and must instead use its second return value to accomplish this.

    Unlike `lax.while_loop`, it uses a `max_steps` argument, which bounds the maximum
    number of steps in the while loop; after this the loop will exit unconditionally.
    """

    if not isinstance(max_steps, int) or max_steps < 0:
        raise ValueError("max_steps must be a non-negative integer")
    if max_steps == 0:
        return init_val
    if max_steps == 1:
        return _maybe(cond_fun(init_val), body_fun, lambda x: x, init_val)
    if max_steps & (max_steps - 1) != 0:
        raise ValueError("max_steps must be a power of two")

    fun1, fun2 = _fun(cond_fun, body_fun)
    init_data = (cond_fun(init_val), init_val)
    _, init_val = _while_loop(fun1, fun2, init_data, max_steps)
    return init_val


def _while_loop(fun1, fun2, data, max_steps):
    assert max_steps > 1

    half_steps = max_steps // 2

    if half_steps == 1:
        _fun1 = fun1
        _fun2 = fun2
    else:

        def _fun1(_data):
            return _while_loop(fun1, fun2, _data, half_steps), None

        _fun2 = lambda x: x

    def _scan_fn(_data, _):
        _pred, _ = _data
        return _maybe(_pred, _fun1, _fun2, _data), None

    return lax.scan(_scan_fn, data, xs=None, length=2)[0]
