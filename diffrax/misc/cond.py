import jax
import jax.lax as lax
import jax.numpy as jnp

from .unvmap import unvmap_all, unvmap_any


# TODO: factor these out into their own package?


def _empty(struct):
    return jnp.zeros(struct.shape, struct.dtype)


# TODO: keep or not? Diffrax doesn't use it since `maybe` was introduced.
def cond(pred, true_fun, false_fun, operand):
    """vmap-efficient version of `lax.cond`.

    The default `lax.cond` evaluates both branches unconditionally when wrapped in a
    `jax.vmap`. This version instead checks if pred is entirely true or entirely false
    (over the batch) and if the whole batch is true/false, only one branch is executed.

    This version will be slower for non-vmap cases though (or vmap cases where the
    above optimisation doesn't help much).
    """

    # Note that this implementation is such that true_fun and false_fun only appear
    # once when traced into jaxpr. This is important for efficient compilation, in
    # particular for the custom while_loop also in this folder. (Nested cond calls
    # would otherwise produce an exponentially-sized jaxpr in the nesting depth.)

    _unvmap_all = unvmap_all(pred)
    _unvmap_any = unvmap_any(pred)

    # TODO: is this the most efficient implementation of this behaviour?
    true_shape = jax.eval_shape(true_fun, operand)
    false_shape = jax.eval_shape(false_fun, operand)
    true_dummy = jax.tree_map(_empty, true_shape)
    false_dummy = jax.tree_map(_empty, false_shape)

    true_vals = lax.cond(_unvmap_any, true_fun, lambda _: true_dummy, operand)
    false_vals = lax.cond(_unvmap_all, lambda _: false_dummy, false_fun, operand)
    keep = lambda a, b: jnp.where(pred, a, b)
    return jax.tree_map(keep, true_vals, false_vals)


def maybe(pred, fun, operand):
    """Possibly executes an automorphic function.

    Functionally equivalent to
    ```
    lax.cond(pred, fun, lambda x: x, operand)
    ```
    but is (a) more efficient, and (b) unlike `lax.cond`, has better semantics with
    `vmap`: if the whole batch of `pred` is `False` then `fun` will not be run.
    """

    # As with cond, this implementation only uses the jaxpr for `fun` once.

    _unvmap_any = unvmap_any(pred)

    def _fun(_):
        out = fun(operand)
        keep = lambda a, b: lax.select(pred, a, b)
        return jax.tree_map(keep, out, operand)

    return lax.cond(_unvmap_any, _fun, lambda _: operand, None)
