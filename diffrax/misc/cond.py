import jax.lax as lax
import jax.numpy as jnp

from .unvmap import unvmap


def cond(pred, true_fun, false_fun, operand):
    """More-efficient version of `lax.cond`.

    The default `lax.cond` evaluates both branches unconditionally when wrapped in a
    `jax.vmap`. This version instead checks if pred is entirely true or entirely false
    (over the batch) and if the whole batch is true/false, only one branch is executed.
    """

    # Note that the two extra lax.conds introduced here are genuine conditional
    # computation: their predicates are always scalars, even under vmap.

    unvmap_pred = unvmap(pred)

    def _cond(op):
        return lax.cond(cond, true_fun, false_fun, operand)

    def __cond(op):
        return lax.cond(jnp.all(~unvmap_pred), false_fun, _cond, operand)

    return lax.cond(jnp.all(unvmap_pred), true_fun, __cond, operand)
