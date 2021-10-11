import jax.lax as lax
import jax.numpy as jnp

from .unvmap import unvmap


def cond(pred, true_fun, false_fun, operand):
    """More-efficient version of `lax.cond`.

    The default `lax.cond` evaluates both branches unconditionally when wrapped in a
    `jax.vmap`. This version instead checks if pred is entirely true or entirely false
    (over the batch) and if the whole batch is true/false, only one branch is executed.
    """

    unvmap_pred = unvmap(pred)

    if unvmap_pred.ndim == 0:
        # Fast path
        return lax.cond(pred, true_fun, false_fun, operand)
    else:

        def _cond(op):
            return lax.cond(pred, true_fun, false_fun, operand)

        index = jnp.all(unvmap_pred).astype(int) + jnp.any(unvmap_pred)
        return lax.switch(index, [false_fun, _cond, true_fun], operand)
