import jax


def unvmap(x):
    while hasattr(x, "_trace") and isinstance(
        x._trace, jax.interpreters.batching.BatchTrace
    ):
        x = x.val
    return x
