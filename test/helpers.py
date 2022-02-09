import functools as ft
import gc
import operator
import time

import diffrax
import jax
import jax.numpy as jnp
import jax.random as jrandom


all_ode_solvers = (
    diffrax.Bosh3,
    diffrax.Dopri5,
    diffrax.Dopri8,
    diffrax.Euler,
    diffrax.Ralston,
    diffrax.Midpoint,
    diffrax.Fehlberg2,
    diffrax.Heun,
    diffrax.LeapfrogMidpoint,
    diffrax.ReversibleHeun,
    diffrax.Tsit5,
    diffrax.ImplicitEuler,
    diffrax.Kvaerno3,
    diffrax.Kvaerno4,
    diffrax.Kvaerno5,
)


def random_pytree(key, treedef):
    keys = jrandom.split(key, treedef.num_leaves)
    leaves = []
    for key in keys:
        dimkey, sizekey, valuekey = jrandom.split(key, 3)
        num_dims = jrandom.randint(dimkey, (), 0, 5)
        dim_sizes = jrandom.randint(sizekey, (num_dims,), 0, 5)
        value = jrandom.normal(valuekey, dim_sizes)
        leaves.append(value)
    return jax.tree_unflatten(treedef, leaves)


treedefs = [
    jax.tree_structure(x)
    for x in (
        None,
        0,
        {"a": [0, 0], "b": 0},
    )
]


def _shaped_allclose(x, y, **kwargs):
    return jnp.shape(x) == jnp.shape(y) and jnp.allclose(x, y, **kwargs)


def shaped_allclose(x, y, **kwargs):
    """As `jnp.allclose`, except:
    - It also supports PyTree arguments.
    - It mandates that shapes match as well (no broadcasting)
    """
    same_structure = jax.tree_structure(x) == jax.tree_structure(y)
    allclose = ft.partial(_shaped_allclose, **kwargs)
    return same_structure and jax.tree_util.tree_reduce(
        operator.and_, jax.tree_map(allclose, x, y), True
    )


def time_fn(fn, repeat=1):
    fn()  # Compile
    gc_enabled = gc.isenabled()
    if gc_enabled:
        gc.collect()
    gc.disable()
    try:
        times = []
        for _ in range(repeat):
            start = time.perf_counter_ns()
            fn()
            end = time.perf_counter_ns()
            times.append(end - start)
        return min(times)
    finally:
        if gc_enabled:
            gc.enable()
            gc.collect()
