import functools as ft
import gc
import operator
import time

import diffrax
import jax
import jax.numpy as jnp
import jax.random as jrandom


all_ode_solvers = (
    diffrax.bosh3,
    diffrax.dopri5,
    diffrax.dopri8,
    diffrax.euler,
    diffrax.fehlberg2,
    diffrax.heun,
    diffrax.leapfrog_midpoint,
    diffrax.reversible_heun,
    diffrax.tsit5,
    diffrax.implicit_euler,
    diffrax.kvaerno3,
    diffrax.kvaerno4,
    diffrax.kvaerno5,
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
        [0],
        {"a": 0},
        {"a": 0, "b": 0},
        {"a": [0, 0], "b": 0},
    )
]


def tree_allclose(x, y, **kwargs):
    same_structure = jax.tree_structure(x) == jax.tree_structure(y)
    allclose = ft.partial(jnp.allclose, **kwargs)
    return same_structure and jax.tree_reduce(
        operator.and_, jax.tree_map(allclose, x, y)
    )


def time_fn(fn, repeat=1):
    fn()  # Compile
    if gc_enabled := gc.isenabled():
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
