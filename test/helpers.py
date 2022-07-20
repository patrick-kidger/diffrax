import functools as ft
import gc
import operator
import time

import diffrax
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jrandom


all_ode_solvers = (
    diffrax.Bosh3(scan_stages=False),
    diffrax.Bosh3(scan_stages=True),
    diffrax.Dopri5(scan_stages=False),
    diffrax.Dopri5(scan_stages=True),
    diffrax.Dopri8(scan_stages=False),
    diffrax.Dopri8(scan_stages=True),
    diffrax.Euler(),
    diffrax.Ralston(scan_stages=False),
    diffrax.Ralston(scan_stages=True),
    diffrax.Midpoint(scan_stages=False),
    diffrax.Midpoint(scan_stages=True),
    diffrax.Heun(scan_stages=False),
    diffrax.Heun(scan_stages=True),
    diffrax.LeapfrogMidpoint(),
    diffrax.ReversibleHeun(),
    diffrax.Tsit5(scan_stages=False),
    diffrax.Tsit5(scan_stages=True),
    diffrax.ImplicitEuler(),
    diffrax.Kvaerno3(scan_stages=False),
    diffrax.Kvaerno3(scan_stages=True),
    diffrax.Kvaerno4(scan_stages=False),
    diffrax.Kvaerno4(scan_stages=True),
    diffrax.Kvaerno5(scan_stages=False),
    diffrax.Kvaerno5(scan_stages=True),
)


def implicit_tol(solver):
    if isinstance(solver, diffrax.AbstractImplicitSolver):
        return eqx.tree_at(
            lambda s: (s.nonlinear_solver.rtol, s.nonlinear_solver.atol),
            solver,
            (1e-3, 1e-6),
            is_leaf=lambda x: x is None,
        )
    return solver


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
        0,
        None,
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
