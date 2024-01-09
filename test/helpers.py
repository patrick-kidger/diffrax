from typing import Callable, Literal

import diffrax
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
import optimistix as optx
from jaxtyping import Array, PRNGKeyArray, PyTree, Shaped


all_ode_solvers = (
    diffrax.Bosh3(),
    diffrax.Dopri5(),
    diffrax.Dopri8(),
    diffrax.Euler(),
    diffrax.Ralston(),
    diffrax.Midpoint(),
    diffrax.Heun(),
    diffrax.LeapfrogMidpoint(),
    diffrax.ReversibleHeun(),
    diffrax.Tsit5(),
    diffrax.ImplicitEuler(),
    diffrax.Kvaerno3(),
    diffrax.Kvaerno4(),
    diffrax.Kvaerno5(),
)

all_split_solvers = (
    diffrax.Sil3(),
    diffrax.KenCarp3(),
    diffrax.KenCarp4(),
    diffrax.KenCarp5(),
)


def implicit_tol(solver):
    if isinstance(solver, diffrax.AbstractImplicitSolver):
        return eqx.tree_at(
            lambda s: (s.root_finder.rtol, s.root_finder.atol, s.root_finder.norm),
            solver,
            (1e-3, 1e-6, optx.rms_norm),
        )
    return solver


def random_pytree(key, treedef, dtype):
    keys = jr.split(key, treedef.num_leaves)
    leaves = []
    for key in keys:
        dimkey, sizekey, valuekey = jr.split(key, 3)
        num_dims = jr.randint(dimkey, (), 0, 5).item()
        dim_sizes = jr.randint(sizekey, (num_dims,), 0, 5)
        value = jr.normal(valuekey, tuple(dim_sizes.tolist()), dtype=dtype)
        leaves.append(value)
    return jtu.tree_unflatten(treedef, leaves)


treedefs = [
    jtu.tree_structure(x)
    for x in (
        0,
        None,
        {"a": [0, 0], "b": 0},
    )
]


def _no_nan(x):
    if eqx.is_array(x):
        return x.at[jnp.isnan(x)].set(8.9568)  # arbitrary magic value
    else:
        return x


def tree_allclose(x, y, *, rtol=1e-5, atol=1e-8, equal_nan=False):
    if equal_nan:
        x = jtu.tree_map(_no_nan, x)
        y = jtu.tree_map(_no_nan, y)
    return eqx.tree_equal(x, y, typematch=True, rtol=rtol, atol=atol)


def _path_l2_dist(
    ys1: PyTree[Shaped[Array, "repeats times ?*channels"], " T"],
    ys2: PyTree[Shaped[Array, "repeats times ?*channels"], " T"],
):
    # first compute the square of the difference and sum over
    # all but the first two axes (which represent the number of samples
    # and the length of saveat). Also sum all the PyTree leaves.
    def sum_square_diff(y1, y2):
        square_diff = jnp.square(y1 - y2)
        # sum all but the first two axes
        axes = range(2, square_diff.ndim)
        out = jnp.sum(square_diff, axis=axes)
        return out

    dist = jtu.tree_map(sum_square_diff, ys1, ys2)
    dist = sum(jtu.tree_leaves(dist))  # shape=(num_samples, len(saveat))
    dist = jnp.max(dist, axis=1)  # take sup along the length of integration
    dist = jnp.sqrt(jnp.mean(dist))
    return dist


@eqx.filter_jit
@eqx.filter_vmap(in_axes=(0, None, None, None, None, None, None, None, None, None))
def _batch_sde_solve(
    key: PRNGKeyArray,
    get_terms: Callable[[diffrax.AbstractBrownianPath], diffrax.AbstractTerm],
    levy_area: Literal["", "space-time"],
    solver: diffrax.AbstractSolver,
    w_shape: tuple[int, ...],
    t0: float,
    t1: float,
    dt0: float,
    y0: PyTree[Array],
    args: PyTree,
):
    # TODO: add a check whether the solver needs levy area
    dtype = jnp.result_type(*jtu.tree_leaves(y0))
    struct = jax.ShapeDtypeStruct(w_shape, dtype)
    bm = diffrax.VirtualBrownianTree(
        t0=t0,
        t1=t1,
        shape=struct,
        tol=2**-14,
        key=key,
        levy_area=levy_area,
    )
    terms = get_terms(bm)
    sol = diffrax.diffeqsolve(
        terms,
        solver,
        t0,
        t1,
        dt0=dt0,
        y0=y0,
        args=args,
        max_steps=None,
    )
    return sol.ys


def sde_solver_strong_order(
    get_terms: Callable[[diffrax.AbstractBrownianPath], diffrax.AbstractTerm],
    w_shape: tuple[int, ...],
    solver: diffrax.AbstractSolver,
    ref_solver: diffrax.AbstractSolver,
    t0: float,
    t1: float,
    dt_precise: float,
    y0: PyTree[Array],
    args: PyTree,
    num_samples: int,
    num_levels: int,
    key: PRNGKeyArray,
):
    dtype = jnp.result_type(*jtu.tree_leaves(y0))
    levy_area = ""  # TODO: add a check whether the solver needs levy area
    keys = jr.split(key, num_samples)  # deliberately reused across all solves

    correct_sols = _batch_sde_solve(
        keys,
        get_terms,
        levy_area,
        ref_solver,
        w_shape,
        t0,
        t1,
        dt_precise,
        y0,
        args,
    )
    dts = 2.0 ** jnp.arange(-3, -3 - num_levels, -1, dtype=dtype)

    @jax.jit
    @jax.vmap
    def get_single_err(dt):
        sols = _batch_sde_solve(
            keys,
            get_terms,
            levy_area,
            solver,
            w_shape,
            t0,
            t1,
            dt,
            y0,
            args,
        )
        return _path_l2_dist(sols, correct_sols)

    errs = get_single_err(dts)
    order, _ = jnp.polyfit(jnp.log(dts), jnp.log(errs), 1)
    return dts, errs, order
