import dataclasses
from typing import Callable, Literal

import diffrax
import equinox as eqx
import jax
import jax.random as jr
import jax.tree_util as jtu
import optimistix as optx
from diffrax import (
    AbstractBrownianPath,
    AbstractTerm,
    ConstantStepSize,
    diffeqsolve,
    SaveAt,
    UnsafeBrownianPath,
    VirtualBrownianTree,
)
from diffrax._custom_types import RealScalarLike
from jax import numpy as jnp
from jaxtyping import PyTree


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


def path_l2_dist(ys1: PyTree[jax.Array], ys2: PyTree[jax.Array]):
    # first compute the square of the difference and sum over
    # all but the first two axes (which represent the number of samples
    # and the length of saveat). Also sum all the PyTree leaves
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


@dataclasses.dataclass
class SDE:
    get_terms: Callable[[AbstractBrownianPath], AbstractTerm]
    args: PyTree
    y0: PyTree
    t0: RealScalarLike
    t1: RealScalarLike
    w_shape: tuple[int]

    def get_dtype(self):
        return jnp.result_type(*jtu.tree_leaves(self.y0))

    def get_bm(
        self,
        key,
        levy_area: Literal["", "space-time"] = "space-time",
        use_tree=True,
        tol=2**-14,
    ):
        shp_dtype = jax.ShapeDtypeStruct(self.w_shape, dtype=self.get_dtype())
        if use_tree:
            return VirtualBrownianTree(
                t0=self.t0,
                t1=self.t1,
                shape=shp_dtype,
                tol=tol,
                key=key,
                levy_area=levy_area,
            )
        else:
            return UnsafeBrownianPath(shape=shp_dtype, key=key, levy_area=levy_area)


def batch_sde_solve(
    keys,
    sde: SDE,
    dt0,
    solver,
    stepsize_controller=ConstantStepSize(),
    levy_area: Literal["", "space-time"] = "space-time",
):
    _saveat = SaveAt(ts=[sde.t1])

    # TODO: add a check whether the solver needs levy area

    def end_value(key):
        path = sde.get_bm(key, levy_area=levy_area, use_tree=True)
        terms = sde.get_terms(path)

        sol = diffeqsolve(
            terms,
            solver,
            sde.t0,
            sde.t1,
            dt0=dt0,
            y0=sde.y0,
            args=sde.args,
            saveat=_saveat,
            stepsize_controller=stepsize_controller,
            max_steps=None,
        )
        return sol.ys

    return jax.vmap(end_value)(keys)


def sde_solver_order(keys, sde: SDE, solver, ref_solver, dt_precise, hs_num=5, hs=None):
    dtype = sde.get_dtype()
    need_stla = False  # TODO: add a check whether the solver needs levy area
    levy_area: Literal["", "space-time"] = "space-time" if need_stla else ""

    correct_sols = batch_sde_solve(
        keys, sde, dt_precise, ref_solver, levy_area=levy_area
    )
    if hs is None:
        hs = jnp.power(2.0, jnp.arange(-3, -3 - hs_num, -1, dtype=dtype))

    def get_single_err(h):
        sols = batch_sde_solve(keys, sde, h, solver, levy_area=levy_area)
        return path_l2_dist(sols, correct_sols)

    errs = jax.vmap(get_single_err)(hs)
    order, _ = jnp.polyfit(jnp.log(hs), jnp.log(errs), 1)
    return hs, errs, order
