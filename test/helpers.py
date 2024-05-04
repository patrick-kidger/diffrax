import dataclasses
from typing import Callable, Optional, Union

import diffrax
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
import optimistix as optx
from diffrax import (
    AbstractBrownianPath,
    AbstractTerm,
    ControlTerm,
    MultiTerm,
    ODETerm,
    VirtualBrownianTree,
)
from jax import Array
from jaxtyping import PRNGKeyArray, PyTree, Shaped


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


def path_l2_dist(
    ys1: PyTree[Shaped[Array, "repeats times ?*channels"], " T"],
    ys2: PyTree[Shaped[Array, "repeats times ?*channels"], " T"],
):
    # first compute the square of the difference and sum over
    # all but the first two axes (which represent the number of samples
    # and the length of saveat). Also sum all the PyTree leaves.
    def sum_square_diff(y1, y2):
        with jax.numpy_dtype_promotion("standard"):
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


def _get_minimal_la(solver):
    while isinstance(solver, diffrax.HalfSolver):
        solver = solver.solver
    return getattr(solver, "minimal_levy_area", diffrax.BrownianIncrement)


def _abstract_la_to_la(abstract_la):
    if issubclass(abstract_la, diffrax.AbstractSpaceTimeTimeLevyArea):
        return diffrax.SpaceTimeTimeLevyArea
    elif issubclass(abstract_la, diffrax.AbstractSpaceTimeLevyArea):
        return diffrax.SpaceTimeLevyArea
    elif issubclass(abstract_la, diffrax.AbstractBrownianIncrement):
        return diffrax.BrownianIncrement
    else:
        raise ValueError(f"Unknown levy area {abstract_la}")


@eqx.filter_jit
@eqx.filter_vmap(
    in_axes=(0, None, None, None, None, None, None, None, None, None, None, None, None)
)
def _batch_sde_solve(
    key: PRNGKeyArray,
    get_terms: Callable[[diffrax.AbstractBrownianPath], diffrax.AbstractTerm],
    w_shape: tuple[int, ...],
    t0: float,
    t1: float,
    y0: PyTree[Array],
    args: PyTree,
    solver: diffrax.AbstractSolver,
    levy_area: Optional[type[diffrax.AbstractBrownianIncrement]],
    dt0: Optional[float],
    controller: Optional[diffrax.AbstractStepSizeController],
    bm_tol: float,
    saveat: diffrax.SaveAt,
):
    abstract_levy_area = _get_minimal_la(solver) if levy_area is None else levy_area
    concrete_la = _abstract_la_to_la(abstract_levy_area)
    dtype = jnp.result_type(*jtu.tree_leaves(y0))
    struct = jax.ShapeDtypeStruct(w_shape, dtype)
    bm = diffrax.VirtualBrownianTree(
        t0=t0,
        t1=t1,
        shape=struct,
        tol=bm_tol,
        key=key,
        levy_area=concrete_la,  # pyright: ignore
    )
    terms = get_terms(bm)
    if controller is None:
        controller = diffrax.ConstantStepSize()
    sol = diffrax.diffeqsolve(
        terms,
        solver,
        t0,
        t1,
        dt0=dt0,
        y0=y0,
        args=args,
        max_steps=2**19,
        stepsize_controller=controller,
        saveat=saveat,
    )
    return sol.ys, sol.stats["num_accepted_steps"]


def _resulting_levy_area(
    levy_area1: type[diffrax.AbstractBrownianIncrement],
    levy_area2: type[diffrax.AbstractBrownianIncrement],
) -> type[diffrax.AbstractBrownianIncrement]:
    """A helper that returns the stricter Levy area.

    **Arguments:**

    - `levy_area1`: The first Levy area type.
    - `levy_area2`: The second Levy area type.

    **Returns:**

    `BrownianIncrement`, `SpaceTimeLevyArea`, or `SpaceTimeTimeLevyArea`.
    """
    if issubclass(levy_area1, diffrax.AbstractSpaceTimeTimeLevyArea) or issubclass(
        levy_area2, diffrax.AbstractSpaceTimeTimeLevyArea
    ):
        return diffrax.SpaceTimeTimeLevyArea
    elif issubclass(levy_area1, diffrax.AbstractSpaceTimeLevyArea) or issubclass(
        levy_area2, diffrax.AbstractSpaceTimeLevyArea
    ):
        return diffrax.SpaceTimeLevyArea
    elif issubclass(levy_area1, diffrax.AbstractBrownianIncrement) or issubclass(
        levy_area2, diffrax.AbstractBrownianIncrement
    ):
        return diffrax.BrownianIncrement
    else:
        raise ValueError("Invalid levy area types.")


@eqx.filter_jit
def sde_solver_strong_order(
    keys: PRNGKeyArray,
    get_terms: Callable[[diffrax.AbstractBrownianPath], diffrax.AbstractTerm],
    w_shape: tuple[int, ...],
    t0: float,
    t1: float,
    y0: PyTree[Array],
    args: PyTree,
    solver: diffrax.AbstractSolver,
    ref_solver: diffrax.AbstractSolver,
    levels: tuple[int, int],
    ref_level: int,
    get_dt_and_controller: Callable[
        [int], tuple[float, diffrax.AbstractStepSizeController]
    ],
    saveat: diffrax.SaveAt,
    bm_tol: float,
):
    levy_area1 = _get_minimal_la(solver)
    levy_area2 = _get_minimal_la(ref_solver)
    # Stricter levy_area requirements inherit from less strict ones
    levy_area = _resulting_levy_area(levy_area1, levy_area2)

    level_coarse, level_fine = levels

    dt, step_controller = get_dt_and_controller(ref_level)
    correct_sols, _ = _batch_sde_solve(
        keys,
        get_terms,
        w_shape,
        t0,
        t1,
        y0,
        args,
        ref_solver,
        levy_area,
        dt,
        step_controller,
        bm_tol,
        saveat,
    )

    errs_list, steps_list = [], []
    for level in range(level_coarse, level_fine + 1):
        dt, step_controller = get_dt_and_controller(level)
        sols, steps = _batch_sde_solve(
            keys,
            get_terms,
            w_shape,
            t0,
            t1,
            y0,
            args,
            solver,
            levy_area,
            dt,
            step_controller,
            bm_tol,
            saveat,
        )
        errs = path_l2_dist(sols, correct_sols)
        errs_list.append(errs)
        steps_list.append(jnp.average(steps))
    errs_arr = jnp.array(errs_list)
    steps_arr = jnp.array(steps_list)
    with jax.numpy_dtype_promotion("standard"):
        order, _ = jnp.polyfit(jnp.log(1 / steps_arr), jnp.log(errs_arr), 1)
    return steps_arr, errs_arr, order


@dataclasses.dataclass(frozen=True)
class SDE:
    get_terms: Callable[[AbstractBrownianPath], AbstractTerm]
    args: PyTree
    y0: PyTree[Array]
    t0: float
    t1: float
    w_shape: tuple[int, ...]

    def get_dtype(self):
        return jnp.result_type(*jtu.tree_leaves(self.y0))

    def get_bm(
        self,
        bm_key: PRNGKeyArray,
        levy_area: type[Union[diffrax.BrownianIncrement, diffrax.SpaceTimeLevyArea]],
        tol: float,
    ):
        shp_dtype = jax.ShapeDtypeStruct(self.w_shape, dtype=self.get_dtype())
        return VirtualBrownianTree(self.t0, self.t1, tol, shp_dtype, bm_key, levy_area)


# A more concise function for use in the examples
def simple_sde_order(
    keys,
    sde: SDE,
    solver,
    ref_solver,
    levels,
    get_dt_and_controller,
    saveat,
    bm_tol,
):
    _, level_fine = levels
    ref_level = level_fine + 2
    return sde_solver_strong_order(
        keys,
        sde.get_terms,
        sde.w_shape,
        sde.t0,
        sde.t1,
        sde.y0,
        sde.args,
        solver,
        ref_solver,
        levels,
        ref_level,
        get_dt_and_controller,
        saveat,
        bm_tol,
    )


def simple_batch_sde_solve(
    keys, sde: SDE, solver, levy_area, dt0, controller, bm_tol, saveat
):
    return _batch_sde_solve(
        keys,
        sde.get_terms,
        sde.w_shape,
        sde.t0,
        sde.t1,
        sde.y0,
        sde.args,
        solver,
        levy_area,
        dt0,
        controller,
        bm_tol,
        saveat,
    )


def _squareplus(x):
    return 0.5 * (x + jnp.sqrt(x**2 + 4))


def drift(t, y, args):
    mlp, _, _ = args
    with jax.numpy_dtype_promotion("standard"):
        return 0.25 * mlp(y)


def diffusion(t, y, args):
    _, mlp, noise_dim = args
    with jax.numpy_dtype_promotion("standard"):
        return 1.0 * mlp(y).reshape(3, noise_dim)


def get_mlp_sde(t0, t1, dtype, key, noise_dim):
    driftkey, diffusionkey, ykey = jr.split(key, 3)
    drift_mlp = eqx.nn.MLP(
        in_size=3,
        out_size=3,
        width_size=8,
        depth=2,
        activation=_squareplus,
        final_activation=jnp.tanh,
        key=driftkey,
    )
    diffusion_mlp = eqx.nn.MLP(
        in_size=3,
        out_size=3 * noise_dim,
        width_size=8,
        depth=2,
        activation=_squareplus,
        final_activation=jnp.tanh,
        key=diffusionkey,
    )
    args = (drift_mlp, diffusion_mlp, noise_dim)
    y0 = jr.normal(ykey, (3,), dtype=dtype)

    def get_terms(bm):
        return MultiTerm(ODETerm(drift), ControlTerm(diffusion, bm))

    return SDE(get_terms, args, y0, t0, t1, (noise_dim,))


# This is needed for time_sde (i.e. the additive noise SDE) because initializing
# the weights in the drift MLP with a Gaussian makes the SDE too linear and nice,
# so we need to use a Laplace distribution, which is heavier-tailed.
def lap_init(weight: jax.Array, key) -> jax.Array:
    stddev = 1.0
    return stddev * jax.random.laplace(key, shape=weight.shape, dtype=weight.dtype)


def init_linear_weight(model, init_fn, key):
    is_linear = lambda x: isinstance(x, eqx.nn.Linear)

    def get_weights(model):
        list = []
        for x in jax.tree_util.tree_leaves(model, is_leaf=is_linear):
            if is_linear(x):
                list.extend([x.weight, x.bias])
        return list

    weights = get_weights(model)
    new_weights = [
        init_fn(weight, subkey)
        for weight, subkey in zip(weights, jax.random.split(key, len(weights)))
    ]
    new_model = eqx.tree_at(get_weights, model, new_weights)
    return new_model


def get_time_sde(t0, t1, dtype, key, noise_dim):
    y_dim = 7
    driftkey, diffusionkey, ykey = jr.split(key, 3)

    def ft(t):
        return jnp.array(
            [jnp.sin(t), jnp.cos(4 * t), 1.0, 1.0 / (t + 0.5)], dtype=dtype
        )

    drift_mlp = eqx.nn.MLP(
        in_size=y_dim + 4,
        out_size=y_dim,
        width_size=16,
        depth=5,
        activation=_squareplus,
        key=driftkey,
    )

    # The drift weights must be Laplace-distributed,
    # otherwise the SDE is too linear and nice.
    drift_mlp = init_linear_weight(drift_mlp, lap_init, driftkey)

    def _drift(t, y, _):
        with jax.numpy_dtype_promotion("standard"):
            mlp_out = drift_mlp(jnp.concatenate([y, ft(t)]))
            return (0.01 * mlp_out - 0.5 * y**3) / (jnp.sum(y**2) + 1)

    diffusion_mx = jr.normal(diffusionkey, (4, y_dim, noise_dim), dtype=dtype)

    def _diffusion(t, _, __):
        # This needs a large coefficient to make the SDE not too easy.
        return 1.0 * jnp.tensordot(ft(t), diffusion_mx, axes=1)

    args = (drift_mlp, None, None)
    y0 = jr.normal(ykey, (y_dim,), dtype=dtype)

    def get_terms(bm):
        return MultiTerm(ODETerm(_drift), ControlTerm(_diffusion, bm))

    return SDE(get_terms, args, y0, t0, t1, (noise_dim,))
