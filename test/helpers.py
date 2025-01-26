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
    UnderdampedLangevinDiffusionTerm,
    UnderdampedLangevinDriftTerm,
    VirtualBrownianTree,
)
from diffrax._misc import is_tuple_of_ints
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
            square_diff = jnp.square(jnp.abs(y1 - y2))
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
    return getattr(solver, "minimal_levy_area", diffrax.AbstractBrownianIncrement)


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
    w_shape: Union[tuple[int, ...], PyTree[jax.ShapeDtypeStruct]],
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
    if is_tuple_of_ints(w_shape):
        struct = jax.ShapeDtypeStruct(w_shape, dtype)
    else:
        struct = w_shape
    bm = diffrax.VirtualBrownianTree(
        t0=t0,
        t1=t1,
        shape=struct,
        tol=bm_tol,
        key=key,
        levy_area=concrete_la,
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
    steps = sol.stats["num_accepted_steps"]
    return sol.ys, steps


def _resulting_levy_area(
    levy_area1: type[diffrax.AbstractBrownianIncrement],
    levy_area2: type[diffrax.AbstractBrownianIncrement],
) -> type[diffrax.AbstractBrownianIncrement]:
    """A helper that returns the stricter Lévy area.

    **Arguments:**

    - `levy_area1`: The first Lévy area type.
    - `levy_area2`: The second Lévy area type.

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
    ref_solver: Optional[diffrax.AbstractSolver],
    levels: tuple[int, int],
    ref_level: int,
    get_dt_and_controller: Callable[
        [int], tuple[float, diffrax.AbstractStepSizeController]
    ],
    saveat: diffrax.SaveAt,
    bm_tol: float,
    levy_area: Optional[type[diffrax.AbstractBrownianIncrement]],
    ref_solution: Optional[PyTree[Array]],
):
    if levy_area is None:
        levy_area1 = _get_minimal_la(solver)
        levy_area2 = _get_minimal_la(ref_solver)
        # Stricter levy_area requirements inherit from less strict ones
        levy_area = _resulting_levy_area(levy_area1, levy_area2)

    level_coarse, level_fine = levels

    if ref_solution is None:
        assert ref_solver is not None
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
    else:
        correct_sols = ref_solution

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
    w_shape: Union[tuple[int, ...], PyTree[jax.ShapeDtypeStruct]]

    def get_dtype(self):
        return jnp.result_type(*jtu.tree_leaves(self.y0))

    def get_bm(
        self,
        bm_key: PRNGKeyArray,
        levy_area: type[
            Union[
                diffrax.BrownianIncrement,
                diffrax.SpaceTimeLevyArea,
                diffrax.SpaceTimeTimeLevyArea,
            ]
        ],
        tol: float,
    ):
        return VirtualBrownianTree(
            self.t0, self.t1, tol, self.w_shape, bm_key, levy_area
        )


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
    levy_area,
    ref_solution,
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
        levy_area,
        ref_solution,
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


def make_underdamped_langevin_term(gamma, u, grad_f, bm):
    drift = UnderdampedLangevinDriftTerm(gamma, u, grad_f)
    diffusion = UnderdampedLangevinDiffusionTerm(gamma, u, bm)
    return MultiTerm(drift, diffusion)


def get_bqp(t0=0.3, t1=15.0, dtype=jnp.float32):
    grad_f_bqp = lambda x, _: 4 * x * (jnp.square(x) - 1)
    gamma, u = dtype(0.8), dtype(0.2)
    y0_bqp = (dtype(0), dtype(0))
    w_shape_bqp = ()

    def get_terms_bqp(bm):
        return make_underdamped_langevin_term(gamma, u, grad_f_bqp, bm)

    return SDE(get_terms_bqp, None, y0_bqp, t0, t1, w_shape_bqp)


def get_harmonic_oscillator(t0=0.3, t1=15.0, dtype=jnp.float32):
    gamma_hosc = jnp.array([2, 0.5], dtype=dtype)
    u_hosc = jnp.array([0.5, 2], dtype=dtype)
    x0 = jnp.zeros((2,), dtype=dtype)
    v0 = jnp.zeros((2,), dtype=dtype)
    y0_hosc = (x0, v0)
    w_shape_hosc = (2,)

    def get_terms_hosc(bm):
        return make_underdamped_langevin_term(
            gamma_hosc, u_hosc, lambda x, _: 2 * x, bm
        )

    return SDE(get_terms_hosc, None, y0_hosc, t0, t1, w_shape_hosc)


def get_neals_funnel(t0=0.0, t1=16.0, dtype=jnp.float32):
    def log_p(x):
        z_term = x[0] ** 2 / 6.0
        y_term = jnp.sum(x[1:] ** 2) / jax.lax.stop_gradient(2.0 * jnp.exp(x[0] / 4.0))
        return z_term + y_term

    grad_log_p = jax.grad(log_p)

    gamma = 2.0
    u = 1.0
    x0 = jnp.zeros((10,), dtype=dtype)
    v0 = jnp.zeros((10,), dtype=dtype)
    y0_neal = (x0, v0)
    w_shape_neal = (10,)

    def get_terms_neal(bm):
        return make_underdamped_langevin_term(gamma, u, grad_log_p, bm)

    return SDE(get_terms_neal, None, y0_neal, t0, t1, w_shape_neal)


def get_uld3_langevin(t0=0.3, t1=15.0, dtype=jnp.float32):
    # Three particles in 3D space with a potential that has three local minima,
    # at (2, 2, 2), (-2, -2, -2) and (3, -1, 0).
    def single_particle_potential(x):
        assert x.shape == (3,)
        return 1.0 * (
            jnp.sum((x - 2.0 * jnp.ones((3,), dtype=dtype)) ** 2)
            * jnp.sum((x + 2.0 * jnp.ones((3,), dtype=dtype)) ** 2)
            * jnp.sum((x - jnp.array([3, -1, 0], dtype=dtype)) ** 2)
        )

    def potential(x):
        assert x.shape == (9,)
        return (
            single_particle_potential(x[:3])
            + single_particle_potential(x[3:6])
            + single_particle_potential(x[6:])
        )

    grad_potential = jax.grad(potential)

    def single_circ(x):
        assert x.shape == (3,)
        return 0.1 * jnp.array([x[1], -x[0], 0.0])

    def circular_term(x):
        assert x.shape == (9,)
        return jnp.concatenate(
            [
                single_circ(x[:3]),
                single_circ(x[3:6]),
                single_circ(x[6:]),
            ]
        )

    def grad_f(x):
        assert x.shape == (9,)
        # x0 and x1 will do a circular motion, so we will add a term of the form
        force = grad_potential(x) + circular_term(x)
        return 10.0 * force / (jnp.sum(jnp.abs(force)) + 10.0)

    u = 1.0
    gamma = 2.0
    x0 = jnp.array([-1, 0, 1, 1, 0, -1, 1, 0, -1], dtype=dtype)
    v0 = jnp.zeros((9,), dtype=dtype)
    y0_uld3 = (x0, v0)
    w_shape_uld3 = (9,)

    def get_terms_uld3(bm):
        return make_underdamped_langevin_term(u, gamma, grad_f, bm)

    return SDE(get_terms_uld3, None, y0_uld3, t0, t1, w_shape_uld3)
