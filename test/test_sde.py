from typing import Literal

import diffrax
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
import pytest
from diffrax import ControlTerm, MultiTerm, ODETerm, WeaklyDiagonalControlTerm

from .helpers import (
    get_mlp_sde,
    get_time_sde,
    path_l2_dist,
    simple_batch_sde_solve,
    simple_sde_order,
)


def _solvers_and_orders():
    # solver, noise, order
    # noise is "any" or "com" or "add" where "com" means commutative and "add" means
    # additive.
    yield diffrax.SPaRK, "any", 0.5
    yield diffrax.GeneralShARK, "any", 0.5
    yield diffrax.SlowRK, "any", 0.5
    yield diffrax.SPaRK, "com", 1
    yield diffrax.GeneralShARK, "com", 1
    yield diffrax.SlowRK, "com", 1.5
    yield diffrax.SPaRK, "add", 1.5
    yield diffrax.GeneralShARK, "add", 1.5
    yield diffrax.ShARK, "add", 1.5
    yield diffrax.SRA1, "add", 1.5
    yield diffrax.SEA, "add", 1.0


# For solvers of high order, comparing to Euler or Heun is not sufficient,
# because they are substantially worse than e.g. ShARK. ShARK is more precise
# at dt=2**-4 than Euler is at dt=2**-14 (and it takes forever to run at such
# a small dt). Hence , the order of convergence of ShARK seems to plateau at
# discretisations finer than 2**-4.
# Therefore, we use two separate tests. First we determine how fast the solver
# converges to its own limit (i.e. using itself as reference), and then in a
# different test check whether that limit is the same as the Euler/Heun limit.
@pytest.mark.parametrize("solver_ctr,noise,theoretical_order", _solvers_and_orders())
@pytest.mark.parametrize(
    "dtype",
    (jnp.float64,),
)
def test_sde_strong_order_new(
    solver_ctr, noise: Literal["any", "com", "add"], theoretical_order, dtype
):
    bmkey = jr.PRNGKey(5678)
    sde_key = jr.PRNGKey(11)
    num_samples = 100
    bmkeys = jr.split(bmkey, num=num_samples)
    t0 = 0.3
    t1 = 5.3

    if noise == "add":
        sde = get_time_sde(t0, t1, dtype, sde_key, noise_dim=7)
    else:
        if noise == "com":
            noise_dim = 1
        elif noise == "any":
            noise_dim = 5
        else:
            assert False
        sde = get_mlp_sde(t0, t1, dtype, sde_key, noise_dim=noise_dim)

    ref_solver = solver_ctr()
    level_coarse, level_fine = 1, 7

    # We specify the times to which we step in way that each level contains all the
    # steps of the previous level. This is so that we can compare the solutions at
    # all the times in saveat, and not just at the end time.
    def get_dt_and_controller(level):
        step_ts = jnp.linspace(t0, t1, 2**level + 1, endpoint=True)
        return None, diffrax.StepTo(ts=step_ts)

    saveat = diffrax.SaveAt(ts=jnp.linspace(t0, t1, 2**level_coarse + 1, endpoint=True))

    hs, errors, order = simple_sde_order(
        bmkeys,
        sde,
        solver_ctr(),
        ref_solver,
        (level_coarse, level_fine),
        get_dt_and_controller,
        saveat,
        bm_tol=2**-14,
    )
    # The upper bound needs to be 0.25, otherwise we fail.
    # This still preserves a 0.05 buffer between the intervals
    # corresponding to the different orders.
    assert -0.2 < order - theoretical_order < 0.25


# Make variables to store the correct solutions in.
# This is to avoid recomputing the correct solutions for every solver.
solutions = {
    "Ito": {
        "any": None,
        "com": None,
        "add": None,
    },
    "Stratonovich": {
        "any": None,
        "com": None,
        "add": None,
    },
}


# Now compare the limit of Euler/Heun to the limit of the other solvers,
# using a single reference solution. We use Euler if the solver is Ito
# and Heun if the solver is Stratonovich.
@pytest.mark.parametrize("solver_ctr,noise,theoretical_order", _solvers_and_orders())
@pytest.mark.parametrize(
    "dtype",
    (jnp.float64,),
)
def test_sde_strong_limit(
    solver_ctr, noise: Literal["any", "com", "add"], theoretical_order, dtype
):
    bmkey = jr.PRNGKey(5678)
    sde_key = jr.PRNGKey(11)
    num_samples = 100
    bmkeys = jr.split(bmkey, num=num_samples)
    t0 = 0.3
    t1 = 5.3

    if noise == "add":
        sde = get_time_sde(t0, t1, dtype, sde_key, noise_dim=3)
        level_fine = 12
        if theoretical_order <= 1.0:
            level_coarse = 11
        else:
            level_coarse = 8
    else:
        level_coarse, level_fine = 7, 11
        if noise == "com":
            noise_dim = 1
        elif noise == "any":
            noise_dim = 5
        else:
            assert False
        sde = get_mlp_sde(t0, t1, dtype, sde_key, noise_dim=noise_dim)

    # Reference solver is always an ODE-viable solver, so its implementation has been
    # verified by the ODE tests like test_ode_order.
    if issubclass(solver_ctr, diffrax.AbstractItoSolver):
        sol_type = "Ito"
        ref_solver = diffrax.Euler()
    elif issubclass(solver_ctr, diffrax.AbstractStratonovichSolver):
        sol_type = "Stratonovich"
        ref_solver = diffrax.Heun()
    else:
        assert False

    ts_fine = jnp.linspace(t0, t1, 2**level_fine + 1, endpoint=True)
    ts_coarse = jnp.linspace(t0, t1, 2**level_coarse + 1, endpoint=True)
    contr_fine = diffrax.StepTo(ts=ts_fine)
    contr_coarse = diffrax.StepTo(ts=ts_coarse)
    save_ts = jnp.linspace(t0, t1, 2**5 + 1, endpoint=True)
    assert len(jnp.intersect1d(ts_fine, save_ts)) == len(save_ts)
    assert len(jnp.intersect1d(ts_coarse, save_ts)) == len(save_ts)
    saveat = diffrax.SaveAt(ts=save_ts)
    levy_area = diffrax.SpaceTimeLevyArea  # must be common for all solvers

    if solutions[sol_type][noise] is None:
        correct_sol, _ = simple_batch_sde_solve(
            bmkeys, sde, ref_solver, levy_area, None, contr_fine, 2**-10, saveat
        )
        solutions[sol_type][noise] = correct_sol
    else:
        correct_sol = solutions[sol_type][noise]

    sol, _ = simple_batch_sde_solve(
        bmkeys, sde, solver_ctr(), levy_area, None, contr_coarse, 2**-10, saveat
    )
    error = path_l2_dist(correct_sol, sol)
    assert error < 0.05


def _solvers():
    yield diffrax.SPaRK
    yield diffrax.GeneralShARK
    yield diffrax.SlowRK
    yield diffrax.ShARK
    yield diffrax.SRA1
    yield diffrax.SEA


# Define the SDE
def dict_drift(t, y, args):
    pytree, _ = args
    return jtu.tree_map(lambda _, x: -0.5 * x, pytree, y)


def dict_diffusion(t, y, args):
    pytree, additive = args

    def get_matrix(y_leaf):
        if additive:
            return 2.0 * jnp.ones(y_leaf.shape + (3,), dtype=jnp.float64)
        else:
            return 2.0 * jnp.broadcast_to(
                jnp.expand_dims(y_leaf, axis=y_leaf.ndim), y_leaf.shape + (3,)
            )

    return jtu.tree_map(get_matrix, y)


@pytest.mark.parametrize("shape", [(), (5, 2)])
@pytest.mark.parametrize("solver_ctr", _solvers())
@pytest.mark.parametrize(
    "dtype",
    (jnp.float64, jnp.complex128),
)
def test_sde_solver_shape(shape, solver_ctr, dtype):
    pytree = ({"a": 0, "b": [0, 0]}, 0, 0)
    key = jr.PRNGKey(0)
    y0 = jtu.tree_map(lambda _: jr.normal(key, shape, dtype=dtype), pytree)
    t0, t1, dt0 = 0.0, 1.0, 0.3

    # Some solvers only work with additive noise
    additive = solver_ctr in [diffrax.ShARK, diffrax.SRA1, diffrax.SEA]
    args = (pytree, additive)
    solver = solver_ctr()
    bmkey = jr.PRNGKey(1)
    struct = jax.ShapeDtypeStruct((3,), dtype)
    bm_shape = jtu.tree_map(lambda _: struct, pytree)
    bm = diffrax.VirtualBrownianTree(
        t0, t1, 0.1, bm_shape, bmkey, diffrax.SpaceTimeLevyArea
    )
    terms = MultiTerm(ODETerm(dict_drift), ControlTerm(dict_diffusion, bm))
    solution = diffrax.diffeqsolve(
        terms, solver, t0, t1, dt0, y0, args, saveat=diffrax.SaveAt(t1=True)
    )
    assert jtu.tree_structure(solution.ys) == jtu.tree_structure(y0)
    for leaf in jtu.tree_leaves(solution.ys):
        assert leaf[0].shape == shape


def _weakly_diagonal_noise_helper(solver, dtype):
    w_shape = (3,)
    args = (0.5, 1.2)

    def _diffusion(t, y, args):
        a, b = args
        return jnp.array([b, t, 1 / (t + 1.0)], dtype=dtype)

    def _drift(t, y, args):
        a, b = args
        return -a * y

    y0 = jnp.ones(w_shape, dtype)

    bm = diffrax.VirtualBrownianTree(
        0.0, 1.0, 0.05, w_shape, jr.PRNGKey(0), diffrax.SpaceTimeLevyArea
    )

    terms = MultiTerm(ODETerm(_drift), WeaklyDiagonalControlTerm(_diffusion, bm))
    saveat = diffrax.SaveAt(t1=True)
    solution = diffrax.diffeqsolve(
        terms, solver, 0.0, 1.0, 0.1, y0, args, saveat=saveat
    )
    assert solution.ys is not None
    assert solution.ys.shape == (1, 3)


@pytest.mark.parametrize("solver_ctr", _solvers())
@pytest.mark.parametrize(
    "dtype",
    (jnp.float64, jnp.complex128),
)
def test_weakly_diagonal_noise(solver_ctr, dtype):
    _weakly_diagonal_noise_helper(solver_ctr(), dtype)


@pytest.mark.parametrize(
    "dtype",
    (jnp.float64, jnp.complex128),
)
def test_halfsolver_term_compatible(dtype):
    _weakly_diagonal_noise_helper(diffrax.HalfSolver(diffrax.SPaRK()), dtype)
