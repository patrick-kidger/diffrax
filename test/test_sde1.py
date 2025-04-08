from typing import Literal

import diffrax
import jax.numpy as jnp
import jax.random as jr
import pytest

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
    bmkey = jr.key(5678)
    sde_key = jr.key(11)
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
        levy_area=None,
        ref_solution=None,
    )
    # TODO: this is a pretty wide range to check. Maybe fixable by being better about
    # the randomness (e.g. average over multiple original seeds)?
    assert -0.4 < order - theoretical_order < 0.4


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
@pytest.mark.parametrize("dtype", (jnp.float64,))
def test_sde_strong_limit(
    solver_ctr, noise: Literal["any", "com", "add"], theoretical_order, dtype
):
    bmkey = jr.key(5678)
    sde_key = jr.key(11)
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
