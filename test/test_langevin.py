import diffrax
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
import pytest
from diffrax import diffeqsolve, make_langevin_term, SaveAt

from .helpers import (
    get_bqp,
    get_harmonic_oscillator,
    path_l2_dist,
    SDE,
    simple_batch_sde_solve,
    simple_sde_order,
)


def _only_langevin_solvers_cls():
    yield diffrax.ALIGN
    yield diffrax.ShOULD
    yield diffrax.QUICSORT


def _solvers_and_orders():
    # solver, order
    yield diffrax.ALIGN(0.1), 2.0
    yield diffrax.ShOULD(0.1), 3.0
    yield diffrax.QUICSORT(0.1), 3.0
    yield diffrax.ShARK(), 2.0


def get_pytree_langevin(t0=0.3, t1=1.0, dtype=jnp.float32):
    def make_pytree(array_factory):
        return {
            "rr": (
                array_factory((2,), dtype),
                array_factory((2,), dtype),
            ),
            "qq": (
                array_factory((2,), dtype),
                array_factory((3,), dtype),
            ),
        }

    x0 = make_pytree(jnp.ones)
    v0 = make_pytree(jnp.zeros)
    y0 = (x0, v0)

    g1 = {
        "rr": 0.001 * jnp.ones((2,), dtype),
        "qq": (
            jnp.ones((), dtype),
            10 * jnp.ones((3,), dtype),
        ),
    }

    u1 = {
        "rr": (jnp.ones((), dtype), 10.0),
        "qq": jnp.ones((), dtype),
    }

    def grad_f(x):
        xa = x["rr"]
        xb = x["qq"]
        return {"rr": jtu.tree_map(lambda _x: 0.2 * _x, xa), "qq": xb}

    w_shape = jtu.tree_map(lambda _x: jax.ShapeDtypeStruct(_x.shape, _x.dtype), x0)

    def get_terms(bm):
        return make_langevin_term(g1, u1, grad_f, bm, x0)

    return SDE(get_terms, None, y0, t0, t1, w_shape)


@pytest.mark.parametrize("solver_cls", _only_langevin_solvers_cls())
@pytest.mark.parametrize("taylor", [True, False])
@pytest.mark.parametrize("dtype", [jnp.float16, jnp.float32, jnp.float64])
def test_shape(solver_cls, taylor, dtype):
    if taylor:
        solver = solver_cls(100.0)
    else:
        solver = solver_cls(0.01)

    t0, t1 = 0.3, 1.0
    dt0 = 0.3
    saveat = SaveAt(ts=jnp.linspace(t0, t1, 7, dtype=dtype))

    sde = get_pytree_langevin(t0, t1, dtype)
    bm = sde.get_bm(jr.key(5678), diffrax.SpaceTimeTimeLevyArea, tol=0.2)
    terms = sde.get_terms(bm)

    sol = diffeqsolve(
        terms, solver, t0, t1, dt0=dt0, y0=sde.y0, args=None, saveat=saveat
    )
    assert sol.ys is not None
    assert sol.ts is not None

    # check that the output has the correct pytree structure and shape
    def check_shape(y0_leaf, sol_leaf):
        assert (
            sol_leaf.shape == (7,) + y0_leaf.shape
        ), f"shape={sol_leaf.shape}, expected={(7,) + y0_leaf.shape}"
        assert sol_leaf.dtype == dtype, f"dtype={sol_leaf.dtype}, expected={dtype}"
        return sol_leaf.shape

    jtu.tree_map(check_shape, sde.y0, sol.ys)


sdes = (
    (get_harmonic_oscillator, "hosc"),
    (get_bqp, "bqp"),
)


@pytest.fixture(scope="module")
def fine_langevin_solutions():
    bmkey = jr.key(5678)
    num_samples = 2000
    bmkeys = jr.split(bmkey, num=num_samples)
    t0 = 0.1
    t1 = 5.3
    level_precise = 10
    level_coarse, level_fine = 3, 6
    saveat = diffrax.SaveAt(ts=jnp.linspace(t0, t1, 2**level_coarse + 1, endpoint=True))
    ref_solver = diffrax.ShARK()
    levy_area = diffrax.SpaceTimeTimeLevyArea
    controller = diffrax.StepTo(
        ts=jnp.linspace(t0, t1, 2**level_precise + 1, endpoint=True)
    )
    bm_tol = 0.5 * (t1 - t0) * 2**-level_precise

    hosc_sde = get_harmonic_oscillator(t0, t1, jnp.float64)
    hosc_sol, _ = simple_batch_sde_solve(
        bmkeys, hosc_sde, ref_solver, levy_area, None, controller, bm_tol, saveat
    )

    bqp_sde = get_bqp(t0, t1, jnp.float64)
    bqp_sol, _ = simple_batch_sde_solve(
        bmkeys, bqp_sde, ref_solver, levy_area, None, controller, bm_tol, saveat
    )

    sols = {
        "hosc": hosc_sol,
        "bqp": bqp_sol,
    }
    return sols, t0, t1, bmkeys, saveat, level_coarse, level_fine, levy_area, bm_tol


@pytest.mark.parametrize("get_sde,sde_name", sdes)
@pytest.mark.parametrize("solver,theoretical_order", _solvers_and_orders())
def test_langevin_strong_order(
    get_sde, sde_name, solver, theoretical_order, fine_langevin_solutions
):
    (
        true_sols,
        t0,
        t1,
        bmkeys,
        saveat,
        level_coarse,
        level_fine,
        levy_area,
        bm_tol,
    ) = fine_langevin_solutions
    true_sol = true_sols[sde_name]

    if theoretical_order < 3:
        level_fine += 1

    sde = get_sde(t0, t1, jnp.float64)

    # We specify the times to which we step in way that each level contains all the
    # steps of the previous level. This is so that we can compare the solutions at
    # all the times in saveat, and not just at the end time.
    def get_dt_and_controller(level):
        step_ts = jnp.linspace(t0, t1, 2**level + 1, endpoint=True)
        return None, diffrax.StepTo(ts=step_ts)

    hs, errors, order = simple_sde_order(
        bmkeys,
        sde,
        solver,
        None,
        (level_coarse, level_fine),
        get_dt_and_controller,
        saveat,
        bm_tol=bm_tol,
        levy_area=levy_area,
        ref_solution=true_sol,
    )
    # The upper bound needs to be 0.25, otherwise we fail.
    # This still preserves a 0.05 buffer between the intervals
    # corresponding to the different orders.
    assert (
        -0.2 < order - theoretical_order < 0.25
    ), f"order={order}, theoretical_order={theoretical_order}"


@pytest.mark.parametrize("solver_cls", _only_langevin_solvers_cls())
def test_reverse_solve(solver_cls):
    t0, t1 = 0.7, -1.2
    dt0 = -0.01
    saveat = SaveAt(ts=jnp.linspace(t0, t1, 20, endpoint=True))

    gamma = jnp.array([2, 0.5], dtype=jnp.float64)
    u = jnp.array([0.5, 2], dtype=jnp.float64)
    x0 = jnp.zeros((2,), dtype=jnp.float64)
    v0 = jnp.zeros((2,), dtype=jnp.float64)
    y0 = (x0, v0)

    bm = diffrax.VirtualBrownianTree(
        t1,
        t0,
        tol=0.005,
        shape=(2,),
        key=jr.key(0),
        levy_area=diffrax.SpaceTimeTimeLevyArea,
    )
    terms = diffrax.make_langevin_term(gamma, u, lambda x: 2 * x, bm, x0)

    solver = solver_cls(0.01)
    sol = diffeqsolve(terms, solver, t0, t1, dt0=dt0, y0=y0, args=None, saveat=saveat)

    ref_solver = diffrax.Heun()
    ref_sol = diffeqsolve(
        terms, ref_solver, t0, t1, dt0=dt0, y0=y0, args=None, saveat=saveat
    )

    # print(jtu.tree_map(lambda x: x.shape, sol.ys))
    # print(jtu.tree_map(lambda x: x.shape, ref_sol.ys))
    error = path_l2_dist(sol.ys, ref_sol.ys)
    assert error < 0.1
