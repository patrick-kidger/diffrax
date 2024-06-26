import diffrax
import jax
import jax.numpy as jnp
import jax.random as jr
import pytest
from diffrax import diffeqsolve, LangevinTerm, SaveAt, VirtualBrownianTree

from .helpers import (
    _abstract_la_to_la,
    get_bqp,
    get_harmonic_oscillator,
    get_pytree_langevin,
    simple_batch_sde_solve,
    simple_sde_order,
)


def _only_langevin_solvers_cls():
    yield diffrax.ALIGN
    yield diffrax.SORT
    yield diffrax.ShOULD
    yield diffrax.UBU3


def _solvers():
    # solver, order
    yield diffrax.ALIGN(0.1), 2.0
    yield diffrax.ShARK(), 2.0
    yield diffrax.SRA1(), 2.0
    yield diffrax.SORT(0.01), 3.0
    yield diffrax.ShOULD(0.01), 3.0
    yield diffrax.UBU3(0.0), 3.0


@pytest.mark.parametrize("solver_cls", _only_langevin_solvers_cls())
@pytest.mark.parametrize("taylor", [True, False])
@pytest.mark.parametrize("dtype", [jnp.float16, jnp.float32, jnp.float64])
@pytest.mark.parametrize("dim", [1, 4])
def test_shape(solver_cls, taylor, dtype, dim):
    if taylor:
        solver = solver_cls(100.0)
    else:
        solver = solver_cls(0.0)

    if dtype == jnp.float16 and isinstance(
        solver, (diffrax.SORT, diffrax.ShOULD, diffrax.UBU3)
    ):
        pytest.skip(
            "Due to the use of multivariate normal in the the computation"
            " of space-time-time Levy area, SORT and ShOULD are not"
            " compatible with float16"
        )
    t0, t1 = 0.3, 1.0
    saveat = SaveAt(ts=jnp.linspace(t0, t1, 10, dtype=dtype))
    u = jnp.astype(1.0, dtype)
    gam = jnp.astype(1.0, dtype)
    vec_u = jnp.ones((dim,), dtype=dtype)
    vec_gam = jnp.ones((dim,), dtype=dtype)
    x0 = jnp.zeros((dim,), dtype=dtype)
    v0 = jnp.zeros((dim,), dtype=dtype)
    y0 = (x0, v0)
    f = lambda x: 0.5 * x
    shp_dtype = jax.ShapeDtypeStruct((dim,), dtype)
    levy_area = _abstract_la_to_la(solver.minimal_levy_area)
    bm = VirtualBrownianTree(
        t0,
        t1,
        tol=2**-4,
        shape=shp_dtype,
        key=jr.key(4),
        levy_area=levy_area,
    )
    for args in [
        (gam, u, f),
        (vec_gam, u, f),
        (gam, vec_u, f),
        (vec_gam, vec_u, f),
    ]:
        terms = LangevinTerm(args, bm, x0)
        sol = diffeqsolve(
            terms, solver, t0, t1, dt0=0.3, y0=y0, args=None, saveat=saveat
        )
        assert sol.ys is not None
        for entry in sol.ys:
            assert entry.shape == (10, dim)
            assert jnp.dtype(entry) == dtype


sdes = (
    (get_harmonic_oscillator, "hosc"),
    (get_bqp, "bqp"),
    (get_pytree_langevin, "pytree"),
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

    pytree_sde = get_pytree_langevin(t0, t1, jnp.float64)
    pytree_sol, _ = simple_batch_sde_solve(
        bmkeys, pytree_sde, ref_solver, levy_area, None, controller, bm_tol, saveat
    )

    sols = {
        "hosc": hosc_sol,
        "bqp": bqp_sol,
        "pytree": pytree_sol,
    }
    return sols, t0, t1, bmkeys, saveat, level_coarse, level_fine, levy_area, bm_tol


@pytest.mark.parametrize("get_sde,sde_name", sdes)
@pytest.mark.parametrize("solver,theoretical_order", _solvers())
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
        bm_tol=2**-14,
        levy_area=levy_area,
        ref_solution=true_sol,
    )
    # The upper bound needs to be 0.25, otherwise we fail.
    # This still preserves a 0.05 buffer between the intervals
    # corresponding to the different orders.
    assert (
        -0.2 < order - theoretical_order < 0.25
    ), f"order={order}, theoretical_order={theoretical_order}"
