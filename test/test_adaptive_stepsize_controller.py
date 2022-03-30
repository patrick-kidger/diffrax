import diffrax
import jax.numpy as jnp


def test_step_ts():
    term = diffrax.ODETerm(lambda t, y, args: -0.2 * y)
    solver = diffrax.Dopri5()
    t0 = 0
    t1 = 5
    dt0 = None
    y0 = 1.0
    stepsize_controller = diffrax.PIDController(rtol=1e-4, atol=1e-6, step_ts=[3, 4])
    saveat = diffrax.SaveAt(steps=True)
    sol = diffrax.diffeqsolve(
        term,
        solver,
        t0,
        t1,
        dt0,
        y0,
        stepsize_controller=stepsize_controller,
        saveat=saveat,
    )
    assert 3 in sol.ts
    assert 4 in sol.ts


def test_jump_ts():
    # Tests no regression of https://github.com/patrick-kidger/diffrax/issues/58

    def vector_field(t, y, args):
        x, v = y
        force = jnp.where(t < 7.5, 10, -10)
        return v, -4 * jnp.pi**2 * x - 4 * jnp.pi * 0.05 * v + force

    term = diffrax.ODETerm(vector_field)
    solver = diffrax.Dopri5()
    t0 = 0
    t1 = 15
    dt0 = None
    y0 = 1.5, 0
    saveat = diffrax.SaveAt(steps=True)

    def run(**kwargs):
        stepsize_controller = diffrax.PIDController(rtol=1e-4, atol=1e-6, **kwargs)
        return diffrax.diffeqsolve(
            term,
            solver,
            t0,
            t1,
            dt0,
            y0,
            stepsize_controller=stepsize_controller,
            saveat=saveat,
        )

    sol_no_jump_ts = run()
    sol_with_jump_ts = run(jump_ts=[7.5])
    assert sol_no_jump_ts.stats["num_steps"] > sol_with_jump_ts.stats["num_steps"]
    assert sol_with_jump_ts.result == 0

    sol = run(jump_ts=[7.5], step_ts=[7.5])
    assert sol.result == 0
    sol = run(jump_ts=[7.5], step_ts=[3.5, 8])
    assert sol.result == 0
    assert 3.5 in sol.ts
    assert 8 in sol.ts
