import jax.lax as lax
import jax.numpy as jnp
import jax.random as jrandom
import pytest

import diffrax
from diffrax._delays import Delays


def open_process_file(path):
    ts, ys = [], []
    with open(path, "r", encoding="utf-8") as infile:
        for line in infile:
            data = line.split()
            ts.append(float(data[0])), ys.append(float(data[1]))
    return jnp.array(ts), jnp.array(ys)


def test_dde_solver_with_ode():
    # testing that dde solver solves ode
    # when history not specific
    key = jrandom.PRNGKey(5678)
    akey, ykey = jrandom.split(key, 2)

    A = jrandom.normal(akey, (10, 10), dtype=jnp.float64) * 0.5

    def dde_f(t, y, args, history):
        return A @ y

    def ode_f(t, y, args):
        return A @ y

    dde_term = diffrax.ODETerm(dde_f)
    ode_term = diffrax.ODETerm(ode_f)
    t0 = 0
    t1 = 4
    ts = jnp.linspace(t0, t1, int(10 * (t1 - t0)))
    y0 = jrandom.normal(ykey, (10,), dtype=jnp.float64)
    delays = diffrax.Delays(
        delays=[lambda t, y, args: 0.2],
        initial_discontinuities=jnp.array([0.0]),
        max_discontinuities=100,
        recurrent_checking=True,
        rtol=10e-3,
        atol=10e-6,
    )

    dt0 = 0.1
    dde_sol = diffrax.diffeqsolve(
        dde_term,
        diffrax.Dopri5(),
        t0,
        t1,
        dt0,
        y0=lambda t: y0,
        delays=delays,
        saveat=diffrax.SaveAt(ts=ts, dense=True),
    )
    ode_sol = diffrax.diffeqsolve(
        ode_term,
        diffrax.Dopri5(),
        t0,
        t1,
        dt0,
        y0,
        saveat=diffrax.SaveAt(ts=ts, dense=True),
    )

    error = jnp.mean(jnp.abs(dde_sol.ys - ode_sol.ys))
    assert error < 10**-5


def test_jump_ts_dde_solver():
    # test jump ts with dde solver
    # when t=2.0 the vf changes

    key = jrandom.PRNGKey(5678)

    def vf(t, y, args, history):
        sign = jnp.where(t < 2, 1, -1)
        return sign * history[0]

    def first_part_vf(t, y, args, history):
        return history[0]

    def second_part_vf(t, y, args, history):
        return -history[0]

    t0, t1 = 0.0, 4.0
    dt0 = 0.1
    ts_first = jnp.linspace(0, 2.0, 20)
    ts_second = jnp.linspace(2.0, 4.0, 20)
    ts = jnp.concatenate([ts_first, ts_second[1:]])
    y0 = jrandom.normal(key, (1,), dtype=jnp.float64)
    delays = diffrax.Delays(
        delays=[lambda t, y, args: 1.0],
        initial_discontinuities=jnp.array([0.0]),
        max_discontinuities=10,
        recurrent_checking=False,
        rtol=10e-3,
        atol=10e-6,
    )

    delays2 = diffrax.Delays(
        delays=[lambda t, y, args: 1.0],
        initial_discontinuities=jnp.array([2.0]),
        max_discontinuities=10,
        recurrent_checking=False,
        rtol=10e-3,
        atol=10e-6,
    )

    first_part_dde = diffrax.diffeqsolve(
        diffrax.ODETerm(first_part_vf),
        diffrax.Dopri5(),
        t0,
        ts_first[-1],
        dt0,
        y0=lambda t: y0,
        delays=delays,
        saveat=diffrax.SaveAt(ts=ts_first, dense=True),
    )
    second_part_dde = diffrax.diffeqsolve(
        diffrax.ODETerm(second_part_vf),
        diffrax.Dopri5(),
        ts_first[-1],
        t1,
        dt0,
        y0=lambda t: first_part_dde.interpolation.evaluate(t),
        delays=delays2,
        stepsize_controller=diffrax.PIDController(rtol=1e-6, atol=1e-3),
        saveat=diffrax.SaveAt(ts=ts_second, dense=True),
    )
    complete_dde = diffrax.diffeqsolve(
        diffrax.ODETerm(vf),
        diffrax.Dopri5(),
        t0,
        t1,
        dt0,
        y0=lambda t: y0,
        delays=delays,
        stepsize_controller=diffrax.PIDController(
            rtol=1e-9, atol=1e-6, jump_ts=jnp.array([2.0])
        ),
        saveat=diffrax.SaveAt(ts=ts, dense=True),
    )

    error = jnp.mean(
        jnp.abs(
            complete_dde.ys
            - jnp.concatenate([first_part_dde.ys, second_part_dde.ys[1:]])
        )
    )
    assert error < 1**-5


def test_smooth_dde():
    # testing a smooth dde with no initial discontinuities
    # y' = y(t-1), phi = t + 1
    # for t in [0,1], y(t) = t**2/2 + 1
    # for t in [1,2], y(t) = (t-1)**3/(2*3) + t + 1/2
    # we compare the values at t=1,2 with their analytical
    # solution
    def dde_f(t, y, args, history):
        return history[0]

    dde_term = diffrax.ODETerm(dde_f)
    t0, t1 = 0.0, 2.0
    ts = jnp.linspace(t0, t1, 100)
    delays = diffrax.Delays(
        delays=[lambda t, y, args: 1.0],
        initial_discontinuities=None,
        max_discontinuities=10,
        recurrent_checking=True,
        rtol=10e-3,
        atol=10e-6,
    )

    dt0 = 0.1
    dde_sol = diffrax.diffeqsolve(
        dde_term,
        diffrax.Dopri5(),
        t0,
        t1,
        dt0,
        y0=lambda t: t + 1,
        delays=delays,
        saveat=diffrax.SaveAt(ts=ts, dense=True),
    )

    error1 = jnp.mean(jnp.abs(dde_sol.ys[50] - 3 / 2))
    error2 = jnp.mean(jnp.abs(dde_sol.ys[100] - 8 / 3))

    assert error1 < 10**-1
    assert error2 < 10**-3


def _test_exceed_max_discontinuities():
    # we recurrent_checking to True and
    # integrate a DDE with a delay equal
    # to 1 and hence from t > 10.0 we
    # should have RunTimeError picked
    # up
    def dde_f(t, y, args, history):
        return -history[0]

    t0, t1 = 0.0, 12.0
    ts = jnp.linspace(t0, t1, 120)
    delays = diffrax.Delays(
        delays=[lambda t, y, args: 1.0],
        initial_discontinuities=jnp.array([0.0]),
        max_discontinuities=10,
        recurrent_checking=True,
        rtol=10e-3,
        atol=10e-6,
    )

    dt0 = 0.1

    return diffrax.diffeqsolve(
        diffrax.ODETerm(dde_f),
        diffrax.Dopri5(),
        t0,
        t1,
        dt0,
        y0=lambda t: 1.0,
        delays=delays,
        saveat=diffrax.SaveAt(ts=ts, dense=True),
    )


def test_exceed_max_discontinuities():
    with pytest.raises(RuntimeError):
        _test_exceed_max_discontinuities()


def test_only_explicit_stepping():
    # we check that we only do explicit
    # stepping here by putting
    # dt = 0.9 < delays
    def dde_f(t, y, args, history):
        return -history[0]

    t0, t1 = 0.0, 12.0
    ts = jnp.linspace(t0, t1, 120)
    delays = diffrax.Delays(
        delays=[lambda t, y, args: 1.0],
        initial_discontinuities=jnp.array([0.0]),
        max_discontinuities=10,
        recurrent_checking=False,
        rtol=10e-3,
        atol=10e-6,
    )

    dt0 = 0.1
    sol = diffrax.diffeqsolve(
        diffrax.ODETerm(dde_f),
        diffrax.Bosh3(),
        t0=ts[0],
        t1=ts[-1],
        dt0=dt0,
        y0=lambda t: 1.0,
        stepsize_controller=diffrax.PIDController(rtol=1e-6, atol=1e-9, dtmax=0.9),
        saveat=diffrax.SaveAt(ts=ts, dense=True),
        delays=delays,
    )

    assert sol.stats["num_dde_implicit_step"] == 0
    assert sol.stats["num_dde_explicit_step"] > 0


def test_hit_explicit_and_implicit_stepping():
    # we check that we only do implicit
    # stepping here by putting
    # dt=1.1 > delays
    def dde_f(t, y, args, history):
        return -history[0]

    t0, t1 = 0.0, 25.0
    ts = jnp.linspace(t0, t1, 120)
    delays = diffrax.Delays(
        delays=[lambda t, y, args: 1.0],
        initial_discontinuities=jnp.array([0.0]),
        max_discontinuities=10,
        recurrent_checking=False,
        rtol=10e-3,
        atol=10e-6,
    )

    sol = diffrax.diffeqsolve(
        diffrax.ODETerm(dde_f),
        diffrax.Dopri5(),
        t0=ts[0],
        t1=ts[-1],
        dt0=0.1,
        y0=lambda t: 1.0,
        stepsize_controller=diffrax.PIDController(rtol=1e-3, atol=1e-6),
        saveat=diffrax.SaveAt(ts=ts, dense=True),
        delays=delays,
    )
    assert sol.stats["num_dde_implicit_step"] > 0
    assert sol.stats["num_dde_explicit_step"] > 0


def test_basic_check_1():
    def vector_field(t, y, args, *, history):
        return y * (1 - history[0])

    y0_history = lambda t: 1.2

    delays = Delays(
        delays=[lambda t, y, args: 1.0],
        initial_discontinuities=jnp.array([0.0]),
        max_discontinuities=100,
        recurrent_checking=False,
        rtol=10e-3,
        atol=10e-6,
    )
    made_jump = delays.initial_discontinuities is None
    t0, t1 = 0.0, 50.0
    ts = jnp.linspace(t0, t1, 1001)
    sol = diffrax.diffeqsolve(
        diffrax.ODETerm(vector_field),
        diffrax.Tsit5(),
        t0=ts[0],
        t1=ts[-1],
        dt0=ts[1] - ts[0],
        y0=y0_history,
        stepsize_controller=diffrax.PIDController(rtol=1e-6, atol=1e-9),
        saveat=diffrax.SaveAt(ts=ts, dense=True),
        delays=delays,
        made_jump=made_jump,
    )

    delays = Delays(
        delays=[lambda t, y, args: 1.0],
        initial_discontinuities=jnp.array([0.0]),
        max_discontinuities=100,
        recurrent_checking=True,
        rtol=10e-3,
        atol=10e-6,
    )
    sol2 = diffrax.diffeqsolve(
        diffrax.ODETerm(vector_field),
        diffrax.Tsit5(),
        t0=ts[0],
        t1=ts[-1],
        dt0=ts[1] - ts[0],
        y0=y0_history,
        stepsize_controller=diffrax.PIDController(rtol=1e-6, atol=1e-9),
        saveat=diffrax.SaveAt(ts=ts, dense=True),
        delays=delays,
        made_jump=made_jump,
    )

    _, juliays = open_process_file("test/julia_dde/test_basic_check_1.txt")
    error1, error2, error3 = (
        jnp.mean(jnp.abs(juliays - sol.ys)),
        jnp.mean(jnp.abs(juliays - sol2.ys)),
        jnp.mean(jnp.abs(sol.ys - sol2.ys)),
    )

    assert error1 < 1e-4
    assert error2 < 1e-4
    assert error3 < 1e-5


def test_basic_check_2():
    def vector_field(t, y, args, *, history):
        return y * (1 - history[0])

    y0_history = lambda t: 1.2

    delays = Delays(
        delays=[lambda t, y, args: 2.0],
        initial_discontinuities=jnp.array([0.0]),
        max_discontinuities=100,
        recurrent_checking=False,
        rtol=10e-3,
        atol=10e-6,
    )
    made_jump = delays.initial_discontinuities is None
    t0, t1 = 0.0, 50.0
    ts = jnp.linspace(t0, t1, 1001)
    sol = diffrax.diffeqsolve(
        diffrax.ODETerm(vector_field),
        diffrax.Bosh3(),
        t0=ts[0],
        t1=ts[-1],
        dt0=ts[1] - ts[0],
        y0=y0_history,
        stepsize_controller=diffrax.PIDController(rtol=1e-6, atol=1e-9),
        saveat=diffrax.SaveAt(ts=ts, dense=True),
        delays=delays,
        made_jump=made_jump,
    )

    delays = Delays(
        delays=[lambda t, y, args: 2.0],
        initial_discontinuities=jnp.array([0.0]),
        max_discontinuities=100,
        recurrent_checking=True,
        rtol=10e-3,
        atol=10e-6,
    )
    sol2 = diffrax.diffeqsolve(
        diffrax.ODETerm(vector_field),
        diffrax.Bosh3(),
        t0=ts[0],
        t1=ts[-1],
        dt0=ts[1] - ts[0],
        y0=y0_history,
        stepsize_controller=diffrax.PIDController(rtol=1e-6, atol=1e-9),
        saveat=diffrax.SaveAt(ts=ts, dense=True),
        delays=delays,
        made_jump=made_jump,
    )

    _, juliays = open_process_file("test/julia_dde/test_basic_check_2.txt")
    error1, error2, error3 = (
        jnp.mean(jnp.abs(juliays - sol.ys)),
        jnp.mean(jnp.abs(juliays - sol2.ys)),
        jnp.mean(jnp.abs(sol.ys - sol2.ys)),
    )

    assert error1 < 1e-2
    assert error2 < 1e-2
    assert error3 < 1e-5


def test_basic_check_3():
    # same test as test_basic_check_2 but we
    # have a larger delay =3
    def vector_field(t, y, args, *, history):
        return y * (1 - history[0])

    y0_history = lambda t: 1.2

    delays = Delays(
        delays=[lambda t, y, args: 3.0],
        initial_discontinuities=jnp.array([0.0]),
        max_discontinuities=100,
        recurrent_checking=False,
        rtol=10e-3,
        atol=10e-6,
    )
    made_jump = delays.initial_discontinuities is None
    t0, t1 = 0.0, 50.0
    ts = jnp.linspace(t0, t1, 1001)
    sol = diffrax.diffeqsolve(
        diffrax.ODETerm(vector_field),
        diffrax.Bosh3(),
        t0=ts[0],
        t1=ts[-1],
        dt0=ts[1] - ts[0],
        y0=y0_history,
        stepsize_controller=diffrax.PIDController(rtol=1e-6, atol=1e-9),
        saveat=diffrax.SaveAt(ts=ts, dense=True),
        delays=delays,
        made_jump=made_jump,
    )

    delays = Delays(
        delays=[lambda t, y, args: 3.0],
        initial_discontinuities=jnp.array([0.0]),
        max_discontinuities=100,
        recurrent_checking=True,
        rtol=10e-3,
        atol=10e-6,
    )
    sol2 = diffrax.diffeqsolve(
        diffrax.ODETerm(vector_field),
        diffrax.Bosh3(),
        t0=ts[0],
        t1=ts[-1],
        dt0=ts[1] - ts[0],
        y0=y0_history,
        stepsize_controller=diffrax.PIDController(rtol=1e-6, atol=1e-9),
        saveat=diffrax.SaveAt(ts=ts, dense=True),
        delays=delays,
        made_jump=made_jump,
    )

    _, juliays = open_process_file("test/julia_dde/test_basic_check_3.txt")
    error1, error2, error3 = (
        jnp.mean(jnp.abs(juliays - sol.ys)),
        jnp.mean(jnp.abs(juliays - sol2.ys)),
        jnp.mean(jnp.abs(sol.ys - sol2.ys)),
    )

    assert error1 < 0.15
    assert error2 < 0.15
    assert error3 < 1e-5


def test_basic_check_4():
    # same experiment as test_basic_check_3
    # but solver is Tsit5
    def vector_field(t, y, args, *, history):
        return y * (1 - history[0])

    y0_history = lambda t: 1.2

    delays = Delays(
        delays=[lambda t, y, args: 3.0],
        initial_discontinuities=jnp.array([0.0]),
        max_discontinuities=100,
        recurrent_checking=False,
        rtol=10e-3,
        atol=10e-6,
    )
    made_jump = delays.initial_discontinuities is None
    t0, t1 = 0.0, 50.0
    ts = jnp.linspace(t0, t1, 1001)
    sol = diffrax.diffeqsolve(
        diffrax.ODETerm(vector_field),
        diffrax.Tsit5(),
        t0=ts[0],
        t1=ts[-1],
        dt0=ts[1] - ts[0],
        y0=y0_history,
        stepsize_controller=diffrax.PIDController(rtol=1e-6, atol=1e-9),
        saveat=diffrax.SaveAt(ts=ts, dense=True),
        delays=delays,
        made_jump=made_jump,
    )

    delays = Delays(
        delays=[lambda t, y, args: 3.0],
        initial_discontinuities=jnp.array([0.0]),
        max_discontinuities=100,
        recurrent_checking=True,
        rtol=10e-3,
        atol=10e-6,
    )
    sol2 = diffrax.diffeqsolve(
        diffrax.ODETerm(vector_field),
        diffrax.Tsit5(),
        t0=ts[0],
        t1=ts[-1],
        dt0=ts[1] - ts[0],
        y0=y0_history,
        stepsize_controller=diffrax.PIDController(rtol=1e-6, atol=1e-9),
        saveat=diffrax.SaveAt(ts=ts, dense=True),
        delays=delays,
        made_jump=made_jump,
    )

    _, juliays = open_process_file("test/julia_dde/test_basic_check_4.txt")
    error1, error2, error3 = (
        jnp.mean(jnp.abs(juliays - sol.ys)),
        jnp.mean(jnp.abs(juliays - sol2.ys)),
        jnp.mean(jnp.abs(sol.ys - sol2.ys)),
    )

    assert error1 < 1e-2
    assert error2 < 1e-2
    assert error3 < 1e-5


def test_basic_check_5():
    # same test as test_basic_check_3 but we
    # have a larger delay =4
    def vector_field(t, y, args, *, history):
        return y * (1 - history[0])

    y0_history = lambda t: 1.2

    delays = Delays(
        delays=[lambda t, y, args: 4.0],
        initial_discontinuities=jnp.array([0.0]),
        max_discontinuities=10,
        recurrent_checking=False,
        rtol=10e-3,
        atol=10e-6,
    )
    made_jump = delays.initial_discontinuities is None
    t0, t1 = 0.0, 50.0
    ts = jnp.linspace(t0, t1, 1001)
    sol = diffrax.diffeqsolve(
        diffrax.ODETerm(vector_field),
        diffrax.Bosh3(),
        t0=ts[0],
        t1=ts[-1],
        dt0=ts[1] - ts[0],
        y0=y0_history,
        stepsize_controller=diffrax.PIDController(rtol=1e-6, atol=1e-9),
        saveat=diffrax.SaveAt(ts=ts, dense=True),
        delays=delays,
        made_jump=made_jump,
    )

    delays = Delays(
        delays=[lambda t, y, args: 4.0],
        initial_discontinuities=jnp.array([0.0]),
        max_discontinuities=100,
        recurrent_checking=True,
        rtol=10e-3,
        atol=10e-6,
    )
    sol2 = diffrax.diffeqsolve(
        diffrax.ODETerm(vector_field),
        diffrax.Bosh3(),
        t0=ts[0],
        t1=ts[-1],
        dt0=ts[1] - ts[0],
        y0=y0_history,
        stepsize_controller=diffrax.PIDController(rtol=1e-6, atol=1e-9),
        saveat=diffrax.SaveAt(ts=ts, dense=True),
        delays=delays,
        made_jump=made_jump,
    )

    _, juliays = open_process_file("test/julia_dde/test_basic_check_5.txt")
    error1, error2, error3 = (
        jnp.mean(jnp.abs(juliays - sol.ys)),
        jnp.mean(jnp.abs(juliays - sol2.ys)),
        jnp.mean(jnp.abs(sol.ys - sol2.ys)),
    )

    assert error1 < 0.2
    assert error2 < 0.2
    assert error3 < 1e-5


def test_basic_check_6():
    # same experiment as test_basic_check_5
    # but solver is Tsit5
    def vector_field(t, y, args, *, history):
        return y * (1 - history[0])

    y0_history = lambda t: 1.2

    delays = Delays(
        delays=[lambda t, y, args: 4.0],
        initial_discontinuities=jnp.array([0.0]),
        max_discontinuities=100,
        recurrent_checking=False,
        rtol=10e-3,
        atol=10e-6,
    )
    made_jump = delays.initial_discontinuities is None
    t0, t1 = 0.0, 50.0
    ts = jnp.linspace(t0, t1, 1001)
    sol = diffrax.diffeqsolve(
        diffrax.ODETerm(vector_field),
        diffrax.Tsit5(),
        t0=ts[0],
        t1=ts[-1],
        dt0=ts[1] - ts[0],
        y0=y0_history,
        stepsize_controller=diffrax.PIDController(rtol=1e-6, atol=1e-9),
        saveat=diffrax.SaveAt(ts=ts, dense=True),
        delays=delays,
        made_jump=made_jump,
    )

    delays = Delays(
        delays=[lambda t, y, args: 4.0],
        initial_discontinuities=jnp.array([0.0]),
        max_discontinuities=100,
        recurrent_checking=True,
        rtol=10e-3,
        atol=10e-6,
    )
    sol2 = diffrax.diffeqsolve(
        diffrax.ODETerm(vector_field),
        diffrax.Tsit5(),
        t0=ts[0],
        t1=ts[-1],
        dt0=ts[1] - ts[0],
        y0=y0_history,
        stepsize_controller=diffrax.PIDController(rtol=1e-6, atol=1e-9),
        saveat=diffrax.SaveAt(ts=ts, dense=True),
        delays=delays,
        made_jump=made_jump,
    )

    _, juliays = open_process_file("test/julia_dde/test_basic_check_6.txt")
    error1, error2, error3 = (
        jnp.mean(jnp.abs(juliays - sol.ys)),
        jnp.mean(jnp.abs(juliays - sol2.ys)),
        jnp.mean(jnp.abs(sol.ys - sol2.ys)),
    )

    assert error1 < 1e-2
    assert error2 < 1e-2
    assert error3 < 1e-5


def test_basic_check_7():
    # same experiment as test_basic_check_7
    # but solver is implicit
    def vector_field(t, y, args, *, history):
        return y * (1 - history[0])

    y0_history = lambda t: 1.2

    delays = Delays(
        delays=[lambda t, y, args: 4.0],
        initial_discontinuities=jnp.array([0.0]),
        max_discontinuities=100,
        recurrent_checking=False,
        rtol=10e-3,
        atol=10e-6,
    )
    made_jump = delays.initial_discontinuities is None
    t0, t1 = 0.0, 50.0
    ts = jnp.linspace(t0, t1, 1001)
    sol = diffrax.diffeqsolve(
        diffrax.ODETerm(vector_field),
        diffrax.Kvaerno5(),
        t0=ts[0],
        t1=ts[-1],
        dt0=ts[1] - ts[0],
        y0=y0_history,
        stepsize_controller=diffrax.PIDController(rtol=1e-6, atol=1e-9),
        saveat=diffrax.SaveAt(ts=ts, dense=True),
        delays=delays,
        made_jump=made_jump,
    )

    delays = Delays(
        delays=[lambda t, y, args: 4.0],
        initial_discontinuities=jnp.array([0.0]),
        max_discontinuities=100,
        recurrent_checking=True,
        rtol=10e-3,
        atol=10e-6,
    )
    sol2 = diffrax.diffeqsolve(
        diffrax.ODETerm(vector_field),
        diffrax.Kvaerno5(),
        t0=ts[0],
        t1=ts[-1],
        dt0=ts[1] - ts[0],
        y0=y0_history,
        stepsize_controller=diffrax.PIDController(rtol=1e-6, atol=1e-9),
        saveat=diffrax.SaveAt(ts=ts, dense=True),
        delays=delays,
        made_jump=made_jump,
    )

    _, juliays = open_process_file("test/julia_dde/test_basic_check_7.txt")
    error1, error2, error3 = (
        jnp.mean(jnp.abs(juliays - sol.ys)),
        jnp.mean(jnp.abs(juliays - sol2.ys)),
        jnp.mean(jnp.abs(sol.ys - sol2.ys)),
    )

    assert error1 < 1e-1
    assert error2 < 1e-1
    assert error3 < 1e-2


def test_basic_check_8():
    # new system with 2 delays

    def vector_field(t, y, args, *, history):
        return -history[0] - history[1]

    y0_history = lambda t: 1.2

    delays = Delays(
        delays=[lambda t, y, args: 1 / 5, lambda t, y, args: 1 / 3],
        initial_discontinuities=jnp.array([0.0]),
        max_discontinuities=100,
        recurrent_checking=False,
        rtol=10e-3,
        atol=10e-6,
    )
    made_jump = delays.initial_discontinuities is None
    t0, t1 = 0.0, 10.0
    ts = jnp.linspace(t0, t1, 101)
    sol = diffrax.diffeqsolve(
        diffrax.ODETerm(vector_field),
        diffrax.Tsit5(),
        t0=ts[0],
        t1=ts[-1],
        dt0=ts[1] - ts[0],
        y0=y0_history,
        stepsize_controller=diffrax.PIDController(rtol=1e-6, atol=1e-9),
        saveat=diffrax.SaveAt(ts=ts, dense=True),
        delays=delays,
        made_jump=made_jump,
    )

    delays = Delays(
        delays=[lambda t, y, args: 1 / 5, lambda t, y, args: 1 / 3],
        initial_discontinuities=jnp.array([0.0]),
        max_discontinuities=1000,
        recurrent_checking=True,
        rtol=10e-3,
        atol=10e-6,
    )
    sol2 = diffrax.diffeqsolve(
        diffrax.ODETerm(vector_field),
        diffrax.Tsit5(),
        t0=ts[0],
        t1=ts[-1],
        dt0=ts[1] - ts[0],
        y0=y0_history,
        stepsize_controller=diffrax.PIDController(rtol=1e-6, atol=1e-9),
        saveat=diffrax.SaveAt(ts=ts, dense=True),
        delays=delays,
        made_jump=made_jump,
    )

    _, juliays = open_process_file("test/julia_dde/test_basic_check_8.txt")
    error1, error2, error3 = (
        jnp.mean(jnp.abs(juliays - sol.ys)),
        jnp.mean(jnp.abs(juliays - sol2.ys)),
        jnp.mean(jnp.abs(sol.ys - sol2.ys)),
    )

    assert error1 < 1e-6
    assert error2 < 1e-6
    assert error3 < 1e-6


def test_basic_check_9():
    # new system ie Mackey Glass

    def vector_field(t, y, args, *, history):
        return 0.2 * (history[0]) / (1 + history[0] ** 10) - 0.1 * y

    y0_history = lambda t: 1.2

    delays = Delays(
        delays=[lambda t, y, args: 6.0],
        initial_discontinuities=jnp.array([0.0]),
        max_discontinuities=100,
        recurrent_checking=False,
        rtol=10e-3,
        atol=10e-6,
    )

    t0, t1, nb_points = 0.0, 50.0, 501
    ts = jnp.linspace(t0, t1, nb_points)
    sol = diffrax.diffeqsolve(
        diffrax.ODETerm(vector_field),
        diffrax.Tsit5(),
        t0=ts[0],
        t1=ts[-1],
        dt0=ts[1] - ts[0],
        y0=y0_history,
        stepsize_controller=diffrax.PIDController(rtol=1e-6, atol=1e-9),
        saveat=diffrax.SaveAt(ts=ts, dense=True),
        delays=delays,
    )

    delays = Delays(
        delays=[lambda t, y, args: 6.0],
        initial_discontinuities=jnp.array([0.0]),
        max_discontinuities=100,
        recurrent_checking=True,
        rtol=10e-3,
        atol=10e-6,
    )

    sol2 = diffrax.diffeqsolve(
        diffrax.ODETerm(vector_field),
        diffrax.Tsit5(),
        t0=ts[0],
        t1=ts[-1],
        dt0=ts[1] - ts[0],
        y0=y0_history,
        stepsize_controller=diffrax.PIDController(rtol=1e-6, atol=1e-9),
        saveat=diffrax.SaveAt(ts=ts, dense=True),
        delays=delays,
    )

    _, juliays = open_process_file("test/julia_dde/test_basic_check_9.txt")
    error1, error2, error3 = (
        jnp.mean(jnp.abs(juliays - sol.ys)),
        jnp.mean(jnp.abs(juliays - sol2.ys)),
        jnp.mean(jnp.abs(sol.ys - sol2.ys)),
    )

    assert error1 < 1e-4
    assert error2 < 1e-4
    assert error3 < 1e-4


def test_basic_check_10():
    # testing a time dependent equation

    def vector_field(t, y, args, *, history):
        return y * (1 - history[0])

    y0_history = lambda t: 1.2

    delays = Delays(
        delays=[lambda t, y, args: 2 + jnp.sin(t)],
        initial_discontinuities=jnp.array([0.0]),
        max_discontinuities=100,
        recurrent_checking=False,
        rtol=10e-3,
        atol=10e-6,
    )

    t0, t1, nb_points = 0.0, 40.0, 401
    ts = jnp.linspace(t0, t1, nb_points)
    sol = diffrax.diffeqsolve(
        diffrax.ODETerm(vector_field),
        diffrax.Bosh3(),
        t0=ts[0],
        t1=ts[-1],
        dt0=ts[1] - ts[0],
        y0=y0_history,
        stepsize_controller=diffrax.PIDController(rtol=1e-6, atol=1e-9),
        saveat=diffrax.SaveAt(ts=ts, dense=True),
        delays=delays,
    )

    delays = Delays(
        delays=[lambda t, y, args: 2 + jnp.sin(t)],
        initial_discontinuities=jnp.array([0.0]),
        max_discontinuities=100,
        recurrent_checking=True,
        rtol=10e-3,
        atol=10e-6,
    )

    sol2 = diffrax.diffeqsolve(
        diffrax.ODETerm(vector_field),
        diffrax.Bosh3(),
        t0=ts[0],
        t1=ts[-1],
        dt0=ts[1] - ts[0],
        y0=y0_history,
        stepsize_controller=diffrax.PIDController(rtol=1e-6, atol=1e-9),
        saveat=diffrax.SaveAt(ts=ts, dense=True),
        delays=delays,
    )

    _, juliays = open_process_file("test/julia_dde/test_basic_check_10.txt")
    error1, error2, error3 = (
        jnp.mean(jnp.abs(juliays - sol.ys)),
        jnp.mean(jnp.abs(juliays - sol2.ys)),
        jnp.mean(jnp.abs(sol.ys - sol2.ys)),
    )

    assert error1 < 1e-3
    assert error2 < 1e-3
    assert error3 < 1e-5


def test_basic_numerical_check_1():
    # testing a dde where we know its analytical value
    # http://www.cs.toronto.edu/pub/reports/na/hzpEnrightNA09Preprint.pdf
    # test problem 1

    def vector_field(t, y, args, history):
        return history[0]

    y0_history = lambda t: lax.cond(t < 2.0, lambda: 0.5, lambda: 1.0)

    delays = Delays(
        delays=[lambda t, y, args: t - y],
        initial_discontinuities=jnp.array([2.0]),
        max_discontinuities=100,
        recurrent_checking=False,
        rtol=1e-3,
        atol=1e-6,
    )

    t0, t1, nb_points = 2.0, 5.5, 350
    ts = jnp.linspace(t0, t1, nb_points)
    sol = diffrax.diffeqsolve(
        diffrax.ODETerm(vector_field),
        diffrax.Bosh3(),
        t0=ts[0],
        t1=ts[-1],
        dt0=ts[1] - ts[0],
        y0=y0_history,
        stepsize_controller=diffrax.PIDController(rtol=1e-6, atol=1e-9),
        saveat=diffrax.SaveAt(ts=ts, dense=True),
        delays=delays,
    )

    delays = Delays(
        delays=[lambda t, y, args: t - y],
        initial_discontinuities=jnp.array([2.0]),
        max_discontinuities=100,
        recurrent_checking=True,
        rtol=1e-3,
        atol=1e-6,
    )

    sol2 = diffrax.diffeqsolve(
        diffrax.ODETerm(vector_field),
        diffrax.Bosh3(),
        t0=ts[0],
        t1=ts[-1],
        dt0=ts[1] - ts[0],
        y0=y0_history,
        stepsize_controller=diffrax.PIDController(rtol=1e-6, atol=1e-9),
        saveat=diffrax.SaveAt(ts=ts, dense=True),
        delays=delays,
    )

    def f1(t):
        return t / 2

    def f2(t):
        return 2 * jnp.exp(t / 2 - 2)

    # y(t) = t/2 for 2 <= t <= 4
    # y(t) = 2 exp(t/2-2) for 4 <= t <= 5.386
    # y(t) = 4 - 2 ln(1+ 5.386 -t )] for 5.386 <= t <= 5.5
    error1 = jnp.sum(jnp.abs(sol.ys[:200] - f1(sol.ts[:200])))
    error2 = jnp.sum(jnp.abs(sol.ys[202:300] - f2(sol.ts[202:300])))

    error11 = jnp.sum(jnp.abs(sol2.ys[:200] - f1(sol2.ts[:200])))
    error21 = jnp.sum(jnp.abs(sol2.ys[202:300] - f2(sol2.ts[202:300])))
    juliats, juliays = open_process_file(
        "test/julia_dde/test_basic_numerical_check_1.txt"
    )

    error3 = jnp.sum(jnp.abs(juliays[:200] - f1(juliats[:200])))
    error4 = jnp.sum(jnp.abs(juliays[200:300] - f2(juliats[200:300])))
    assert error2 < error4
    assert error1 < 1e-5
    assert error2 < 1e-2
    assert error11 < 1e-5
    assert error21 < 1e-2
    assert error3 < 1e-5


def test_basic_numerical_check_2():
    # testing a dde where we know its analytical value
    # http://www.cs.toronto.edu/pub/reports/na/hzpEnrightNA09Preprint.pdf
    # test problem 3

    def vector_field(t, y, args, history):
        return y * history[0] / t

    y0_history = lambda t: 1.0

    delays = Delays(
        delays=[lambda t, y, args: t - jnp.log(y)],
        initial_discontinuities=jnp.array([1.0]),
        max_discontinuities=100,
        recurrent_checking=False,
        rtol=1e-3,
        atol=1e-6,
    )

    t0, t1, nb_points = 1.0, 10.0, 901
    ts = jnp.linspace(t0, t1, nb_points)
    sol = diffrax.diffeqsolve(
        diffrax.ODETerm(vector_field),
        diffrax.Bosh3(),
        t0=ts[0],
        t1=ts[-1],
        dt0=ts[1] - ts[0],
        y0=y0_history,
        stepsize_controller=diffrax.PIDController(rtol=1e-6, atol=1e-9),
        saveat=diffrax.SaveAt(ts=ts, dense=True),
        delays=delays,
    )

    delays = Delays(
        delays=[lambda t, y, args: t - jnp.log(y)],
        initial_discontinuities=jnp.array([1.0]),
        max_discontinuities=100,
        recurrent_checking=True,
        rtol=1e-3,
        atol=1e-6,
    )

    sol2 = diffrax.diffeqsolve(
        diffrax.ODETerm(vector_field),
        diffrax.Bosh3(),
        t0=ts[0],
        t1=ts[-1],
        dt0=ts[1] - ts[0],
        y0=y0_history,
        stepsize_controller=diffrax.PIDController(rtol=1e-6, atol=1e-9),
        saveat=diffrax.SaveAt(ts=ts, dense=True),
        delays=delays,
    )

    def f1(t):
        return t

    def f2(t):
        return jnp.exp(t / jnp.exp(1))

    # y(t) = t for 1 <= t <= e
    # y(t) = exp(t/e) for e <= t <= e^2
    error1 = jnp.sum(jnp.abs(sol.ys[:100] - f1(sol.ts[:100])))
    error2 = jnp.sum(jnp.abs(sol.ys[300:400] - f2(sol.ts[300:400])))

    error11 = jnp.sum(jnp.abs(sol2.ys[:100] - f1(sol2.ts[:100])))
    error21 = jnp.sum(jnp.abs(sol2.ys[300:400] - f2(sol2.ts[300:400])))
    juliats, juliays = open_process_file(
        "test/julia_dde/test_basic_numerical_check_2.txt"
    )

    error3 = jnp.sum(jnp.abs(juliays[:100] - f1(juliats[:100])))
    error4 = jnp.sum(jnp.abs(juliays[300:400] - f2(juliats[300:400])))

    assert error1 < 1e-6
    assert error11 < 1e-6
    assert error3 < 1e-6
    assert error2 < 1e-2
    assert error21 < 1e-2
    assert error2 < error4
