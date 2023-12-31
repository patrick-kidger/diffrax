import contextlib
import math

import diffrax
import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

from .helpers import tree_allclose


def test_results():
    assert len(diffrax.RESULTS) > 5
    assert isinstance(diffrax.RESULTS[diffrax.RESULTS.max_steps_reached], str)

    # In principle no code should rely on this, but in practice something may slip
    # through the cracks so it's worth checking anyway.
    assert diffrax.RESULTS.successful._value == 0


_t0 = jnp.array(0.1)
_t1 = jnp.array(1.1)
_y0 = jnp.array([2.1])


def _integrate(saveat):
    term = diffrax.ODETerm(lambda t, y, args: -0.5 * y)
    stepsize_controller = diffrax.PIDController(rtol=1e-8, atol=1e-8)
    return diffrax.diffeqsolve(
        term,
        t0=_t0,
        t1=_t1,
        y0=_y0,
        dt0=None,
        solver=diffrax.Dopri5(),
        saveat=saveat,
        stepsize_controller=stepsize_controller,
    )


def test_saveat_solution():
    saveat = diffrax.SaveAt(t0=True)
    sol = _integrate(saveat)
    assert sol.t0 == _t0
    assert sol.t1 == _t1
    assert sol.ts.shape == (1,)  # pyright: ignore
    assert sol.ys.shape == (1, 1)  # pyright: ignore
    assert sol.ts[0] == _t0  # pyright: ignore
    assert sol.ys[0, 0] == _y0  # pyright: ignore
    assert sol.controller_state is None
    assert sol.solver_state is None
    with pytest.raises(ValueError):
        sol.evaluate(0.2, 0.8)
    with pytest.raises(ValueError):
        sol.derivative(0.2)
    assert sol.stats["num_steps"] > 0
    assert sol.result == diffrax.RESULTS.successful

    for controller_state in (True, False):
        for solver_state in (True, False):
            saveat = diffrax.SaveAt(
                t1=True, solver_state=solver_state, controller_state=controller_state
            )
            sol = _integrate(saveat)
            assert sol.t0 == _t0
            assert sol.t1 == _t1
            assert sol.ts.shape == (1,)  # pyright: ignore
            assert sol.ys.shape == (1, 1)  # pyright: ignore
            assert sol.ts[0] == _t1  # pyright: ignore
            assert tree_allclose(sol.ys[0], _y0 * math.exp(-0.5))  # pyright: ignore
            if controller_state:
                assert sol.controller_state is not None
            else:
                assert sol.controller_state is None
            if solver_state:
                assert sol.solver_state is not None
            else:
                assert sol.solver_state is None
            with pytest.raises(ValueError):
                sol.evaluate(0.2, 0.8)
            with pytest.raises(ValueError):
                sol.derivative(0.2)
            assert sol.stats["num_steps"] > 0
            assert sol.result == diffrax.RESULTS.successful

    # Outside [t0, t1]
    saveat = diffrax.SaveAt(ts=[0])
    with pytest.raises(RuntimeError):
        sol = _integrate(saveat)
    saveat = diffrax.SaveAt(ts=[3])
    with pytest.raises(RuntimeError):
        sol = _integrate(saveat)

    saveat = diffrax.SaveAt(ts=[0.5, 0.8])
    sol = _integrate(saveat)
    assert sol.t0 == _t0
    assert sol.t1 == _t1
    assert sol.ts.shape == (2,)  # pyright: ignore
    assert sol.ys.shape == (2, 1)  # pyright: ignore
    assert sol.ts[0] == jnp.asarray(0.5)  # pyright: ignore
    assert sol.ts[1] == jnp.asarray(0.8)  # pyright: ignore
    assert tree_allclose(sol.ys[0], _y0 * math.exp(-0.2))  # pyright: ignore
    assert tree_allclose(sol.ys[1], _y0 * math.exp(-0.35))  # pyright: ignore
    assert sol.controller_state is None
    assert sol.solver_state is None
    with pytest.raises(ValueError):
        sol.evaluate(0.2, 0.8)
    with pytest.raises(ValueError):
        sol.derivative(0.2)
    assert sol.stats["num_steps"] > 0
    assert sol.result == diffrax.RESULTS.successful

    saveat = diffrax.SaveAt(steps=True)
    sol = _integrate(saveat)
    assert sol.t0 == _t0
    assert sol.t1 == _t1
    assert sol.ts.shape == (4096,)  # pyright: ignore
    assert sol.ys.shape == (4096, 1)  # pyright: ignore
    _ts = jnp.where(sol.ts == jnp.inf, jnp.nan, sol.ts)
    with jax.numpy_rank_promotion("allow"):
        _ys = _y0 * jnp.exp(-0.5 * (_ts - _t0))[:, None]
    _ys = jnp.where(jnp.isnan(_ys), jnp.inf, _ys)
    assert tree_allclose(sol.ys, _ys)
    assert sol.controller_state is None
    assert sol.solver_state is None
    with pytest.raises(ValueError):
        sol.evaluate(0.2, 0.8)
    with pytest.raises(ValueError):
        sol.derivative(0.2)
    assert sol.stats["num_steps"] > 0
    assert sol.result == diffrax.RESULTS.successful

    saveat = diffrax.SaveAt(dense=True)
    sol = _integrate(saveat)
    assert sol.t0 == _t0
    assert sol.t1 == _t1
    assert sol.ts is None
    assert sol.ys is None
    assert sol.controller_state is None
    assert sol.solver_state is None
    assert tree_allclose(sol.evaluate(0.2, 0.8), sol.evaluate(0.8) - sol.evaluate(0.2))
    assert tree_allclose(sol.evaluate(0.2), _y0 * math.exp(-0.05))
    assert tree_allclose(sol.evaluate(0.8), _y0 * math.exp(-0.35))
    assert tree_allclose(sol.derivative(0.2), -0.5 * _y0 * math.exp(-0.05))
    assert sol.stats["num_steps"] > 0
    assert sol.result == diffrax.RESULTS.successful


def test_trivial_dense():
    term = diffrax.ODETerm(lambda t, y, args: -0.5 * y)
    y0 = jnp.array([2.1])
    saveat = diffrax.SaveAt(dense=True)
    stepsize_controller = diffrax.PIDController(rtol=1e-8, atol=1e-8)
    sol = diffrax.diffeqsolve(
        term,
        t0=2.0,
        t1=2.0,
        y0=y0,
        dt0=None,
        solver=diffrax.Dopri5(),
        saveat=saveat,
        stepsize_controller=stepsize_controller,
    )
    assert tree_allclose(sol.evaluate(2.0), y0)


@pytest.mark.parametrize(
    "adjoint",
    [
        diffrax.RecursiveCheckpointAdjoint(),
        diffrax.DirectAdjoint(),
        diffrax.ImplicitAdjoint(),
        diffrax.BacksolveAdjoint(),
    ],
)
@pytest.mark.parametrize("multi_subs", [True, False])
@pytest.mark.parametrize("with_fn", [True, False])
def test_subsaveat(adjoint, multi_subs, with_fn, getkey):
    if with_fn:
        mlp = eqx.nn.MLP(3, 1, 32, 2, key=getkey())
        apply = lambda _, x, __: mlp(x)
        subsaveat_kwargs: dict = dict(fn=apply)
    else:
        mlp = lambda x: x
        subsaveat_kwargs: dict = dict()
    get2 = diffrax.SubSaveAt(t0=True, ts=jnp.linspace(0.5, 1.5, 3), **subsaveat_kwargs)
    if multi_subs:
        get0 = diffrax.SubSaveAt(steps=True, fn=lambda _, y, __: y[0])
        get1 = diffrax.SubSaveAt(
            ts=jnp.linspace(0, 1, 5), t1=True, fn=lambda _, y, __: y[1]
        )
        subs = (get0, get1, get2)
    else:
        subs = get2

    context = contextlib.nullcontext()
    if isinstance(adjoint, diffrax.ImplicitAdjoint):
        context = pytest.raises(ValueError)
    elif isinstance(adjoint, diffrax.BacksolveAdjoint):
        if with_fn or multi_subs:
            context = pytest.raises(NotImplementedError)

    term = diffrax.ODETerm(lambda t, y, args: -0.5 * y)
    y0 = jnp.array([2.1, 1.1, 0.1])
    saveat = diffrax.SaveAt(subs=subs)
    stepsize_controller = diffrax.PIDController(rtol=1e-8, atol=1e-8)

    with context:
        sol = diffrax.diffeqsolve(
            term,
            t0=0,
            t1=2,
            y0=y0,
            dt0=None,
            solver=diffrax.Dopri5(),
            saveat=saveat,
            stepsize_controller=stepsize_controller,
            adjoint=adjoint,
        )
        steps = sol.stats["num_accepted_steps"]

        sol2 = diffrax.diffeqsolve(
            term,
            t0=0,
            t1=2,
            y0=y0,
            dt0=None,
            solver=diffrax.Dopri5(),
            saveat=diffrax.SaveAt(dense=True),
            stepsize_controller=stepsize_controller,
        )

        if multi_subs:
            ts0, ts1, ts2 = sol.ts  # pyright: ignore
            ys0, ys1, ys2 = sol.ys  # pyright: ignore
            assert ts0.shape == (4096,)
            assert tree_allclose(ts1, jnp.array([0, 0.25, 0.5, 0.75, 1, 2]))
            assert tree_allclose(
                ys0[:steps], jax.vmap(sol2.evaluate)(ts0[:steps])[:, 0]
            )
            assert tree_allclose(ys1, jax.vmap(sol2.evaluate)(ts1)[:, 1])
        else:
            ts2 = sol.ts
            ys2 = sol.ys
        assert tree_allclose(ts2, jnp.array([0, 0.5, 1.0, 1.5]))
        assert tree_allclose(ys2, jax.vmap(mlp)(jax.vmap(sol2.evaluate)(ts2)))  # pyright: ignore


def test_backprop_none_subs():
    saveat = diffrax.SaveAt(dense=True)

    @jax.grad
    def run(y0):
        sol = diffrax.diffeqsolve(
            diffrax.ODETerm(lambda t, y, args: -y),
            diffrax.Tsit5(),
            0,
            1,
            0.1,
            y0,
            saveat=saveat,
        )
        return sol.evaluate(0.5)

    run(1.0)
