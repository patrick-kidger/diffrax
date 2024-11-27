from typing import cast

import diffrax
import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import optimistix as optx
import pytest
from diffrax._adjoint import _Reversible
from jaxtyping import Array

from .helpers import tree_allclose


class _VectorField(eqx.Module):
    nondiff_arg: int
    diff_arg: float

    def __call__(self, t, y, args):
        assert y.shape == (2,)
        diff_arg, nondiff_arg = args
        dya = diff_arg * y[0] + nondiff_arg * y[1]
        dyb = self.nondiff_arg * y[0] + self.diff_arg * y[1]
        return jnp.stack([dya, dyb])


class _PyTreeVectorField(eqx.Module):
    nondiff_arg: int
    diff_arg: float

    def __call__(self, t, y, args):
        diff_arg, nondiff_arg = args
        dya = diff_arg * y[0] + nondiff_arg * y[1][0]
        dyb = self.nondiff_arg * y[0] + self.diff_arg * y[1][0]
        dyc = diff_arg * y[1][1] + nondiff_arg * y[1][0]
        return (dya, (dyb, dyc))


class QuadraticPath(diffrax.AbstractPath):
    @property
    def t0(self):
        return 0

    @property
    def t1(self):
        return 3

    def evaluate(self, t0, t1=None, left=True):
        del left
        if t1 is not None:
            return self.evaluate(t1) - self.evaluate(t0)
        return t0**2


def _compare_solve(y0__args__term, solver, stepsize_controller):
    y0, args, term = y0__args__term
    sol = diffrax.diffeqsolve(
        term,
        solver,
        t0=0,
        t1=5,
        dt0=0.01,
        y0=y0,
        args=args,
        stepsize_controller=stepsize_controller,
    )
    y1_base = sol.ys

    # Reversible
    sol = diffrax.diffeqsolve(
        term,
        solver,
        t0=0,
        t1=5,
        dt0=0.01,
        y0=y0,
        args=args,
        adjoint=diffrax.ReversibleAdjoint(),
        stepsize_controller=stepsize_controller,
    )
    y1_rev = sol.ys

    assert tree_allclose(y1_base, y1_rev, atol=1e-5)


@eqx.filter_value_and_grad
def _loss(y0__args__term, solver, saveat, adjoint, stepsize_controller, pytree_state):
    y0, args, term = y0__args__term

    sol = diffrax.diffeqsolve(
        term,
        solver,
        t0=0,
        t1=5,
        dt0=0.01,
        y0=y0,
        args=args,
        saveat=saveat,
        adjoint=adjoint,
        stepsize_controller=stepsize_controller,
    )
    if pytree_state:
        y1, (y2, y3) = sol.ys  # type: ignore
        y1 = y1 + y2 + y3
    else:
        y1 = sol.ys
    return jnp.sum(cast(Array, y1))


# The adjoint comparison looks wrong at first glance so here's an explanation:
# We want to check that the gradients calculated by ReversibleAdjoint
# are the same as those calculated by RecursiveCheckpointAdjoint, for a fixed
# solver.
#
# ReversibleAdjoint auto-wraps the solver to create a reversible version. So when
# calculating gradients we use base_solver + ReversibleAdjoint and reversible_solver +
# RecursiveCheckpointAdjoint, to ensure that the reversible solver is fixed across both
# adjoints.
def _compare_grads(
    y0__args__term, base_solver, saveat, stepsize_controller, pytree_state
):
    reversible_solver = _Reversible(base_solver)

    loss, grads_base = _loss(
        y0__args__term,
        reversible_solver,
        saveat,
        adjoint=diffrax.RecursiveCheckpointAdjoint(),
        stepsize_controller=stepsize_controller,
        pytree_state=pytree_state,
    )
    loss, grads_reversible = _loss(
        y0__args__term,
        base_solver,
        saveat,
        adjoint=diffrax.ReversibleAdjoint(),
        stepsize_controller=stepsize_controller,
        pytree_state=pytree_state,
    )

    assert tree_allclose(grads_base, grads_reversible, atol=1e-5)


@pytest.mark.parametrize(
    "solver",
    [diffrax.Tsit5(), diffrax.Kvaerno5(), diffrax.KenCarp5()],
)
@pytest.mark.parametrize(
    "stepsize_controller",
    [diffrax.ConstantStepSize(), diffrax.PIDController(rtol=1e-8, atol=1e-8)],
)
@pytest.mark.parametrize("pytree_state", [True, False])
def test_forward_solve(solver, stepsize_controller, pytree_state):
    if pytree_state:
        y0 = (jnp.array(0.9), (jnp.array(5.4), jnp.array(1.2)))
        term = diffrax.ODETerm(_PyTreeVectorField(nondiff_arg=1, diff_arg=-0.1))
    else:
        y0 = jnp.array([0.9, 5.4])
        term = diffrax.ODETerm(_VectorField(nondiff_arg=1, diff_arg=-0.1))

    if isinstance(stepsize_controller, diffrax.ConstantStepSize) and isinstance(
        solver, diffrax.AbstractImplicitSolver
    ):
        return

    if isinstance(solver, diffrax.KenCarp5):
        term = diffrax.MultiTerm(term, term)

    args = (0.1, -1)
    y0__args__term = (y0, args, term)
    _compare_solve(y0__args__term, solver, stepsize_controller)


@pytest.mark.parametrize(
    "solver",
    [diffrax.Tsit5(), diffrax.Kvaerno5(), diffrax.KenCarp5()],
)
@pytest.mark.parametrize(
    "stepsize_controller",
    [diffrax.ConstantStepSize(), diffrax.PIDController(rtol=1e-8, atol=1e-8)],
)
@pytest.mark.parametrize(
    "saveat",
    [
        diffrax.SaveAt(t1=True),
        diffrax.SaveAt(steps=True),
        diffrax.SaveAt(t0=True, steps=True),
    ],
)
@pytest.mark.parametrize("pytree_state", [True, False])
def test_reversible_adjoint(solver, stepsize_controller, saveat, pytree_state):
    if pytree_state:
        y0 = (jnp.array(0.9), (jnp.array(5.4), jnp.array(1.2)))
        term = diffrax.ODETerm(_PyTreeVectorField(nondiff_arg=1, diff_arg=-0.1))
    else:
        y0 = jnp.array([0.9, 5.4])
        term = diffrax.ODETerm(_VectorField(nondiff_arg=1, diff_arg=-0.1))

    if isinstance(stepsize_controller, diffrax.ConstantStepSize) and isinstance(
        solver, diffrax.AbstractImplicitSolver
    ):
        return

    if isinstance(solver, diffrax.KenCarp5):
        term = diffrax.MultiTerm(term, term)

    args = (0.1, -1)
    y0__args__term = (y0, args, term)
    del y0, args, term

    _compare_grads(y0__args__term, solver, saveat, stepsize_controller, pytree_state)


@pytest.mark.parametrize(
    "solver, diffusion",
    [
        (diffrax.ShARK(), lambda t, y, args: 1.0),
        (diffrax.SlowRK(), lambda t, y, args: 0.1 * y),
    ],
)
@pytest.mark.parametrize("adjoint", [False, True])
def test_sde(solver, diffusion, adjoint):
    y0 = jnp.array([0.9, 5.4])
    args = (0.1, -1)
    drift = diffrax.ODETerm(_VectorField(nondiff_arg=1, diff_arg=-0.1))
    brownian_motion = diffrax.VirtualBrownianTree(
        0, 5, tol=1e-3, shape=(), levy_area=diffrax.SpaceTimeLevyArea, key=jr.PRNGKey(0)
    )
    terms = diffrax.MultiTerm(drift, diffrax.ControlTerm(diffusion, brownian_motion))
    y0__args__term = (y0, args, terms)
    stepsize_controller = diffrax.ConstantStepSize()

    if adjoint:
        saveat = diffrax.SaveAt(t1=True)
        _compare_grads(y0__args__term, solver, saveat, stepsize_controller, False)

    else:
        _compare_solve(y0__args__term, solver, stepsize_controller)


@pytest.mark.parametrize(
    "solver",
    [diffrax.Tsit5(), diffrax.Kvaerno5(), diffrax.KenCarp5()],
)
@pytest.mark.parametrize(
    "stepsize_controller",
    [diffrax.ConstantStepSize(), diffrax.PIDController(rtol=1e-8, atol=1e-8)],
)
def test_cde(solver, stepsize_controller):
    if isinstance(stepsize_controller, diffrax.ConstantStepSize) and isinstance(
        solver, diffrax.AbstractImplicitSolver
    ):
        return

    vf = _VectorField(nondiff_arg=1, diff_arg=-0.1)
    control = diffrax.ControlTerm(vf, QuadraticPath())
    terms = diffrax.MultiTerm(control, diffrax.ODETerm(vf))
    y0 = jnp.array([0.9, 5.4])
    args = (0.1, -1)
    y0__args__term = (y0, args, terms)
    _compare_solve(y0__args__term, solver, stepsize_controller)


def test_events():
    def vector_field(t, y, args):
        _, v = y
        return jnp.array([v, -8.0])

    def cond_fn(t, y, args, **kwargs):
        x, _ = y
        return x

    @eqx.filter_value_and_grad
    def _event_loss(y0, adjoint):
        sol = diffrax.diffeqsolve(
            term, solver, t0, t1, dt0, y0, adjoint=adjoint, event=event
        )
        return cast(Array, sol.ys)[0, 1]

    y0 = jnp.array([10.0, 0.0])
    t0 = 0
    t1 = jnp.inf
    dt0 = 0.1
    term = diffrax.ODETerm(vector_field)
    root_finder = optx.Newton(1e-5, 1e-5, optx.rms_norm)
    event = diffrax.Event(cond_fn, root_finder)
    solver = diffrax.Tsit5()

    msg = "`diffrax.ReversibleAdjoint` is not compatible with events."
    with pytest.raises(NotImplementedError, match=msg):
        _event_loss(y0, adjoint=diffrax.ReversibleAdjoint())


@pytest.mark.parametrize(
    "saveat",
    [
        diffrax.SaveAt(ts=jnp.linspace(0, 5)),
        diffrax.SaveAt(dense=True),
        diffrax.SaveAt(t0=True),
        diffrax.SaveAt(ts=jnp.linspace(0, 5), fn=lambda t, y, args: t),
    ],
)
@pytest.mark.parametrize(
    "solver",
    [diffrax.SemiImplicitEuler(), diffrax.ReversibleHeun(), diffrax.LeapfrogMidpoint()],
)
def test_incompatible_arguments(solver, saveat):
    y0 = jnp.array([0.9, 5.4])
    args = (0.1, -1)
    term = diffrax.ODETerm(_VectorField(nondiff_arg=1, diff_arg=-0.1))
    y0__args__term = (y0, args, term)

    if isinstance(solver, diffrax.SemiImplicitEuler):
        y0 = (y0, y0)
        term = (term, term)

    with pytest.raises(ValueError):
        loss, grads_reversible = _loss(
            y0__args__term,
            solver,
            saveat,
            adjoint=diffrax.ReversibleAdjoint(),
            stepsize_controller=diffrax.ConstantStepSize(),
            pytree_state=False,
        )
