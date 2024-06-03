import contextlib
import math
import operator
from typing import Any, cast

import diffrax
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
import pytest
import scipy.stats
from diffrax import ControlTerm, MultiTerm, ODETerm
from equinox.internal import ω
from jaxtyping import Array, ArrayLike, Float

from .helpers import (
    all_ode_solvers,
    all_split_solvers,
    implicit_tol,
    random_pytree,
    sde_solver_strong_order,
    tree_allclose,
    treedefs,
)


def _all_pairs(*args):
    defaults = [arg["default"] for arg in args]
    yield defaults
    for i in range(len(args)):
        for opt in args[i]["opts"]:
            opts = defaults.copy()
            opts[i] = opt
            yield opts
    for i in range(len(args)):
        for j in range(i + 1, len(args)):
            for opt1 in args[i]["opts"]:
                for opt2 in args[j]["opts"]:
                    opts = defaults.copy()
                    opts[i] = opt1
                    opts[j] = opt2
                    yield opts


@pytest.mark.parametrize(
    "solver",
    (
        diffrax.Euler(),
        diffrax.LeapfrogMidpoint(),
        diffrax.ReversibleHeun(),
        diffrax.Tsit5(),
        diffrax.ImplicitEuler(root_finder=diffrax.VeryChord(rtol=1e-3, atol=1e-6)),
        diffrax.Kvaerno3(root_finder=diffrax.VeryChord(rtol=1e-3, atol=1e-6)),
    ),
)
@pytest.mark.parametrize("t_dtype", (jnp.float32, int, float, jnp.int32))
@pytest.mark.parametrize("y_dtype", (jnp.float32, jnp.complex64))
@pytest.mark.parametrize("treedef", treedefs)
@pytest.mark.parametrize(
    "stepsize_controller",
    (
        diffrax.ConstantStepSize(),
        diffrax.PIDController(rtol=1e-5, atol=1e-8),
        diffrax.PIDController(rtol=1e-5, atol=1e-8, pcoeff=0.3, icoeff=0.3, dcoeff=0.0),
    ),
)
def test_basic(solver, t_dtype, y_dtype, treedef, stepsize_controller, getkey):
    if not isinstance(solver, diffrax.AbstractAdaptiveSolver) and isinstance(
        stepsize_controller, diffrax.PIDController
    ):
        return
    if isinstance(
        solver, diffrax.AbstractImplicitSolver
    ) and treedef == jtu.tree_structure(None):
        return

    if jnp.iscomplexobj(y_dtype) and treedef != jtu.tree_structure(None):
        complex_warn = pytest.warns(match="Complex dtype")

        def f(t, y, args):
            return jtu.tree_map(lambda yi: -1j * yi, y)
    else:
        complex_warn = contextlib.nullcontext()

        def f(t, y, args):
            return jtu.tree_map(operator.neg, y)

    if t_dtype is int:
        t0 = 0
        t1 = 1
        dt0 = 0.01
    elif t_dtype is float:
        t0 = 0.0
        t1 = 1.0
        dt0 = 0.01
    elif t_dtype is jnp.int32:
        t0 = jnp.array(0, dtype=t_dtype)
        t1 = jnp.array(1, dtype=t_dtype)
        dt0 = jnp.array(0.01, dtype=jnp.float32)
    elif t_dtype is jnp.float32:
        t0 = jnp.array(0, dtype=t_dtype)
        t1 = jnp.array(1, dtype=t_dtype)
        dt0 = jnp.array(0.01, dtype=t_dtype)
    else:
        raise ValueError
    y0 = random_pytree(getkey(), treedef, dtype=y_dtype)
    try:
        with complex_warn:
            sol = diffrax.diffeqsolve(
                diffrax.ODETerm(f),
                solver,
                t0,
                t1,
                dt0,
                y0,
                stepsize_controller=stepsize_controller,
            )
    except Exception as e:
        if isinstance(stepsize_controller, diffrax.ConstantStepSize) and str(
            e
        ).startswith("Nonlinear solve diverged"):
            # Implicit method failed to converge. A very normal thing to have happen;
            # usually we'd use adaptive timestepping to handle it.
            pass
        else:
            raise
    else:
        y1 = sol.ys
        # TODO: remove dtype cast, fix Diffrax internals to better respect dtypes.
        if jnp.iscomplexobj(y_dtype):
            true_y1 = jtu.tree_map(
                lambda x, x1: (x * jnp.exp(-1j))[None].astype(x1.dtype), y0, y1
            )
        else:
            true_y1 = jtu.tree_map(
                lambda x, x1: (x * math.exp(-1))[None].astype(x1.dtype), y0, y1
            )
        assert tree_allclose(y1, true_y1, atol=1e-2, rtol=1e-2)


@pytest.mark.parametrize("solver", all_ode_solvers + all_split_solvers)
@pytest.mark.parametrize("dtype", [jnp.float64, jnp.complex128])
def test_ode_order(solver, dtype):
    solver = implicit_tol(solver)
    key = jr.PRNGKey(5678)
    akey, ykey = jr.split(key, 2)

    A = jr.normal(akey, (10, 10), dtype=dtype) * 0.5

    if (
        solver.term_structure
        == diffrax.MultiTerm[tuple[diffrax.AbstractTerm, diffrax.AbstractTerm]]
    ):

        def f1(t, y, args):
            return 0.3 * A @ y

        def f2(t, y, args):
            return 0.7 * A @ y

        term = diffrax.MultiTerm(diffrax.ODETerm(f1), diffrax.ODETerm(f2))
    else:

        def f(t, y, args):
            return A @ y

        term = diffrax.ODETerm(f)
    t0 = 0
    t1 = 4
    y0 = jr.normal(ykey, (10,), dtype=dtype)

    true_yT = jax.scipy.linalg.expm((t1 - t0) * A) @ y0
    exponents = []
    errors = []
    for exponent in [0, -1, -2, -3, -4, -6, -8, -12]:
        dt0 = 2**exponent
        sol = diffrax.diffeqsolve(term, solver, t0, t1, dt0, y0, max_steps=None)
        yT = cast(Array, sol.ys)[-1]
        error = jnp.sum(jnp.abs(yT - true_yT))
        if error < 2**-28:
            break
        exponents.append(exponent)
        errors.append(jnp.log2(error))

    order = scipy.stats.linregress(exponents, errors).slope  # pyright: ignore
    # We accept quite a wide range. Improving this test would be nice.
    assert -0.9 < order - solver.order(term) < 0.9


def _solvers_and_orders():
    # solver, noise, order
    # noise is "any" or "com" or "add" where "com" means commutative and "add" means
    # additive.
    yield diffrax.Euler, "any", 0.5
    yield diffrax.EulerHeun, "any", 0.5
    yield diffrax.Heun, "any", 0.5
    yield diffrax.ItoMilstein, "any", 0.5
    yield diffrax.Midpoint, "any", 0.5
    yield diffrax.ReversibleHeun, "any", 0.5
    yield diffrax.StratonovichMilstein, "any", 0.5
    yield diffrax.SPaRK, "any", 0.5
    yield diffrax.GeneralShARK, "any", 0.5
    yield diffrax.SlowRK, "any", 0.5
    yield diffrax.ReversibleHeun, "com", 1
    yield diffrax.StratonovichMilstein, "com", 1
    yield diffrax.SPaRK, "com", 1
    yield diffrax.GeneralShARK, "com", 1
    yield diffrax.SlowRK, "com", 1.5
    yield diffrax.SPaRK, "add", 1.5
    yield diffrax.GeneralShARK, "add", 1.5
    yield diffrax.ShARK, "add", 1.5
    yield diffrax.SRA1, "add", 1.5
    yield diffrax.SEA, "add", 1.0


def _squareplus(x):
    return 0.5 * (x + jnp.sqrt(x**2 + 4))


def _drift(t, y, args):
    drift_mlp, _, _ = args
    with jax.numpy_dtype_promotion("standard"):
        return 0.5 * drift_mlp(y)


def _diffusion(t, y, args):
    _, diffusion_mlp, noise_dim = args
    with jax.numpy_dtype_promotion("standard"):
        return 0.25 * diffusion_mlp(y).reshape(3, noise_dim)


@pytest.mark.parametrize("solver_ctr,noise,theoretical_order", _solvers_and_orders())
@pytest.mark.parametrize("dtype", [jnp.float64])
def test_sde_strong_order(solver_ctr, noise, theoretical_order, dtype):
    key = jr.PRNGKey(5678)
    driftkey, diffusionkey, ykey, bmkey = jr.split(key, 4)
    num_samples = 20
    bmkeys = jr.split(bmkey, num_samples)

    if noise == "com":
        noise_dim = 1
    elif noise == "any":
        noise_dim = 7
    elif noise == "add":
        return
    else:
        assert False

    drift_mlp = eqx.nn.MLP(
        in_size=3,
        out_size=3,
        width_size=8,
        depth=2,
        activation=_squareplus,
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

    t0 = 0.0
    t1 = 2.0
    y0 = jr.normal(ykey, (3,), dtype=dtype)

    def get_terms(bm):
        return MultiTerm(ODETerm(_drift), ControlTerm(_diffusion, bm))

    # Reference solver is always an ODE-viable solver, so its implementation has been
    # verified by the ODE tests like test_ode_order.
    if issubclass(solver_ctr, diffrax.AbstractItoSolver):
        ref_solver = diffrax.Euler()
    elif issubclass(solver_ctr, diffrax.AbstractStratonovichSolver):
        ref_solver = diffrax.Heun()
    else:
        assert False

    if theoretical_order == 0.5:
        levels = (3, 8)
        ref_level = 10
    elif theoretical_order == 1.0:
        levels = (1, 6)
        ref_level = 12
    elif theoretical_order == 1.5:
        levels = (0, 4)
        ref_level = 12
    else:
        assert False

    def get_dt_and_controller(level):
        return 2**-level, diffrax.ConstantStepSize()

    hs, errors, order = sde_solver_strong_order(
        bmkeys,
        get_terms,
        (noise_dim,),
        t0,
        t1,
        y0,
        args,
        solver_ctr(),
        ref_solver,
        levels,
        ref_level,
        get_dt_and_controller,
        diffrax.SaveAt(t1=True),
        bm_tol=2.0 ** -(ref_level + 2),
    )
    assert -0.2 < order - theoretical_order < 0.2


# Step size deliberately chosen not to divide the time interval
@pytest.mark.parametrize(
    "solver_ctr,dt0",
    ((diffrax.Euler, -0.3), (diffrax.Tsit5, -0.3), (diffrax.Tsit5, None)),
)
@pytest.mark.parametrize(
    "saveat",
    (
        diffrax.SaveAt(t0=True),
        diffrax.SaveAt(t1=True),
        diffrax.SaveAt(ts=[3.5, 0.7]),
        diffrax.SaveAt(steps=True),
        diffrax.SaveAt(dense=True),
    ),
)
@pytest.mark.parametrize("dtype", [jnp.float64, jnp.complex128])
def test_reverse_time(solver_ctr, dt0, saveat, dtype, getkey):
    key = getkey()
    y0 = jr.normal(key, (2, 2), dtype=dtype)
    stepsize_controller = (
        diffrax.PIDController(rtol=1e-3, atol=1e-6)
        if dt0 is None
        else diffrax.ConstantStepSize()
    )

    def f(t, y, args):
        return -y

    t0 = 4
    t1 = 0.3
    sol1 = diffrax.diffeqsolve(
        diffrax.ODETerm(f),
        solver_ctr(),
        t0,
        t1,
        dt0,
        y0,
        stepsize_controller=stepsize_controller,
        saveat=saveat,
    )
    assert tree_allclose(sol1.t0, jnp.array(4.0))
    assert tree_allclose(sol1.t1, jnp.array(0.3))

    def g(t, y, args):
        return y

    t0 = -4
    t1 = -0.3
    negdt0 = None if dt0 is None else -dt0
    if saveat.subs is not None and saveat.subs.ts is not None:
        saveat = diffrax.SaveAt(ts=[-ti for ti in saveat.subs.ts])
    sol2 = diffrax.diffeqsolve(
        diffrax.ODETerm(g),
        solver_ctr(),
        t0,
        t1,
        negdt0,
        y0,
        stepsize_controller=stepsize_controller,
        saveat=saveat,
    )
    assert tree_allclose(sol2.t0, jnp.array(-4.0))
    assert tree_allclose(sol2.t1, jnp.array(-0.3))

    if saveat.subs is not None and (
        saveat.subs.t0
        or saveat.subs.t1
        or saveat.subs.ts is not None
        or saveat.subs.steps
    ):
        assert tree_allclose(sol1.ts, -cast(Array, sol2.ts), equal_nan=True)
        assert tree_allclose(sol1.ys, sol2.ys, equal_nan=True)
    if saveat.dense:
        t = jnp.linspace(0.3, 4, 20)
        for ti in t:
            assert tree_allclose(sol1.evaluate(ti), sol2.evaluate(-ti))
            assert tree_allclose(sol1.derivative(ti), -sol2.derivative(-ti))


def test_semi_implicit_euler():
    term1 = diffrax.ODETerm(lambda t, y, args: -y)
    term2 = diffrax.ODETerm(lambda t, y, args: y)
    y0 = (1.0, -0.5)
    dt0 = 0.00001
    sol1 = diffrax.diffeqsolve(
        (term1, term2),
        diffrax.SemiImplicitEuler(),
        0,
        1,
        dt0,
        y0,
        max_steps=100000,
    )
    term_combined = diffrax.ODETerm(lambda t, y, args: (-y[1], y[0]))
    sol2 = diffrax.diffeqsolve(term_combined, diffrax.Tsit5(), 0, 1, 0.001, y0)
    assert tree_allclose(sol1.ys, sol2.ys)


@pytest.mark.parametrize(
    "solver",
    [
        diffrax.ImplicitEuler(root_finder=diffrax.VeryChord(rtol=1e-3, atol=1e-6)),
        diffrax.Kvaerno5(root_finder=diffrax.VeryChord(rtol=1e-3, atol=1e-6)),
    ],
)
def test_grad_implicit_solve(solver):
    # Check that we work around JAX issue #9374
    # Whilst we're at -- for efficiency -- check the use of PyTree-valued state with
    # implicit solves.

    term = diffrax.ODETerm(lambda t, y, args: (-args * y**ω).ω)

    @jax.jit
    def f(args):
        y0 = (1.0, {"a": 2.0})
        ys = diffrax.diffeqsolve(term, solver, t0=0, t1=1, dt0=0.1, y0=y0, args=args).ys
        ys = cast(Array, ys)
        return jnp.sum(ys[0] + ys[1]["a"])

    grads = jax.jit(jax.grad(f))(1.0)
    assert jnp.isfinite(grads)

    # Test numerical gradients: Diffrax issue #64
    eps = 1e-6
    val = f(1.0)
    val_eps = f(1.0 + eps)
    numerical_grads = (val_eps - val) / eps
    assert tree_allclose(grads, numerical_grads)


def test_concrete_made_jump():
    for constant in (True, False):
        if constant:
            dt0 = 0.1
            stepsize_controller = diffrax.ConstantStepSize()
        else:
            dt0 = None
            stepsize_controller = diffrax.StepTo([0, 0.3, 1])

        @jax.jit
        def run(y0):
            term = diffrax.ODETerm(lambda t, y, args: -y)
            sol = diffrax.diffeqsolve(
                term,
                diffrax.Tsit5(),
                0,
                1,
                dt0,
                y0,
                stepsize_controller=stepsize_controller,
                saveat=diffrax.SaveAt(t1=True, made_jump=True),
                throw=False,
            )
            assert sol.made_jump is False

        run(1)


def test_no_jit():
    # https://github.com/patrick-kidger/diffrax/issues/293
    # https://github.com/patrick-kidger/diffrax/issues/321

    # Test that this doesn't crash.
    with jax.disable_jit():

        def vector_field(t, y, args):
            return jnp.zeros_like(y)

        term = diffrax.ODETerm(vector_field)
        y = jnp.zeros((1,))
        stepsize_controller = diffrax.PIDController(rtol=1e-5, atol=1e-5)
        diffrax.diffeqsolve(
            term,
            diffrax.Kvaerno4(),
            t0=0,
            t1=1e-2,
            dt0=1e-3,
            stepsize_controller=stepsize_controller,
            y0=y,
        )


def test_static(capfd):
    try:
        diffrax._integrate._PRINT_STATIC = True

        def vector_field(t, y, args):
            return jnp.zeros_like(y)

        term = diffrax.ODETerm(vector_field)
        y = jnp.zeros((1,))
        stepsize_controller = diffrax.PIDController(rtol=1e-5, atol=1e-5)
        capfd.readouterr()

        diffrax.diffeqsolve(
            term,
            diffrax.Tsit5(),
            t0=0,
            t1=1e-2,
            dt0=1e-3,
            stepsize_controller=stepsize_controller,
            y0=y,
        )
        text, _ = capfd.readouterr()
        assert (
            text == "static_made_jump=False static_result=diffrax._solution.RESULTS<>\n"
        )

        diffrax.diffeqsolve(
            term,
            diffrax.Kvaerno5(),
            t0=0,
            t1=1e-2,
            dt0=1e-3,
            stepsize_controller=stepsize_controller,
            y0=y,
        )
        text, _ = capfd.readouterr()
        assert text == "static_made_jump=False static_result=None\n"
    finally:
        diffrax._integrate._PRINT_STATIC = False


def test_implicit_tol_error():
    msg = "the tolerances for the implicit solver have not been specified"
    with pytest.raises(ValueError, match=msg):
        diffrax.diffeqsolve(
            diffrax.ODETerm(lambda t, y, args: -y),
            diffrax.Kvaerno5(),
            0,
            1,
            0.01,
            1.0,
        )


def test_term_compatibility():
    class TestControl(eqx.Module):
        dt: Float[ArrayLike, ""]

        def __rmul__(self, other):
            return other.__mul__(self.dt)

        def __mul__(self, other):
            return self.dt * other

    class TestSolver(diffrax.Euler):
        term_structure = diffrax.AbstractTerm[
            tuple[Float[Array, "n 3"]], tuple[TestControl]
        ]

    solver = TestSolver()
    incompatible_vf = lambda t, y, args: jnp.ones((2, 1))
    compatible_vf = lambda t, y, args: (jnp.ones((2, 3)),)
    incompatible_control = lambda t0, t1: t1 - t0
    compatible_control = lambda t0, t1: (TestControl(t1 - t0),)

    incompatible_terms = [
        diffrax.WeaklyDiagonalControlTerm(incompatible_vf, incompatible_control),
        diffrax.WeaklyDiagonalControlTerm(incompatible_vf, compatible_control),
        diffrax.WeaklyDiagonalControlTerm(compatible_vf, incompatible_control),
    ]
    compatible_term = diffrax.WeaklyDiagonalControlTerm(
        compatible_vf, compatible_control
    )
    for term in incompatible_terms:
        with pytest.raises(ValueError, match=r"`terms` must be a PyTree of"):
            diffrax.diffeqsolve(term, solver, 0.0, 1.0, 0.1, (jnp.zeros((2, 1)),))

    diffrax.diffeqsolve(
        compatible_term, solver, 0.1, 1.1, 0.1, (jnp.zeros((2, 3)),), args=["str"]
    )


def test_term_compatibility_pytree():
    class TestSolver(diffrax.AbstractSolver):
        term_structure = {
            "a": diffrax.ODETerm,
            "b": diffrax.ODETerm[Any],
            "c": diffrax.ODETerm[Float[Array, " 3"]],
            "d": diffrax.AbstractTerm[Float[Array, " 4"], Any],
            "e": diffrax.MultiTerm[
                tuple[diffrax.ODETerm, diffrax.AbstractTerm[Any, Float[Array, " 5"]]]
            ],
        }
        interpolation_cls = diffrax.LocalLinearInterpolation

        def init(self, terms, t0, t1, y0, args):
            return None

        def step(self, terms, t0, t1, y0, args, solver_state, made_jump):
            def _step(_term, _y):
                control = _term.contr(t0, t1)
                return _y + _term.vf_prod(t0, _y, args, control)

            _is_term = lambda x: isinstance(x, diffrax.AbstractTerm)
            y1 = jtu.tree_map(_step, terms, y0, is_leaf=_is_term)
            dense_info = dict(y0=y0, y1=y1)
            return y1, None, dense_info, None, diffrax.RESULTS.successful

        def func(self, terms, t0, y0, args):
            assert False

    ode_term = diffrax.ODETerm(lambda t, y, args: -y)
    solver = TestSolver()
    compatible_term = {
        "a": ode_term,
        "b": ode_term,
        "c": ode_term,
        "d": ode_term,
        "e": diffrax.MultiTerm(
            ode_term,
            diffrax.WeaklyDiagonalControlTerm(
                lambda t, y, args: -y, lambda t0, t1: jnp.array(t1 - t0).repeat(5)
            ),
        ),
    }
    compatible_y0 = {
        "a": jnp.array(1.0),
        "b": jnp.array(2.0),
        "c": jnp.arange(3.0),
        "d": jnp.arange(4.0),
        "e": jnp.arange(5.0),
    }
    diffrax.diffeqsolve(compatible_term, solver, 0.0, 1.0, 0.1, compatible_y0)

    incompatible_term1 = {
        "a": ode_term,
        "b": ode_term,
        "c": ode_term,
        "d": ode_term,
        "e": diffrax.MultiTerm(
            ode_term,
            diffrax.WeaklyDiagonalControlTerm(
                lambda t, y, args: -y,
                lambda t0, t1: t1 - t0,  # wrong control shape
            ),
        ),
    }
    incompatible_term2 = {
        "a": ode_term,
        "b": ode_term,
        "c": ode_term,
        # Missing "d" piece
        "e": diffrax.MultiTerm(
            ode_term,
            diffrax.WeaklyDiagonalControlTerm(
                lambda t, y, args: -y, lambda t0, t1: jnp.array(t1 - t0).repeat(3)
            ),
        ),
    }
    incompatible_term3 = {
        "a": ode_term,
        "b": ode_term,
        "c": ode_term,
        "d": ode_term,
        # No MultiTerm for "e"
        "e": diffrax.WeaklyDiagonalControlTerm(
            lambda t, y, args: -y, lambda t0, t1: jnp.array(t1 - t0).repeat(3)
        ),
    }

    incompatible_y0_1 = {
        "a": jnp.array(1.0),
        "b": jnp.array(2.0),
        "c": jnp.arange(4.0),  # of length 4, not 3
        "d": jnp.arange(4.0),
        "e": jnp.arange(5.0),
    }
    incompatible_y0_2 = {
        "a": jnp.array(1.0),
        "b": jnp.array(2.0),
        "c": jnp.arange(3.0),
        # Missing "d" piece
        "e": jnp.arange(5.0),
    }
    incompatible_y0_3 = jnp.array(4.0)  # Completely the wrong structure!
    for term in (
        compatible_term,
        incompatible_term1,
        incompatible_term2,
        incompatible_term3,
    ):
        for y0 in (
            compatible_y0,
            incompatible_y0_1,
            incompatible_y0_2,
            incompatible_y0_3,
        ):
            if term is compatible_term and y0 is compatible_y0:
                continue
            with pytest.raises(ValueError, match=r"`terms` must be a PyTree of"):
                diffrax.diffeqsolve(term, solver, 0.0, 1.0, 0.1, y0)
