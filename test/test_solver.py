from typing import ClassVar

import diffrax
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
import optimistix as optx
import pytest

from .helpers import implicit_tol, tree_allclose


def test_half_solver():
    term = diffrax.ODETerm(lambda t, y, args: -y)
    t0 = 0
    t1 = 1
    y0 = 1.0
    dt0 = None
    solver = diffrax.HalfSolver(diffrax.Euler())
    stepsize_controller = diffrax.PIDController(rtol=1e-3, atol=1e-6)
    diffrax.diffeqsolve(
        term, solver, t0, t1, dt0, y0, stepsize_controller=stepsize_controller
    )


def test_instance_check():
    assert isinstance(diffrax.HalfSolver(diffrax.Euler()), diffrax.Euler)
    assert not isinstance(diffrax.HalfSolver(diffrax.Euler()), diffrax.Heun)


def test_implicit_euler_adaptive():
    term = diffrax.ODETerm(lambda t, y, args: -10 * y**3)
    solver1 = diffrax.ImplicitEuler(root_finder=diffrax.VeryChord(rtol=1e-5, atol=1e-5))
    solver2 = diffrax.ImplicitEuler()
    t0 = 0
    t1 = 1
    dt0 = 1
    y0 = 1.0
    stepsize_controller = diffrax.PIDController(rtol=1e-5, atol=1e-5)
    out1 = diffrax.diffeqsolve(term, solver1, t0, t1, dt0, y0, throw=False)
    out2 = diffrax.diffeqsolve(
        term,
        solver2,
        t0,
        t1,
        dt0,
        y0,
        stepsize_controller=stepsize_controller,
        throw=False,
    )
    assert out1.result == diffrax.RESULTS.nonlinear_divergence
    assert out2.result == diffrax.RESULTS.successful


class _DoubleDopri5(diffrax.AbstractRungeKutta):
    tableau: ClassVar[diffrax.MultiButcherTableau] = diffrax.MultiButcherTableau(
        diffrax.Dopri5.tableau, diffrax.Dopri5.tableau
    )
    calculate_jacobian: ClassVar[
        diffrax.CalculateJacobian
    ] = diffrax.CalculateJacobian.never

    @staticmethod
    def interpolation_cls(**kwargs):
        kwargs.pop("k")
        return diffrax.LocalLinearInterpolation(**kwargs)

    def order(self, terms):
        return 5


@pytest.mark.parametrize("vf_expensive", (False, True))
def test_multiple_tableau_single_step(vf_expensive):
    mlp1 = eqx.nn.MLP(2, 2, 32, 1, key=jr.PRNGKey(0))
    mlp2 = eqx.nn.MLP(2, 2, 32, 1, key=jr.PRNGKey(1))
    term1 = diffrax.ODETerm(lambda t, y, args: mlp1(y))
    term2 = diffrax.ODETerm(lambda t, y, args: mlp2(y))
    terms = diffrax.MultiTerm(term1, term2)
    solver1 = diffrax.Dopri5()
    solver2 = _DoubleDopri5()
    t0 = 0.3
    t1 = 0.7
    y0 = jnp.array([1.0, 2.0])
    if vf_expensive:
        # Huge hack, do this via subclassing AbstractTerm if you're going to do this
        # properly!
        object.__setattr__(terms, "is_vf_expensive", lambda t0, t1, y, args: True)
        solver_state1 = None
        solver_state2 = None
    else:
        solver_state1 = solver1.init(terms, t0, t1, y0, None)
        solver_state2 = solver2.init(terms, t0, t1, y0, None)
    out1 = solver1.step(
        terms, t0, t1, y0, None, solver_state=solver_state1, made_jump=False
    )
    out2 = solver2.step(
        terms, t0, t1, y0, None, solver_state=solver_state2, made_jump=False
    )
    out2[2]["k"] = out2[2]["k"][0] + out2[2]["k"][1]
    assert tree_allclose(out1, out2)


@pytest.mark.parametrize("adaptive", (True, False))
def test_multiple_tableau1(adaptive):
    mlp1 = eqx.nn.MLP(2, 2, 32, 1, key=jr.PRNGKey(0))
    mlp2 = eqx.nn.MLP(2, 2, 32, 1, key=jr.PRNGKey(1))

    term1 = diffrax.ODETerm(lambda t, y, args: mlp1(y))
    term2 = diffrax.ODETerm(lambda t, y, args: mlp2(y))
    t0 = 0
    t1 = 1
    dt0 = 0.1
    y0 = jnp.array([1.0, 2.0])
    if adaptive:
        stepsize_controller = diffrax.PIDController(rtol=1e-3, atol=1e-6)
    else:
        stepsize_controller = diffrax.ConstantStepSize()
    out_a = diffrax.diffeqsolve(
        diffrax.MultiTerm(term1, term2),
        diffrax.Dopri5(),
        t0,
        t1,
        dt0,
        y0,
        stepsize_controller=stepsize_controller,
    )
    out_b = diffrax.diffeqsolve(
        diffrax.MultiTerm(term1, term2),
        _DoubleDopri5(),
        t0,
        t1,
        dt0,
        y0,
        stepsize_controller=stepsize_controller,
    )
    assert jnp.allclose(out_a.ys, out_b.ys, rtol=1e-8, atol=1e-8)  # pyright: ignore

    with pytest.raises(ValueError):
        diffrax.diffeqsolve(
            (term1, term2),
            _DoubleDopri5(),  # pyright: ignore
            t0,
            t1,
            dt0,
            y0,
            stepsize_controller=stepsize_controller,
        )


def test_multiple_tableau2():
    # Different number of stages
    with pytest.raises(ValueError):

        class X(diffrax.AbstractRungeKutta):
            tableau = diffrax.MultiButcherTableau(
                diffrax.Dopri5.tableau, diffrax.Bosh3.tableau
            )
            calculate_jacobian = diffrax.CalculateJacobian.never

            def interpolation_cls(self, *, k, **kwargs):
                return diffrax.LocalLinearInterpolation(**kwargs)

    # Multiple implicit
    with pytest.raises(ValueError):

        class Y(diffrax.AbstractRungeKutta):
            tableau = diffrax.MultiButcherTableau(
                diffrax.Kvaerno3.tableau, diffrax.Kvaerno3.tableau
            )
            calculate_jacobian = diffrax.CalculateJacobian.never

            def interpolation_cls(self, *, k, **kwargs):
                return diffrax.LocalLinearInterpolation(**kwargs)

    class Z(diffrax.AbstractRungeKutta):
        tableau = diffrax.MultiButcherTableau(
            diffrax.Bosh3.tableau, diffrax.Kvaerno3.tableau
        )
        calculate_jacobian = diffrax.CalculateJacobian.never

        def interpolation_cls(self, *, k, **kwargs):
            return diffrax.LocalLinearInterpolation(**kwargs)


@pytest.mark.parametrize("implicit", (True, False))
@pytest.mark.parametrize("vf_expensive", (True, False))
@pytest.mark.parametrize("adaptive", (True, False))
def test_everything_pytree(implicit, vf_expensive, adaptive):
    class Term(diffrax.AbstractTerm):
        coeff: float

        def vf(self, t, y, args):
            return {"f": -self.coeff * y["y"]}

        def contr(self, t0, t1):
            return {"t": t1 - t0}

        def prod(self, vf, control):
            return {"y": vf["f"] * control["t"]}

        def is_vf_expensive(self, t0, t1, y, args):
            return vf_expensive

    term = diffrax.MultiTerm(Term(0.3), Term(0.7))

    if implicit:
        tableau_ = diffrax.Kvaerno5.tableau
        calculate_jacobian_ = diffrax.CalculateJacobian.second_stage
    else:
        tableau_ = diffrax.Dopri5.tableau
        calculate_jacobian_ = diffrax.CalculateJacobian.never

    class DoubleSolver(diffrax.AbstractRungeKutta):
        tableau = diffrax.MultiButcherTableau(diffrax.Dopri5.tableau, tableau_)
        calculate_jacobian = calculate_jacobian_
        if implicit:
            root_finder: optx.AbstractRootFinder = diffrax.VeryChord(
                rtol=1e-3, atol=1e-3
            )
            root_find_max_steps: int = 10

        @staticmethod
        def interpolation_cls(*, t0, t1, y0, y1, k):
            k_left, k_right = k
            k = {"y": k_left["y"] + k_right["y"]}
            return diffrax._solver.dopri5._Dopri5Interpolation(
                t0=t0,
                t1=t1,
                y0=y0,  # pyright: ignore
                y1=y1,  # pyright: ignore
                k=k,  # pyright: ignore
            )

        def order(self, terms):
            return 5

    solver = DoubleSolver()
    t0 = 0.4
    t1 = 0.9
    dt0 = 0.0007
    y0 = {"y": jnp.array([[1.0, 2.0], [3.0, 4.0]])}
    saveat = diffrax.SaveAt(ts=jnp.linspace(t0, t1, 23))
    if adaptive:
        stepsize_controller = diffrax.PIDController(rtol=1e-10, atol=1e-10)
    else:
        stepsize_controller = diffrax.ConstantStepSize()
    sol = diffrax.diffeqsolve(
        term,
        solver,
        t0,
        t1,
        dt0,
        y0,
        saveat=saveat,
        stepsize_controller=stepsize_controller,
    )
    true_sol = diffrax.diffeqsolve(
        diffrax.ODETerm(lambda t, y, args: {"y": -y["y"]}),
        diffrax.Dopri5(),
        t0,
        t1,
        dt0,
        y0,
        saveat=saveat,
        stepsize_controller=stepsize_controller,
    )
    if implicit:
        tol = 1e-4  # same ODE but different solver
    else:
        tol = 1e-8  # should be exact same numerics, up to floating point weirdness
    assert tree_allclose(sol.ys, true_sol.ys, rtol=tol, atol=tol)


# Essentially used as a check that our general IMEX implementation is correct.
def test_sil3():
    class ReferenceSil3(diffrax.AbstractImplicitSolver):
        term_structure = diffrax.MultiTerm[
            tuple[diffrax.AbstractTerm, diffrax.AbstractTerm]
        ]
        interpolation_cls = diffrax.LocalLinearInterpolation

        root_finder: optx.AbstractRootFinder
        root_find_max_steps: int = 10

        def order(self, terms):
            return 2

        def init(self, terms, t0, t1, y0, args):
            return None

        def func(self, terms, t0, y0, args):
            assert False

        def step(self, terms, t0, t1, y0, args, solver_state, made_jump):
            del solver_state, made_jump
            explicit, implicit = terms.terms
            dt = t1 - t0
            ex_vf_prod = lambda t, y: explicit.vf(t, y, args) * dt
            im_vf_prod = lambda t, y: implicit.vf(t, y, args) * dt
            fs = []
            gs = []

            # first stage is explicit
            fs.append(ex_vf_prod(t0, y0))
            gs.append(im_vf_prod(t0, y0))

            def _second_stage(ya, _):
                [f0] = fs
                [g0] = gs
                g1 = im_vf_prod(ta, ya)
                return ya - (y0 + (1 / 3) * f0 + (1 / 6) * g0 + (1 / 6) * g1)

            ta = t0 + (1 / 3) * dt
            ya = optx.root_find(_second_stage, self.root_finder, y0).value
            fs.append(ex_vf_prod(ta, ya))
            gs.append(im_vf_prod(ta, ya))

            def _third_stage(yb, _):
                [f0, f1] = fs
                [g0, g1] = gs
                g2 = im_vf_prod(tb, yb)
                return yb - (
                    y0 + (1 / 6) * f0 + (1 / 2) * f1 + (1 / 3) * g0 + (1 / 3) * g2
                )

            tb = t0 + (2 / 3) * dt
            yb = optx.root_find(_third_stage, self.root_finder, ya).value
            fs.append(ex_vf_prod(tb, yb))
            gs.append(im_vf_prod(tb, yb))

            def _fourth_stage(yc, _):
                [f0, f1, f2] = fs
                [g0, g1, g2] = gs
                g3 = im_vf_prod(tc, yc)
                return yc - (
                    y0
                    + (1 / 2) * f0
                    + (-1 / 2) * f1
                    + f2
                    + (3 / 8) * g0
                    + (3 / 8) * g2
                    + (1 / 4) * g3
                )

            tc = t1
            yc = optx.root_find(_fourth_stage, self.root_finder, yb).value
            fs.append(ex_vf_prod(tc, yc))
            gs.append(im_vf_prod(tc, yc))

            [f0, f1, f2, f3] = fs
            [g0, g1, g2, g3] = gs
            y1 = (
                y0
                + (1 / 2) * f0
                - (1 / 2) * f1
                + f2
                + (3 / 8) * g0
                + (3 / 8) * g2
                + (1 / 4) * g3
            )

            # Use Heun as the embedded method.
            y_error = y0 + 0.5 * (f0 + g0 + f3 + g3) - y1
            ks = (jnp.stack(fs), jnp.stack(gs))
            dense_info = dict(y0=y0, y1=y1, k=ks)
            state = (False, (f3 / dt, g3 / dt))
            result = jtu.tree_map(jnp.asarray, diffrax.RESULTS.successful)
            return y1, y_error, dense_info, state, result

    reference_solver = ReferenceSil3(root_finder=optx.Newton(rtol=1e-8, atol=1e-8))
    solver = diffrax.Sil3(root_finder=diffrax.VeryChord(rtol=1e-8, atol=1e-8))

    key = jr.PRNGKey(5678)
    mlpkey1, mlpkey2, ykey = jr.split(key, 3)

    mlp1 = eqx.nn.MLP(3, 2, 8, 1, key=mlpkey1)
    mlp2 = eqx.nn.MLP(3, 2, 8, 1, key=mlpkey2)

    def f1(t, y, args):
        y = jnp.concatenate([t[None], y])
        return mlp1(y)

    def f2(t, y, args):
        y = jnp.concatenate([t[None], y])
        return mlp2(y)

    terms = diffrax.MultiTerm(diffrax.ODETerm(f1), diffrax.ODETerm(f2))
    t0 = jnp.array(0.3)
    t1 = jnp.array(1.5)
    y0 = jr.normal(ykey, (2,), dtype=jnp.float64)
    args = None

    state = solver.init(terms, t0, t1, y0, args)
    out = solver.step(terms, t0, t1, y0, args, solver_state=state, made_jump=False)
    reference_out = reference_solver.step(
        terms, t0, t1, y0, args, solver_state=None, made_jump=False
    )
    assert tree_allclose(out, reference_out)


# Honestly not sure how meaningful this test is -- Rober isn't *that* stiff.
# In fact, even Heun will get the correct answer with the tolerances we specify!
@pytest.mark.parametrize(
    "solver",
    (
        diffrax.Kvaerno3(),
        diffrax.Kvaerno4(),
        diffrax.Kvaerno5(),
        diffrax.KenCarp3(),
        diffrax.KenCarp4(),
        diffrax.KenCarp5(),
    ),
)
def test_rober(solver):
    def rober(t, y, args):
        y0, y1, y2 = y
        k1 = 0.04
        k2 = 3e7
        k3 = 1e4
        f0 = -k1 * y0 + k3 * y1 * y2
        f1 = k1 * y0 - k2 * y1**2 - k3 * y1 * y2
        f2 = k2 * y1**2
        return jnp.stack([f0, f1, f2])

    term = diffrax.ODETerm(rober)
    if solver.__class__.__name__.startswith("KenCarp"):
        term = diffrax.MultiTerm(diffrax.ODETerm(lambda t, y, args: 0), term)
    t0 = 0
    t1 = 100
    y0 = jnp.array([1.0, 0, 0])
    dt0 = 0.0002
    saveat = diffrax.SaveAt(ts=jnp.array([0.0, 1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2]))
    stepsize_controller = diffrax.PIDController(rtol=1e-10, atol=1e-10)
    sol = diffrax.diffeqsolve(
        term,
        solver,
        t0,
        t1,
        dt0,
        y0,
        saveat=saveat,
        stepsize_controller=stepsize_controller,
        max_steps=None,
    )
    # Obtained using Kvaerno5 with rtol,atol=1e-20
    true_ys = jnp.array(
        [
            [1.0000000000000000e00, 0.0000000000000000e00, 0.0000000000000000e00],
            [9.9999600000801137e-01, 3.9840684637775332e-06, 1.5923523513217297e-08],
            [9.9996000156321818e-01, 2.9169034944881154e-05, 1.0829401837965007e-05],
            [9.9960068268829505e-01, 3.6450478878442643e-05, 3.6286683282835678e-04],
            [9.9607774744245892e-01, 3.5804372350422432e-05, 3.8864481851928275e-03],
            [9.6645973733301294e-01, 3.0746265785786866e-05, 3.3509516401211095e-02],
            [8.4136992384147014e-01, 1.6233909379904643e-05, 1.5861384224914774e-01],
            [6.1723488239606716e-01, 6.1535912746388841e-06, 3.8275896401264059e-01],
        ]
    )
    assert jnp.allclose(sol.ys, true_ys, rtol=1e-3, atol=1e-8)  # pyright: ignore


def test_implicit_closure_convert():
    @jax.grad
    def f(x):
        def vector_field(t, y, args):
            return x * y

        term = diffrax.ODETerm(vector_field)
        solver = diffrax.Kvaerno3()
        solver = implicit_tol(solver)
        out = diffrax.diffeqsolve(term, solver, 0, 1, 0.1, 1.0)
        return out.ys[0]  # pyright: ignore

    f(1.0)


# Doesn't crash
def test_adaptive_dt0_semiimplicit_euler():
    f = diffrax.ODETerm(lambda t, y, args: y)
    g = diffrax.ODETerm(lambda t, y, args: y)
    solver = diffrax.HalfSolver(diffrax.SemiImplicitEuler())
    y0 = (1.0, 1.0)
    stepsize_controller = diffrax.PIDController(rtol=1e-5, atol=1e-5)
    diffrax.diffeqsolve(
        (f, g), solver, 0, 1, None, y0, stepsize_controller=stepsize_controller
    )


# Doesn't crash
def test_adaptive_dt0_milstein(getkey):
    bm = diffrax.VirtualBrownianTree(0, 1, 1e-3, (), key=getkey())
    f = diffrax.ODETerm(lambda t, y, args: y)
    g = diffrax.ControlTerm(lambda t, y, args: y, bm)
    terms = diffrax.MultiTerm(f, g)
    solver = diffrax.HalfSolver(diffrax.ItoMilstein())
    stepsize_controller = diffrax.PIDController(rtol=1e-5, atol=1e-5)
    diffrax.diffeqsolve(
        terms, solver, 0, 1, None, 1, stepsize_controller=stepsize_controller
    )
