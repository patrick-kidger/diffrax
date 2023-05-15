from dataclasses import dataclass, field
from typing import Literal, Optional, Union

import equinox as eqx
import equinox.internal as eqxi
import jax
import jax.lax as lax
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np
from equinox.internal import ω
from jaxtyping import Array, Bool, PyTree, Scalar

from ..custom_types import DenseInfo
from ..solution import is_okay, RESULTS, update_result
from ..term import AbstractTerm, ODETerm, WrapTerm
from .base import AbstractAdaptiveSolver, AbstractImplicitSolver, vector_tree_dot


# Not a pytree node!
@dataclass(frozen=True)
class ButcherTableau:
    """The Butcher tableau for an explicit or diagonal Runge--Kutta method."""

    # Explicit RK methods
    c: np.ndarray
    b_sol: np.ndarray
    b_error: np.ndarray
    a_lower: tuple[np.ndarray, ...]

    # Implicit RK methods
    a_diagonal: Optional[np.ndarray] = None
    a_predictor: Optional[tuple[np.ndarray, ...]] = None

    # Properties implied by the above tableaus, e.g. used to define fast-paths.
    ssal: bool = field(init=False)
    fsal: bool = field(init=False)
    implicit: bool = field(init=False)
    num_stages: int = field(init=False)

    def __post_init__(self):
        assert self.c.ndim == 1
        for a_i in self.a_lower:
            assert a_i.ndim == 1
        assert self.b_sol.ndim == 1
        assert self.b_error.ndim == 1
        assert self.c.shape[0] == len(self.a_lower)
        assert all(i + 1 == a_i.shape[0] for i, a_i in enumerate(self.a_lower))
        assert self.c.shape[0] + 1 == self.b_sol.shape[0]
        assert self.c.shape[0] + 1 == self.b_error.shape[0]
        for i, (a_i, c_i) in enumerate(zip(self.a_lower, self.c)):
            diagonal = 0 if self.a_diagonal is None else self.a_diagonal[i + 1]
            assert np.allclose(sum(a_i) + diagonal, c_i)
        assert np.allclose(sum(self.b_sol), 1.0)
        assert np.allclose(sum(self.b_error), 0.0)

        if self.a_diagonal is None:
            assert self.a_predictor is None
        else:
            assert self.a_predictor is not None
            assert self.a_diagonal.ndim == 1
            assert self.c.shape[0] + 1 == self.a_diagonal.shape[0]
            assert len(self.a_lower) == len(self.a_predictor)
            for a_lower_i, a_predictor_i in zip(self.a_lower, self.a_predictor):
                assert a_lower_i.shape == a_predictor_i.shape
                assert np.allclose(sum(a_predictor_i), 1.0)

        lower_b_sol_equal = (self.b_sol[:-1] == self.a_lower[-1]).all()
        last_diagonal = 0 if self.a_diagonal is None else self.a_diagonal[-1]
        diagonal_b_sol_equal = self.b_sol[-1] == last_diagonal
        explicit_first_stage = self.a_diagonal is None or (self.a_diagonal[0] == 0)
        explicit_last_stage = self.a_diagonal is None or (self.a_diagonal[-1] == 0)
        # Solution y1 is the same as the last stage
        object.__setattr__(
            self,
            "ssal",
            lower_b_sol_equal and diagonal_b_sol_equal and explicit_last_stage,
        )
        # Vector field - control product k1 is the same across first/last stages.
        object.__setattr__(
            self,
            "fsal",
            lower_b_sol_equal and diagonal_b_sol_equal and explicit_first_stage,
        )
        object.__setattr__(self, "implicit", self.a_diagonal is not None)
        object.__setattr__(self, "num_stages", len(self.b_sol))


ButcherTableau.__init__.__doc__ = """**Arguments:**

Let `k` denote the number of stages of the solver.

- `a_lower`: the lower triangle (without the diagonal) of the Butcher tableau. Should
    be a tuple of NumPy arrays, corresponding to the rows of this lower triangle. The
    first array represents the should be of shape `(1,)`. Each subsequent array should
    be of shape `(2,)`, `(3,)` etc. The final array should have shape `(k - 1,)`.
- `b_sol`: the linear combination of stages to take to produce the output at each step.
    Should be a NumPy array of shape `(k,)`.
- `b_error`: the linear combination of stages to take to produce the error estimate at
    each step. Should be a NumPy array of shape `(k,)`. Note that this is *not*
    differenced against `b_sol` prior to evaluation. (i.e. `b_error` gives the linear
    combination for producing the error estimate directly, not for producing some
    alternate solution that is compared against the main solution).
- `c`: the time increments used in the Butcher tableau.
- `a_diagonal`: optional. The diagonal of the Butcher tableau. Should be `None` or a
    NumPy array of shape `(k,)`. Used for diagonal implicit Runge--Kutta methods only.
- `a_predictor`: optional. Used in a similar way to `a_lower`; specifies the linear
    combination of previous stages to use as a predictor for the solution to the
    implicit problem at that stage. See
    [the developer documentation](../../devdocs/predictor_dirk). Used for diagonal
    implicit Runge--Kutta methods only.

Whether the solver exhibits either the FSAL or SSAL properties is determined
automatically.
"""


class CalculateJacobian(metaclass=eqxi.ContainerMeta):
    """An enumeration of possible ways a Runga--Kutta method may wish to calculate a
    Jacobian.

    `never`: used for explicit Runga--Kutta methods.

    `every_step`: the Jacobian is calculated once per step; in particular it is
        calculated at the start of the step and re-used for every stage in the step.
        Used for SDIRK and ESDIRK methods.

    `every_stage`: the Jacobian is calculated once per stage. Used for DIRK methods.
    """

    never = "never"
    every_step = "every_step"
    every_stage = "every_stage"


_SolverState = Optional[tuple[Bool[Scalar, ""], PyTree[Array]]]


# TODO: examine termination criterion for Newton iteration
# TODO: replace fi with fi=(zi + predictor), where this relation defines some zi, and
#       iterate to find zi, using zi=0 as the predictor. This should give better
#       numerical behaviour since the iteration is close to 0. (Although we have
#       multiplied by the increment of the control, i.e. dt, which is small...)
def _implicit_relation_f(fi, nonlinear_solve_args):
    diagonal, vf, prod, ti, yi_partial, args, control = nonlinear_solve_args
    diff = (
        vf(ti, (yi_partial**ω + diagonal * prod(fi, control) ** ω).ω, args) ** ω
        - fi**ω
    ).ω
    return diff


# TODO: consider dividing by diagonal and control
def _implicit_relation_k(ki, nonlinear_solve_args):
    # c.f:
    # https://github.com/SciML/DiffEqDevMaterials/blob/master/newton/output/main.pdf
    # (Bearing in mind that our ki is dt times smaller than theirs.)
    diagonal, vf_prod, ti, yi_partial, args, control = nonlinear_solve_args
    diff = (
        vf_prod(ti, (yi_partial**ω + diagonal * ki**ω).ω, args, control) ** ω
        - ki**ω
    ).ω
    return diff


_unused = eqxi.str2jax("unused")  # Sentinel that can be passed into `while_loop` etc.


def _is_term(x):
    return isinstance(x, AbstractTerm)


# Not a pytree
class _Leaf:
    def __init__(self, value):
        self.value = value


def _sum(*x):
    assert len(x) > 0
    # Not sure if the builtin does the right thing with JAX tracers?
    total = x[0]
    for xi in x[1:]:
        total = total + xi
    return total


class AbstractRungeKutta(AbstractAdaptiveSolver):
    """Abstract base class for all Runge--Kutta solvers. (Other than fully-implicit
    Runge--Kutta methods, which have a different computational structure.)

    Whilst this class can be subclassed directly, when defining your own Runge--Kutta
    methods, it is usally better to work with [`diffrax.AbstractERK`][],
    [`diffrax.AbstractDIRK`][], [`diffrax.AbstractSDIRK`][],
    [`diffrax.AbstractESDIRK`][] directly.

    Subclasses should specify two class-level attributes. The first is `tableau`, an
    instance of [`diffrax.ButcherTableau`][]. The second is `calculate_jacobian`, an
    instance of [`diffrax.CalculateJacobian`][].
    """

    scan_kind: Union[None, Literal["lax"], Literal["checkpointed"]] = None

    tableau: eqxi.AbstractClassVar[PyTree[ButcherTableau]]
    calculate_jacobian: eqxi.AbstractClassVar[CalculateJacobian]

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        seen_implicit = False
        num_stages = None

        def _f(t: ButcherTableau):
            nonlocal seen_implicit
            nonlocal num_stages
            if num_stages is None:
                num_stages = t.num_stages
            if t.num_stages != num_stages:
                raise ValueError("Tableaus must all have the same number of stages")
            if t.implicit:
                if seen_implicit:
                    raise ValueError("May have at most one implicit tableau")
                else:
                    seen_implicit = True
            return AbstractTerm

        if hasattr(cls, "tableau"):  # Abstract subclasses may not have a tableau
            term_structure = jtu.tree_map(_f, cls.tableau)
            # Allow subclasses to specify more specific term structures if desired, e.g.
            # (ODETerm, ControlTerm) rather than (AbstractTerm, AbtstractTerm).
            try:
                term_structure2 = cls.term_structure
            except AttributeError:
                cls.term_structure = term_structure
            else:
                x = jtu.tree_structure(term_structure, is_leaf=_is_term)
                x2 = jtu.tree_structure(term_structure2, is_leaf=_is_term)
                if x != x2:
                    raise ValueError("Mismatched term structures")

    def _common(self, terms, t0, t1, y0, args):
        # For simplicity we share `vf_expensive` and `fsal` across all tableaus.
        # TODO: could we make these work per-tableau?
        vf_expensive = False
        fsal = True
        terms = jtu.tree_leaves(terms, is_leaf=_is_term)
        tableaus = jtu.tree_leaves(self.tableau)
        assert len(terms) == len(tableaus)
        for term, tableau in zip(terms, tableaus):
            vf_expensive = vf_expensive or term.is_vf_expensive(t0, t1, y0, args)
            fsal = fsal and tableau.fsal
        # If the vector field is expensive then we want to use vf_prods instead.
        # FSAL implies evaluating just the vector field, since we need to contract
        # the same vector field evaluation against two different controls.
        fsal = fsal and not vf_expensive
        return vf_expensive, fsal

    def func(
        self,
        terms: PyTree[AbstractTerm],
        t0: Scalar,
        y0: PyTree,
        args: PyTree,
    ) -> PyTree:
        return jtu.tree_map(lambda t: t.vf(t0, y0, args), terms, is_leaf=_is_term)

    def init(
        self,
        terms: PyTree[AbstractTerm],
        t0: Scalar,
        t1: Scalar,
        y0: PyTree,
        args: PyTree,
    ) -> _SolverState:
        _, fsal = self._common(terms, t0, t1, y0, args)
        if fsal:
            first_step = jnp.array(True)
            if (type(terms) is WrapTerm) and (type(terms.term) is ODETerm):
                # Privileged optimisation for the common case
                f0 = jtu.tree_map(lambda x: jnp.zeros(x.shape, x.dtype), y0)
            else:
                # Must be initialiased at zero as it is inserted into `ks` which must be
                # initialised at zero.
                f0 = eqxi.eval_zero(lambda: self.func(terms, t0, y0, args))
            return first_step, f0
        else:
            return None

    def step(
        self,
        terms: PyTree[AbstractTerm],
        t0: Scalar,
        t1: Scalar,
        y0: PyTree,
        args: PyTree,
        solver_state: _SolverState,
        made_jump: Bool,
    ) -> tuple[PyTree, PyTree, DenseInfo, _SolverState, RESULTS]:
        #
        # Some Runge--Kutta methods have special structure that we can use to improve
        # efficiency.
        #
        # The famous one is FSAL; "first same as last". That is, the final evaluation
        # of the vector field on the previous step is the same as the first evaluation
        # on the subsequent step. We can reuse it and save an evaluation.
        # However note that this requires saving a vf evaluation, not a
        # vf-control-product. (This comes up when we have a different control on the
        # next step, e.g. as with adaptive step sizes, or with SDEs.)
        # As such we disable FSAL if a vf is expensive and a vf-control-product is
        # cheap. (The canonical example is the optimise-then-discretise adjoint SDE.
        # For this SDE, the vf-control product is a vector-Jacobian product, which is
        # notably cheaper than evaluating a full Jacobian.)
        #
        # Next we have SSAL; "solution same as last". That is, the output of the step
        # has already been calculated during the internal stage calculations. We can
        # reuse those and save a dot product.
        #
        # Finally we have a choice whether to save and work with vector field
        # evaluations (fs), or to save and work with (vector field)-control products
        # (ks).
        # The former is needed for implicit FSAL solvers: they need to obtain the
        # final f1 for the FSAL property, which means they need to do the implicit
        # solve in vf-space rather than (vf-control-product)-space, which means they
        # need to use `fs` to predict the initial point for the root finding operation.
        # Meanwhile the latter is needed when solving optimise-then-discretise adjoint
        # SDEs, for which vector field evaluations are prohibitively expensive, and we
        # must necessarily work only with the (much cheaper) vf-control-products. (In
        # this case this is the difference between computing a Jacobian and computing a
        # vector-Jacobian product.)
        # For other problems, we choose to use `ks`. This doesn't have a strong
        # rationale although it does have some minor efficiency points in its favour,
        # e.g. we need `ks` to perform dense interpolation if needed.
        #

        assert jtu.tree_structure(terms, is_leaf=_is_term) == jtu.tree_structure(
            self.tableau
        )

        # Structure of `terms` and `self.tableau`.
        def t_map(fn, *trees):
            def _fn(_, *_trees):
                return fn(*_trees)

            return jtu.tree_map(_fn, self.tableau, *trees)

        def t_leaves(tree):
            return [x.value for x in jtu.tree_leaves(t_map(_Leaf, tree))]

        # Structure of `y` and `k`.
        # (but not `f`, which can be arbitrary and different)
        def s_map(fn, *trees):
            def _fn(_, *_trees):
                return fn(*_trees)

            return jtu.tree_map(_fn, y0, *trees)

        def ts_map(fn, *trees):
            return t_map(lambda *_trees: s_map(fn, *_trees), *trees)

        control = t_map(lambda term_i: term_i.contr(t0, t1), terms)
        dt = t1 - t0

        def vf(t, y):
            _vf = lambda term_i, t_i: term_i.vf(t_i, y, args)
            return t_map(_vf, terms, t)

        def vf_prod(t, y):
            _vf = lambda term_i, t_i, control_i: term_i.vf_prod(t_i, y, args, control_i)
            return t_map(_vf, terms, t, control)

        def prod(f):
            _prod = lambda term_i, f_i, control_i: term_i.prod(f_i, control_i)
            return t_map(_prod, terms, f, control)

        num_stages = jtu.tree_leaves(self.tableau)[0].num_stages
        is_vf_expensive, fsal = self._common(terms, t0, t1, y0, args)
        if fsal:
            assert solver_state is not None
            first_step, f0 = solver_state
            stage_index = jnp.where(first_step, 0, 1)
            # `made_jump` can be a tracer, hence the `is`.
            if made_jump is False:
                # Fast-path for compilation in the common case.
                k0 = prod(f0)
            else:
                _t0 = t_map(lambda _: t0)
                k0 = lax.cond(made_jump, lambda: vf_prod(_t0, y0), lambda: prod(f0))
                del _t0
        else:
            f0 = _unused
            k0 = _unused
            stage_index = 0
        del solver_state

        # Must be initialised at zero as we do matmuls against the partially-filled
        # array.
        ks = t_map(
            lambda: s_map(lambda x: jnp.zeros((num_stages,) + x.shape, x.dtype), y0),
        )
        if fsal:
            ks = ts_map(lambda x, xs: xs.at[0].set(x), k0, ks)

        def embed_a_lower(tableau):
            tableau_a_lower = np.zeros((num_stages, num_stages))
            for i, a_lower_i in enumerate(tableau.a_lower):
                tableau_a_lower[i + 1, : i + 1] = a_lower_i
            return jnp.asarray(tableau_a_lower)

        def embed_c(tableau):
            tableau_c = np.zeros(num_stages)
            tableau_c[1:] = tableau.c
            return jnp.asarray(tableau_c)

        tableau_a_lower = t_map(embed_a_lower, self.tableau)
        tableau_c = t_map(embed_c, self.tableau)

        def cond_fun(val):
            _stage_index, *_ = val
            return _stage_index < num_stages

        def body_fun(val):
            stage_index, _, _, _, ks = val
            a_lower_i = t_map(lambda t: t[stage_index], tableau_a_lower)
            c_i = t_map(lambda t: t[stage_index], tableau_c)
            # Unwrap buffers. This is only valid (=correct under autodiff) because we
            # follow a triangular pattern and don't read from a location before it's
            # written to, or write to the same location twice.
            # (The reads in the matmuls don't count, as we initialise at zero.)
            unsafe_ks = ts_map(lambda x: x[...], ks)
            increment = t_map(vector_tree_dot, a_lower_i, unsafe_ks)
            yi_partial = s_map(_sum, y0, *t_leaves(increment))
            # No floating point error
            ti = t_map(lambda _c_i: jnp.where(_c_i == 1, t1, t0 + _c_i * dt), c_i)
            if fsal:
                assert not is_vf_expensive
                fi = vf(ti, yi_partial)
                ki = prod(fi)
            else:
                fi = _unused
                ki = vf_prod(ti, yi_partial)
            ks = ts_map(lambda x, xs: xs.at[stage_index].set(x), ki, ks)
            return stage_index + 1, yi_partial, increment, fi, ks

        def buffers(val):
            _, _, _, _, ks = val
            return ks

        init_val = (stage_index, y0, t_map(lambda: y0), f0, ks)
        final_val = eqxi.while_loop(
            cond_fun,
            body_fun,
            init_val,
            max_steps=num_stages,
            buffers=buffers,
            kind="checkpointed" if self.scan_kind is None else self.scan_kind,
            checkpoints=num_stages,
        )
        _, y1_partial, increment, f1, ks = final_val

        if all(tableau.ssal for tableau in jtu.tree_leaves(self.tableau)):
            y1 = y1_partial
        else:
            increment = t_map(
                lambda t, k, i: i if t.ssal else vector_tree_dot(t.b_sol, k),
                self.tableau,
                ks,
                increment,
            )
            y1 = s_map(_sum, y0, *t_leaves(increment))
        y_error = t_map(lambda t, k: vector_tree_dot(t.b_error, k), self.tableau, ks)
        dense_info = dict(y0=y0, y1=y1, k=ks)
        if fsal:
            new_solver_state = False, f1
        else:
            new_solver_state = None
        result = RESULTS.successful
        return y1, y_error, dense_info, new_solver_state, result

    def old_step(
        self,
        terms: AbstractTerm,
        t0: Scalar,
        t1: Scalar,
        y0: PyTree,
        args: PyTree,
        solver_state: _SolverState,
        made_jump: Bool,
    ) -> tuple[PyTree, PyTree, DenseInfo, _SolverState, RESULTS]:
        #
        # Some Runge--Kutta methods have special structure that we can use to improve
        # efficiency.
        #
        # The famous one is FSAL; "first same as last". That is, the final evaluation
        # of the vector field on the previous step is the same as the first evaluation
        # on the subsequent step. We can reuse it and save an evaluation.
        # However note that this requires saving a vf evaluation, not a
        # vf-control-product. (This comes up when we have a different control on the
        # next step, e.g. as with adaptive step sizes, or with SDEs.)
        # As such we disable FSAL if a vf is expensive and a vf-control-product is
        # cheap. (The canonical example is the optimise-then-discretise adjoint SDE.
        # For this SDE, the vf-control product is a vector-Jacobian product, which is
        # notably cheaper than evaluating a full Jacobian.)
        #
        # Next we have SSAL; "solution same as last". That is, the output of the step
        # has already been calculated during the internal stage calculations. We can
        # reuse those and save a dot product.
        #
        # Finally we have a choice whether to save and work with vector field
        # evaluations (fs), or to save and work with (vector field)-control products
        # (ks).
        # The former is needed for implicit FSAL solvers: they need to obtain the
        # final f1 for the FSAL property, which means they need to do the implicit
        # solve in vf-space rather than (vf-control-product)-space, which means they
        # need to use `fs` to predict the initial point for the root finding operation.
        # Meanwhile the latter is needed when solving optimise-then-discretise adjoint
        # SDEs, for which vector field evaluations are prohibitively expensive, and we
        # must necessarily work only with the (much cheaper) vf-control-products. (In
        # this case this is the difference between computing a Jacobian and computing a
        # vector-Jacobian product.)
        # For other problems, we choose to use `ks`. This doesn't have a strong
        # rationale although it does have some minor efficiency points in its favour,
        # e.g. we need `ks` to perform dense interpolation if needed.
        #

        implicit_first_stage = self.tableau.implicit and self.tableau.a_diagonal[0] != 0
        # If we're computing the Jacobian at the start of the step, then we
        # need this as a linearisation point.
        #
        # If the first stage is implicit, then we need this as a predictor for
        # where to start iterating from.
        need_f0_or_k0 = (
            self.calculate_jacobian == CalculateJacobian.every_step
            or implicit_first_stage
        )
        vf_expensive, fsal = self._common(terms, t0, t1, y0, args)
        if self.tableau.implicit and fsal:
            use_fs = True
        elif vf_expensive:
            use_fs = False
        else:  # Choice not as important here; we use ks for minor efficiency reasons.
            use_fs = False
        del vf_expensive

        control = terms.contr(t0, t1)
        dt = t1 - t0

        #
        # Calculate `f0` and `k0`. If this is just a first explicit stage then we'll
        # sort that out later. But we might need these values for something else too
        # (as a predictor for implicit stages; as a linearisation point for a Jacobian).
        #

        f0 = None
        k0 = None
        if fsal:
            f0 = solver_state
            if not use_fs:
                # `made_jump` can be a tracer, hence the `is`.
                if made_jump is False:
                    # Fast-path for compilation in the common case.
                    k0 = terms.prod(f0, control)
                else:
                    k0 = lax.cond(
                        made_jump,
                        lambda: terms.vf_prod(t0, y0, args, control),
                        lambda: terms.prod(f0, control),  # noqa: F821
                    )
        else:
            if need_f0_or_k0:
                if use_fs:
                    f0 = terms.vf(t0, y0, args)
                else:
                    k0 = terms.vf_prod(t0, y0, args, control)

        #
        # Calculate `jac_f` and `jac_k` (maybe). That is to say, the Jacobian for use
        # throughout an implicit method. In practice this is for SDIRK and ESDIRK
        # methods, which use the same Jacobian throughout every stage.
        #

        jac_f = None
        jac_k = None
        if self.calculate_jacobian == CalculateJacobian.every_step:
            assert self.tableau.a_diagonal is not None
            # Skipping the first element to account for ESDIRK methods.
            assert all(
                x == self.tableau.a_diagonal[1] for x in self.tableau.a_diagonal[2:]
            )
            diagonal0 = self.tableau.a_diagonal[1]
            if use_fs:
                if y0 is not None:
                    assert f0 is not None
                jac_f = self.nonlinear_solver.jac(
                    _implicit_relation_f,
                    f0,
                    (diagonal0, terms.vf, terms.prod, t0, y0, args, control),
                )
            else:
                if y0 is not None:
                    assert k0 is not None
                jac_k = self.nonlinear_solver.jac(
                    _implicit_relation_k,
                    k0,
                    (diagonal0, terms.vf_prod, t0, y0, args, control),
                )
            del diagonal0

        #
        # Allocate `fs` or `ks` as a place to store the stage evaluations.
        #

        if use_fs or fsal:
            if f0 is None:
                # Only perform this trace if we have to; tracing can actually be
                # a bit expensive.
                f0_struct = eqx.filter_eval_shape(terms.vf, t0, y0, args)
            else:
                f0_struct = jax.eval_shape(lambda: f0)  # noqa: F821
        # else f0_struct deliberately left undefined, and is unused.

        num_stages = self.tableau.num_stages
        if use_fs:
            fs = jtu.tree_map(lambda f: jnp.zeros((num_stages,) + f.shape), f0_struct)
            ks = None
        else:
            fs = None
            ks = jtu.tree_map(lambda k: jnp.zeros((num_stages,) + jnp.shape(k)), y0)

        #
        # First stage. Defines `result`, `scan_first_stage`. Places `f0` and `k0` into
        # `fs` and `ks`. (+Redefines them if it's an implicit first stage.) Consumes
        # `f0` and `k0`.
        #

        if fsal:
            scan_first_stage = False
            result = RESULTS.successful
        else:
            if implicit_first_stage:
                scan_first_stage = False
                assert self.tableau.a_diagonal is not None
                diagonal0 = self.tableau.a_diagonal[0]
                if self.tableau.a_diagonal[0] == 1:
                    # No floating point error
                    t0_ = t1
                else:
                    t0_ = t0 + self.tableau.a_diagonal[0] * dt
                if use_fs:
                    if y0 is not None:
                        assert jac_f is not None
                    nonlinear_sol = self.nonlinear_solver(
                        _implicit_relation_f,
                        f0,
                        (diagonal0, terms.vf, terms.prod, t0_, y0, args, control),
                        jac_f,
                    )
                    f0 = nonlinear_sol.root
                    result = nonlinear_sol.result
                else:
                    if y0 is not None:
                        assert jac_k is not None
                    nonlinear_sol = self.nonlinear_solver(
                        _implicit_relation_k,
                        k0,
                        (diagonal0, terms.vf_prod, t0_, y0, args, control),
                        jac_k,
                    )
                    k0 = nonlinear_sol.root
                    result = nonlinear_sol.result
                del diagonal0, t0_, nonlinear_sol
            else:
                scan_first_stage = True
                result = RESULTS.successful

        if scan_first_stage:
            assert f0 is None
            assert k0 is None
        else:
            if use_fs:
                if y0 is not None:
                    assert f0 is not None
                fs = ω(fs).at[0].set(ω(f0)).ω
            else:
                if y0 is not None:
                    assert k0 is not None
                ks = ω(ks).at[0].set(ω(k0)).ω

        del f0, k0

        #
        # Iterate through the stages. Fills in `fs` and `ks`. Consumes
        # `scan_first_stage`.
        #

        def eval_stage(_carry, _input):
            _, _, _fs, _ks, _result = _carry
            _i, _a_lower_i, _a_diagonal_i, _a_predictor_i, _c_i = _input
            # Unwrap buffers. Take advantage of the fact that they're initialised at
            # zero, so that we don't really read from a location before its written to.
            _unsafe_fs_unwrapped = jtu.tree_map(lambda _, x: x[...], fs, _fs)
            _unsafe_ks_unwrapped = jtu.tree_map(lambda _, x: x[...], ks, _ks)

            #
            # Evaluate the linear combination of previous stages
            #

            if use_fs:
                _increment = vector_tree_dot(_a_lower_i, _unsafe_fs_unwrapped)
                _increment = terms.prod(_increment, control)
            else:
                _increment = vector_tree_dot(_a_lower_i, _unsafe_ks_unwrapped)
            _yi_partial = (y0**ω + _increment**ω).ω

            #
            # Figure out if we're computing a vector field ("f") or a
            # vector-field-product ("k")
            #
            # Ask for fi if we're using fs; ask for ki if we're using ks. Makes sense!
            # In addition, ask for fi if we're using an FSAL scheme, as we'll be passing
            # that on to the next step.
            #

            _return_fi = use_fs or fsal
            _return_ki = not use_fs

            #
            # Evaluate the stage
            #

            _ti = jnp.where(_c_i == 1, t1, t0 + _c_i * dt)  # No floating point error
            if self.tableau.implicit:
                assert _a_diagonal_i is not None
                # Predictor for where to start iterating from
                if _return_fi:
                    _f_pred = vector_tree_dot(_a_predictor_i, _unsafe_fs_unwrapped)
                else:
                    _k_pred = vector_tree_dot(_a_predictor_i, _unsafe_ks_unwrapped)
                # Determine Jacobian to use at this stage
                if self.calculate_jacobian == CalculateJacobian.every_stage:
                    if _return_fi:
                        _jac_f = self.nonlinear_solver.jac(
                            _implicit_relation_f,
                            _f_pred,
                            (
                                _a_diagonal_i,
                                terms.vf,
                                terms.prod,
                                _ti,
                                _yi_partial,
                                args,
                                control,
                            ),
                        )
                        _jac_k = None
                    else:
                        _jac_f = None
                        _jac_k = self.nonlinear_solver.jac(
                            _implicit_relation_k,
                            _k_pred,
                            (
                                _a_diagonal_i,
                                terms.vf,
                                terms.prod,
                                _ti,
                                _yi_partial,
                                args,
                                control,
                            ),
                        )
                else:
                    assert self.calculate_jacobian == CalculateJacobian.every_step
                    _jac_f = jac_f
                    _jac_k = jac_k
                # Solve nonlinear problem
                if _return_fi:
                    if y0 is not None:
                        assert _jac_f is not None
                    _nonlinear_sol = self.nonlinear_solver(
                        _implicit_relation_f,
                        _f_pred,
                        (
                            _a_diagonal_i,
                            terms.vf,
                            terms.prod,
                            _ti,
                            _yi_partial,
                            args,
                            control,
                        ),
                        _jac_f,
                    )
                    _fi = _nonlinear_sol.root
                    if _return_ki:
                        _ki = terms.prod(_fi, control)
                    else:
                        _ki = None
                else:
                    if _return_ki:
                        if y0 is not None:
                            assert _jac_k is not None
                        _nonlinear_sol = self.nonlinear_solver(
                            _implicit_relation_k,
                            _k_pred,
                            (
                                _a_diagonal_i,
                                terms.vf_prod,
                                _ti,
                                _yi_partial,
                                args,
                                control,
                            ),
                            _jac_k,
                        )
                        _fi = None
                        _ki = _nonlinear_sol.root
                    else:
                        assert False
                _result = update_result(_result, _nonlinear_sol.result)
                del _nonlinear_sol
            else:
                # Explicit stage
                if _return_fi:
                    _fi = terms.vf(_ti, _yi_partial, args)
                    if _return_ki:
                        _ki = terms.prod(_fi, control)
                    else:
                        _ki = None
                else:
                    _fi = None
                    if _return_ki:
                        _ki = terms.vf_prod(_ti, _yi_partial, args, control)
                    else:
                        assert False

            #
            # Store output
            #

            if use_fs:
                _fs = jtu.tree_map(lambda x, xs: xs.at[_i].set(x), _fi, _fs)
            else:
                _ks = jtu.tree_map(lambda x, xs: xs.at[_i].set(x), _ki, _ks)
            if self.tableau.ssal:
                _yi_partial_out = _yi_partial
            else:
                _yi_partial_out = None
            if fsal:
                _fi_out = _fi
            else:
                _fi_out = None
            return (_yi_partial_out, _fi_out, _fs, _ks, _result), None

        #
        # Iterate over stages
        #

        if scan_first_stage:
            tableau_a_lower = np.zeros((num_stages, num_stages))
            for i, a_lower_i in enumerate(self.tableau.a_lower):
                tableau_a_lower[i + 1, : i + 1] = a_lower_i
            tableau_a_diagonal = self.tableau.a_diagonal
            tableau_a_predictor = self.tableau.a_predictor
            tableau_c = np.zeros(num_stages)
            tableau_c[1:] = self.tableau.c
            i_init = 0
            assert tableau_a_diagonal is None
            assert tableau_a_predictor is None
        else:
            tableau_a_lower = np.zeros((num_stages - 1, num_stages))
            for i, a_lower_i in enumerate(self.tableau.a_lower):
                tableau_a_lower[i, : i + 1] = a_lower_i
            if self.tableau.a_diagonal is None:
                tableau_a_diagonal = None
            else:
                tableau_a_diagonal = self.tableau.a_diagonal[1:]
            if self.tableau.a_predictor is None:
                tableau_a_predictor = None
            else:
                tableau_a_predictor = np.zeros((num_stages - 1, num_stages))
                for i, a_predictor_i in enumerate(self.tableau.a_predictor):
                    tableau_a_predictor[i, : i + 1] = a_predictor_i
            tableau_c = self.tableau.c
            i_init = 1
        if self.tableau.ssal:
            y_dummy = y0
        else:
            y_dummy = None
        if fsal:
            f_dummy = jtu.tree_map(
                lambda x: jnp.zeros(x.shape, dtype=x.dtype), f0_struct
            )
        else:
            f_dummy = None
        if self.scan_kind is None:
            scan_kind = "checkpointed"
        else:
            scan_kind = self.scan_kind
        (y1_partial, f1, fs, ks, result), _ = eqxi.scan(
            eval_stage,
            (y_dummy, f_dummy, fs, ks, result),
            (
                np.arange(i_init, num_stages),
                tableau_a_lower,
                tableau_a_diagonal,
                tableau_a_predictor,
                tableau_c,
            ),
            buffers=lambda x: (x[2], x[3]),  # fs and ks
            kind=scan_kind,
            checkpoints="all",
        )
        del y_dummy, f_dummy, scan_first_stage

        #
        # Compute step output
        #

        if self.tableau.ssal:
            y1 = y1_partial
        else:
            if use_fs:
                increment = vector_tree_dot(self.tableau.b_sol, fs)
                increment = terms.prod(increment, control)
            else:
                increment = vector_tree_dot(self.tableau.b_sol, ks)
            y1 = (y0**ω + increment**ω).ω

        #
        # Compute error estimate
        #

        if use_fs:
            y_error = vector_tree_dot(self.tableau.b_error, fs)
            y_error = terms.prod(y_error, control)
        else:
            y_error = vector_tree_dot(self.tableau.b_error, ks)
        y_error = jtu.tree_map(
            lambda _y_error: jnp.where(is_okay(result), _y_error, jnp.inf),
            y_error,
        )  # i.e. an implicit step failed to converge

        #
        # Compute dense info
        #

        if use_fs:
            if fs is None:
                # Edge case for diffeqsolve(y0=None)
                ks = None
            else:
                ks = jax.vmap(lambda f: terms.prod(f, control))(fs)
        dense_info = dict(y0=y0, y1=y1, k=ks)

        #
        # Compute next solver state
        #

        if fsal:
            solver_state = f1
        else:
            solver_state = None

        return y1, y_error, dense_info, solver_state, result


class AbstractERK(AbstractRungeKutta):
    """Abstract base class for all Explicit Runge--Kutta solvers.

    Subclasses should include a class-level attribute `tableau`, an instance of
    [`diffrax.ButcherTableau`][].
    """

    calculate_jacobian = CalculateJacobian.never


class AbstractDIRK(AbstractRungeKutta, AbstractImplicitSolver):
    """Abstract base class for all Diagonal Implicit Runge--Kutta solvers.

    Subclasses should include a class-level attribute `tableau`, an instance of
    [`diffrax.ButcherTableau`][].
    """

    calculate_jacobian = CalculateJacobian.every_stage


class AbstractSDIRK(AbstractDIRK):
    """Abstract base class for all Singular Diagonal Implict Runge--Kutta solvers.

    Subclasses should include a class-level attribute `tableau`, an instance of
    [`diffrax.ButcherTableau`][].
    """

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if hasattr(cls, "tableau"):  # Abstract subclasses may not have a tableau.
            diagonal = cls.tableau.a_diagonal[0]
            assert (cls.tableau.a_diagonal == diagonal).all()

    calculate_jacobian = CalculateJacobian.every_step


class AbstractESDIRK(AbstractDIRK):
    """Abstract base class for all Explicit Singular Diagonal Implicit Runge--Kutta
    solvers.

    Subclasses should include a class-level attribute `tableau`, an instance of
    [`diffrax.ButcherTableau`][].
    """

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if hasattr(cls, "tableau"):  # Abstract subclasses may not have a tableau.
            assert cls.tableau.a_diagonal[0] == 0
            diagonal = cls.tableau.a_diagonal[1]
            assert (cls.tableau.a_diagonal[1:] == diagonal).all()

    calculate_jacobian = CalculateJacobian.every_step
