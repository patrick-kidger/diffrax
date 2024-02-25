import enum
import functools as ft
from dataclasses import dataclass, field
from typing import (
    cast,
    ClassVar,
    get_args,
    get_origin,
    Literal,
    Optional,
    Tuple,
    TYPE_CHECKING,
    Union,
)
from typing_extensions import TypeAlias

import equinox as eqx
import equinox.internal as eqxi
import jax
import jax.core
import jax.lax as lax
import jax.numpy as jnp
import jax.tree_util as jtu
import lineax.internal as lxi
import numpy as np
import optimistix as optx


if TYPE_CHECKING:
    from typing import ClassVar as AbstractClassVar
else:
    from equinox import AbstractClassVar
from equinox.internal import ω
from jaxtyping import Array, PyTree

from .._custom_types import (
    Args,
    BoolScalarLike,
    DenseInfo,
    RealScalarLike,
    sentinel,
    VF,
    Y,
)
from .._solution import is_okay, RESULTS, update_result
from .._term import AbstractTerm, MultiTerm, ODETerm, WrapTerm
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
    c1: float = 0.0

    # Properties implied by the above tableaus, e.g. used to define fast-paths.
    ssal: bool = field(init=False)
    fsal: bool = field(init=False)
    implicit: bool = field(init=False)
    num_stages: int = field(init=False)

    # Example!
    #
    # Consider a Butcher tableau:
    #
    # c1 | a11 a12 a13 a14
    # c2 | a21 a22 a23 a24
    # c3 | a31 a32 a33 a34
    # c4 | a41 a42 a43 a44
    # ---+----------------
    #    |  b1  b2  b3  b4
    #    |  β1  β2  β3  β4
    #
    # Let y0 be the input to the step, and let y1 denote the output of the step.
    #
    # Then the output is computed via
    # y1 = y0 + Σ_i bi ki
    # where ki = fi dt   (in the case of an ODE -- it is "fi dW" etc. for an SDE)
    # and fi = f(ci, zi)
    # and zi = y0 + Σ_j aij kj
    #
    # Note that "stage" may be used to refer to any of ki, fi, or zi.
    #
    # The error estimate is given by
    # err = Σ_i βi ki
    # (I.e. it is compute directly -- *not* as the difference of two solutions.)
    #
    # ---
    #
    # To encoder the above tableau in Diffrax, you would take:
    # c = np.array([c2, c3, c4])
    # b_sol = np.array([b1, b2, b3, b4])
    # b_error = np.array([β1, β2, β3, β3])
    # a_lower = (
    #    np.array([a21]),
    #    np.array([a31, a32]),
    #    np.array([a41, a42, a43]),
    # )
    # a_diagonal = np.array([a11, a22, a33, a44])  # Optional if all zero
    # c1 = c1  # Optional if zero
    #
    # Noting that a_diagonal and c1 are only used for implicit solvers, hence their
    # optionality.
    #
    # In addition we support an additional `a_predictor` tableau for implicit solvers.
    # This seems to be semi-new here; see
    # https://docs.kidger.site/diffrax/devdocs/predictor_dirk/

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

        lower_b_sol_equal = (self.b_sol[:-1] == self.a_lower[-1]).all().item()
        last_diagonal = 0 if self.a_diagonal is None else self.a_diagonal[-1]
        diagonal_b_sol_equal = (self.b_sol[-1] == last_diagonal).item()
        explicit_first_stage = (
            self.a_diagonal is None or (self.a_diagonal[0] == 0).item()
        )
        explicit_last_stage = (
            self.a_diagonal is None or (self.a_diagonal[-1] == 0).item()
        )
        # (vector field)-control product `k1` is the same across first/last stages.
        object.__setattr__(
            self,
            "fsal",
            lower_b_sol_equal and diagonal_b_sol_equal and explicit_first_stage,
        )
        # Solution `y1` is the same as the last stage
        object.__setattr__(
            self,
            "ssal",
            lower_b_sol_equal and diagonal_b_sol_equal and explicit_last_stage,
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


class MultiButcherTableau(eqx.Module):
    """Wraps multiple [`diffrax.ButcherTableau`][]s together. Used in some multi-tableau
    solvers, like IMEX methods.

    !!! important

        This API is not stable, and deliberately undocumented. (The reason is that we
        might yet adapt this to implement Stochastic Runge--Kutta methods.)
    """

    tableaus: tuple[ButcherTableau, ...]

    def __init__(self, *tableaus: ButcherTableau):
        self.tableaus = tableaus


MultiButcherTableau.__init__.__doc__ = """**Arguments:**

- `*tableaus`: the tableaus to wrap together.
"""


class CalculateJacobian(enum.IntEnum):
    """An enumeration of possible ways a Runga--Kutta method may wish to calculate a
    Jacobian.

    `never`: used for explicit Runga--Kutta methods.

    `every_stage`: the Jacobian is calculated once per stage. Used for DIRK methods.

    `first_stage`: the Jacobian is calculated once per step; in particular it is
        calculated in the first stage and re-used for every subsequent stage in the
        step. Used for SDIRK methods.

    `second_stage`: the Jacobian is calculated once per step; in particular it is
        calculated in the second stage and re-used for every subsequent stage in the
        step. Used for ESDIRK methods.
    """

    never = 0
    every_stage = 1
    first_stage = 2
    second_stage = 3


_SolverState: TypeAlias = Optional[tuple[BoolScalarLike, PyTree[Array]]]


# TODO: examine termination criterion for Newton iteration
# TODO: replace fi with fi=(zi + predictor), where this relation defines some zi, and
#       iterate to find zi, using zi=0 as the predictor. This should give better
#       numerical behaviour since the iteration is close to 0. (Although we have
#       multiplied by the increment of the control, i.e. dt, which is small...)
def _implicit_relation_f(fi, nonlinear_solve_args):
    # We pass stage_index, even without using it, so that custom nonlinear solvers
    # can special-case on the stage if they want to.
    (
        stage_index,
        diagonal,
        vf,
        prod,
        ti,
        yi_partial,
        args,
        control,
    ) = nonlinear_solve_args
    del stage_index
    diff = (
        fi**ω - vf(ti, (yi_partial**ω + diagonal * prod(fi, control) ** ω).ω, args) ** ω
    ).ω
    return diff


# TODO: consider dividing by diagonal and control
def _implicit_relation_k(ki, nonlinear_solve_args):
    # c.f:
    # https://github.com/SciML/DiffEqDevMaterials/blob/master/newton/output/main.pdf
    # (Bearing in mind that our ki is dt times smaller than theirs.)
    #
    # We pass stage_index, even without using it, so that custom nonlinear solvers
    # can special-case on the stage if they want to.
    stage_index, diagonal, vf_prod, ti, yi_partial, args, control = nonlinear_solve_args
    del stage_index
    diff = (
        ki**ω - vf_prod(ti, (yi_partial**ω + diagonal * ki**ω).ω, args, control) ** ω
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


def _filter_stop_gradient(x):
    dynamic, static = eqx.partition(x, eqx.is_inexact_array)
    dynamic = lax.stop_gradient(dynamic)
    return eqx.combine(dynamic, static)


def _is_jaxpr(x):
    return isinstance(x, (jax.core.Jaxpr, jax.core.ClosedJaxpr))


def _filter_maybe_cond(pred, branch, value):
    dynamic, static = eqx.partition(value, eqx.is_array)
    jaxpr_static = eqx.filter(static, _is_jaxpr, inverse=True)

    def branch1():
        new_dynamic, new_static = eqx.partition(branch(), eqx.is_array)
        new_jaxpr_static = eqx.filter(new_static, _is_jaxpr, inverse=True)
        assert eqx.tree_equal(jaxpr_static, new_jaxpr_static, typematch=True) is True
        return new_dynamic

    def branch2():
        return dynamic

    dynamic_out = lax.cond(pred, branch1, branch2)
    return eqx.combine(dynamic_out, static)


def _assert_same_structure(x, y):
    x = jax.eval_shape(lambda: x)
    y = jax.eval_shape(lambda: y)
    x, y = jtu.tree_map(lambda a: (a.shape, a.dtype), (x, y))
    return eqx.tree_equal(x, y) is True


class AbstractRungeKutta(AbstractAdaptiveSolver[_SolverState]):
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

    scan_kind: Union[None, Literal["lax", "checkpointed", "bounded"]] = None

    tableau: AbstractClassVar[Union[ButcherTableau, MultiButcherTableau]]
    calculate_jacobian: AbstractClassVar[CalculateJacobian]

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if hasattr(cls, "tableau"):  # Abstract subclasses may not have a tableau
            if isinstance(cls.tableau, ButcherTableau):
                if hasattr(cls, "term_structure"):
                    assert issubclass(cls.term_structure, AbstractTerm)
                else:
                    cls.term_structure = AbstractTerm
            elif isinstance(cls.tableau, MultiButcherTableau):
                if len({tab.num_stages for tab in cls.tableau.tableaus}) > 1:
                    raise ValueError("Tableaus must all have the same number of stages")
                if len([tab for tab in cls.tableau.tableaus if tab.implicit]) > 1:
                    raise ValueError("May have at most one implicit tableau")
                if hasattr(cls, "term_structure"):
                    assert get_origin(cls.term_structure) is MultiTerm
                    [_tmp] = get_args(cls.term_structure)
                    assert get_origin(_tmp) in (tuple, Tuple)
                    assert all(issubclass(x, AbstractTerm) for x in get_args(_tmp))
                else:
                    terms = tuple(
                        AbstractTerm for _ in range(len(cls.tableau.tableaus))
                    )
                    cls.term_structure = MultiTerm[tuple[terms]]  # pyright: ignore
            else:
                assert False

    def _common(self, terms, t0, t1, y0, args):
        # For simplicity we share `vf_expensive` and `fsal` across all tableaus.
        # TODO: could we make these work per-tableau?
        vf_expensive = terms.is_vf_expensive(t0, t1, y0, args)
        if isinstance(self.tableau, MultiButcherTableau):
            fsal = all(tab.fsal for tab in self.tableau.tableaus)
        else:
            fsal = self.tableau.fsal
        # If the vector field is expensive then we want to use vf_prods instead.
        # FSAL implies evaluating just the vector field, since we need to contract
        # the same vector field evaluation against two different controls.
        fsal = fsal and not vf_expensive
        return vf_expensive, fsal

    def func(
        self,
        terms: AbstractTerm,
        t0: RealScalarLike,
        y0: Y,
        args: Args,
    ) -> VF:
        return terms.vf(t0, y0, args)

    def init(
        self,
        terms: AbstractTerm,
        t0: RealScalarLike,
        t1: RealScalarLike,
        y0: Y,
        args: Args,
    ) -> _SolverState:
        _, fsal = self._common(terms, t0, t1, y0, args)
        if fsal:
            first_step = jnp.array(True)
            f0 = sentinel
            if type(terms) is WrapTerm:
                # Privileged optimisations for some common cases
                _terms = terms.term
                if type(_terms) is ODETerm:
                    f0 = jtu.tree_map(lambda x: jnp.zeros(x.shape, x.dtype), y0)
                elif type(_terms) is MultiTerm:
                    if all(type(x) is ODETerm for x in _terms.terms):
                        f0 = tuple(
                            jtu.tree_map(lambda x: jnp.zeros(x.shape, x.dtype), y0)
                            for _ in range(len(_terms.terms))
                        )
            if f0 is sentinel:
                # Must be initialiased at zero as it is inserted into `ks` which must be
                # initialised at zero.
                f0 = eqxi.eval_zero(self.func, terms, t0, y0, args)
            return first_step, f0
        else:
            return None

    def step(
        self,
        terms: AbstractTerm,  # pyright: ignore
        t0: RealScalarLike,
        t1: RealScalarLike,
        y0: Y,
        args: Args,
        solver_state: _SolverState,
        made_jump: BoolScalarLike,
    ) -> tuple[Y, Y, DenseInfo, _SolverState, RESULTS]:
        #
        # Alright, settle in for what is probably the most advanced Runge-Kutta
        # implementation on the planet.
        #
        # This is capable of handling all of:
        # - Explicit Runge--Kutta methods (ERK)
        # - Diagonal Implicit Runge--Kutta methods (DIRK)
        # - Singular Diagonal Implicit Runge--Kutta methods (SDIRK)
        # - Explicit Singular Diagonal Implicit Runge--Kutta methods (ESDIRK)
        # - Implicit-Explicit Runge--Kutta methods (IMEX)
        #
        # In all cases it can handle applications to both ODEs and SDEs.
        # Several of these are implicit methods. The latter two are multi-tableau
        # methods.
        #
        # Both ODEs and SDEs: this is the usual innovation with Diffrax. We treat
        # everything as a CDE against an arbitrary control. This also means we have a
        # distinction between f-space (vector field values) and k-space
        # ((vector field)-control products).
        #
        # Implicit methods: these all involve computing a Jacobian somewhere, and doing
        # a root find. Any root finder can be used, although in practice the chord
        # method is typical. Indeed it is common (SDIRK; ESDIRK) to reuse the Jacobian
        # between stages.
        #
        # Multi-tableau methods: these are cases where each term has a different
        # tableau, and their stages are interleaved. This means that the y-value at
        # which we evaluate each stage depends on the previous stages of all tableaus.
        # Note that these shouldn't be confused with splitting methods, where typically
        # we solve one term using one solver, and then another term using another
        # solver, without interleaving the stages. (Splitting methods instead interleave
        # steps.)
        #
        # The other main innovation here (besides the unification of all these different
        # solvers) is a JAX-specific thing: getting all of these to compile efficiently,
        # with some tricks to trace through the vector field as few times as possible.
        #
        # As usual with JAX (and with a sprinkle of Equinox innovations), everything is
        # also autovectorisable and autodifferentiable.
        #
        # This *doesn't* handle Fully Implicit Runge--Kutta methods (FIRK), as those
        # have a different computational structure (they're just one big nonlinear
        # solve).
        #
        # This also doesn't (yet) handle Stochastic Runge--Kutta methods (SRK), as those
        # still require a bit more infrastructure: generating space-time Levy areas, or
        # even space-space Levy areas.
        #

        vf_expensive, fsal = self._common(terms, t0, t1, y0, args)

        # The code below is actually quite generic: it handles a PyTree of Butcher
        # tableaus and a PyTree of terms. (Which must match each other.)
        # Our MultiTerm/MultiButcherTableau interface is slightly more restrictive, in
        # that it only admits PyTree structures of `*` or `(*, ...)`.
        if isinstance(self.tableau, ButcherTableau):
            assert isinstance(terms, AbstractTerm)
            tableaus = self.tableau
            implicit_tableau = self.tableau if self.tableau.implicit else None
            implicit_term = terms if self.tableau.implicit else None
        else:
            assert isinstance(terms, MultiTerm)
            tableaus = self.tableau.tableaus
            terms = cast(tuple, terms.terms)
            assert len(tableaus) == len(terms)
            for tab, term in zip(tableaus, terms):
                if tab.implicit:
                    implicit_tableau = tab
                    implicit_term = term
                    break
            else:
                implicit_tableau = None
                implicit_term = None
        terms: PyTree[AbstractTerm]
        assert jtu.tree_structure(terms, is_leaf=_is_term) == jtu.tree_structure(
            tableaus
        )

        #
        # We have a choice whether to evaluate `vf` to get vector field evaluations
        # ("values in f-space"), or to evaluate `vf_prod` to get (vector field)-control
        # products ("values in k-space").
        #
        # In addition we have a choice whether to *store* fs or ks. If we evaluate
        # `vf_prod` then we must store ks, as we can't (cheaply) reconstruct fs from ks.
        # If we evaluate `vf` then we can store either, as we can just do an
        # `fs`-control product prior to storing them.
        #
        # The first most important case is if evaluating the vector field is expensive.
        # The canonical example is solving optimise-then-discretise adjoint SDEs, for
        # which the diffusion term takes the form (dg/dy)dW, which is a vjp against the
        # control. This can be done most efficiently by never materialising the full
        # diffusion matrix (the Jacobian dg/dy): don't call `vf`, and instead work
        # directly with `vf_prod`.
        # Cases of this nature are communicated via the `vf_expensive` flag. (Which
        # in Diffrax by default is applied to all AdjointTerms with vector controls.)
        # - Verdict: eval_fs=False, store_fs=False
        #
        # If we don't hit the above case, we consider FSAL.
        # For any FSAL solver, we must evaluate `vf`: we need the final `f1` to pass to
        # the next step. (The control changes from step-to-step, so we cannot simply
        # pass `k1`.)
        # In addition if the solver has an implicit tableau, then we must store `fs`.
        # This is because to get the final f1, we need to do the implicit solve in
        # f-space, which means we need to store fs to predict the initial point for the
        # root finding operation.
        # - Verdict: eval_fs=True, store_fs=True.
        # If the solver is explicit-only, then we can store either. We choose to store
        # ks instead, as this is perhaps slightly more efficient: other downstream tasks
        # like error estimates and dense information use ks rather than fs.
        # - Verdict: eval_fs=True, store_fs=False
        #
        # For all other cases, we don't have any hard restrictions. It *may* be the case
        # that a user-provided term has an overloaded `vf_prod` to be more efficient.
        # (The canonical example is if `vf` is the product of two matrices and the
        # control is a vector: it's usually cheaper to do `A @ (B @ dx)` rather than
        # `(A @ B) @ dx`.) Moreover downstream tasks like error estimatess and dense
        # information still use ks rather than fs. So we also use ks in this case.
        # - Verdict: eval_fs=False, store_fs=False
        #
        if vf_expensive:
            eval_fs = False
            store_fs = False
            assert not fsal  # fsal is disabled in this case
        elif fsal:
            if implicit_tableau is None:
                eval_fs = True
                store_fs = False
            else:
                eval_fs = True
                store_fs = True
        else:
            eval_fs = False
            store_fs = False
        if not eval_fs:
            assert not store_fs

        #
        # We have a lot of PyTrees of various structures floating around. Here are some
        # helpers to map over each structure.
        #

        # Structure of `terms` and `tableaus`.
        def t_map(fn, *trees, implicit_val=sentinel):
            def _fn(tableau, *_trees):
                if tableau.implicit and implicit_val is not sentinel:
                    return implicit_val
                else:
                    return fn(*_trees)

            return jtu.tree_map(_fn, tableaus, *trees)

        # Structure of `y` and `k`.
        def y_map(fn, *trees):
            def _fn(_, *_trees):
                return fn(*_trees)

            return jtu.tree_map(_fn, y0, *trees)

        # Structure of `f`. Note that this is a suffix of `t_map`.
        def f_map(fn, *trees):
            def _fn(_, *_trees):
                return fn(*_trees)

            assert f0 is not _unused
            return jtu.tree_map(_fn, f0, *trees)

        def t_leaves(tree):
            return [x.value for x in jtu.tree_leaves(t_map(_Leaf, tree))]

        def ty_map(fn, *trees):
            return t_map(lambda *_trees: y_map(fn, *_trees), *trees)

        def get_implicit(xs):
            def _get_implicit_impl(term, x):
                nonlocal value
                if term is implicit_term:
                    if value is sentinel:
                        value = x
                    else:
                        assert False

            value = sentinel
            t_map(_get_implicit_impl, terms, xs)
            assert value is not sentinel
            return value

        dt = t1 - t0
        control = t_map(lambda term_i: term_i.contr(t0, t1), terms)
        if implicit_tableau is None:
            implicit_control = _unused
        else:
            implicit_control = get_implicit(control)

        def vf(t, y, *, implicit_val):
            _assert_same_structure(y, y0)
            _vf = lambda term_i, t_i: term_i.vf(t_i, y, args)
            out = t_map(_vf, terms, t, implicit_val=implicit_val)
            if f0 is not _unused:
                _assert_same_structure(out, f0)
            return out

        def vf_prod(t, y, *, implicit_val):
            _assert_same_structure(y, y0)
            _vf = lambda term_i, t_i, control_i: term_i.vf_prod(t_i, y, args, control_i)
            out = t_map(_vf, terms, t, control, implicit_val=implicit_val)
            t_map(ft.partial(_assert_same_structure, y0), out)
            return out

        def prod(f):
            if f0 is not _unused:
                _assert_same_structure(f, f0)
            _prod = lambda term_i, f_i, control_i: term_i.prod(f_i, control_i)
            out = t_map(_prod, terms, f, control)
            t_map(ft.partial(_assert_same_structure, y0), out)
            return out

        #
        # Now get `f0` from an FSAL condition if possible.
        # FSAL = first-same-as-last. It essentially refers to the last stage of the
        # previous step only being used in error estimates, but not in advancing the
        # solution. This means that it is also the value `vf(t0, y0)` in the this step.
        # So provided our first stage is explicit (=necessarily just `vf(t0, y0)`) then
        # we can skip evaluating our first stage.
        #
        # The only exception is on the very first step, or after a jump, in which case
        # our stored value is invalid and must be (re-)computed.
        #
        if fsal:
            assert solver_state is not None
            first_step, f0 = solver_state
            eval_first_stage = eqxi.unvmap_any(first_step | made_jump)
            init_stage_index = jnp.where(eval_first_stage, 0, 1)
            # We do `fs.at[0].set(f0)` below. If we're actually going to evaluate the
            # first stage, then zero out `f0` so that that is a no-op.
            f0 = jtu.tree_map(lambda x: jnp.where(eval_first_stage, 0, x), f0)
            if store_fs:
                k0 = _unused
            else:
                k0 = prod(f0)
        else:
            # Non-FSAL solvers just iterate over all stages.
            f0 = _unused
            k0 = _unused
            init_stage_index = 0
        del solver_state

        #
        # If using a DIRK or SDIRK implicit solver: we need to pick the location (in
        # f-space or k-space) at which to compute our first Jacobian.
        # See: https://docs.kidger.site/diffrax/devdocs/predictor_dirk/#first-stage
        #
        if self.calculate_jacobian == CalculateJacobian.never:  # Typically ERK methods
            f0_for_jac = _unused
            k0_for_jac = _unused
        else:
            if fsal:  # Typically ESDIRK methods.
                f0_for_jac = _unused
                k0_for_jac = _unused
            else:  # Typically DIRK or SDIRK methods.
                # Sadness. The extra evaluation increases compilation time, as we must
                # trace our vector field again.
                if eval_fs:
                    f0_for_jac = implicit_term.vf(t0, y0, args)  # pyright: ignore
                    k0_for_jac = _unused
                else:
                    f0_for_jac = _unused
                    k0_for_jac = implicit_term.vf_prod(t0, y0, args, implicit_control)  # pyright: ignore
                # (
                # Possible sneaky sadness-ameliorating ideas which we don't do here:
                # 1. Construct a candidate f0 or k0 by combining the stages of the
                #    previous step. I don't know of any theory for this but it sounds
                #    reasonable. As above the exact value here isn't that important.
                # 2. Add an extra explicit stage at the end of the previous step, to do
                #    the above `vf` or `vf_prod` evaluation for us (FSAL-like, although
                #    this would actually end up being SSAL). Note that if we implemented
                #    that as `lax.cond(implicit, nonlinear_solve, explict_step)` then we
                #    would get no compile-time speedup (the goal here) as both branches
                #    involve tracing the vector field. So we would have to
                #    unconditionally run the nonlinear solver -- which is bad for
                #    runtime performance. So we don't do this.
                # )

        #
        # Create the buffers we'll populate with our f- or k-evaluations.
        #

        num_stages = jtu.tree_leaves(tableaus)[0].num_stages
        # Must be initialised at zero as we later do matmuls against the
        # partially-filled arrays.
        if store_fs:
            assert f0 is not _unused
            fs = f_map(lambda x: jnp.zeros((num_stages,) + x.shape, x.dtype), f0)
            ks = _unused
        else:
            fs = _unused
            ks = t_map(
                lambda: y_map(
                    lambda x: jnp.zeros((num_stages,) + x.shape, x.dtype), y0
                ),
            )
        if fsal:
            # !!! This is only valid because:
            # - On the very first step, or if we have a jump, then `f0` and  `k0` are
            #   zero and this is a no-op;
            # - On later steps we have `init_stage_index=1` and thus don't write to
            #   index 0.
            # We recall that the `buffers` of
            # `eqxi.while_loop(..., kind="checkpointed", buffers=...)`
            # must not have the same location written to multiple times, as otherwise
            # we will get incorrect gradients.
            # Either way we are correctly following the principle of "only write once".
            if store_fs:
                fs = f_map(lambda x, xs: xs.at[0].set(x), f0, fs)
            else:
                ks = ty_map(lambda x, xs: xs.at[0].set(x), k0, ks)

        #
        # Transform our tableaus into full square tableaus. (Rather than just the
        # triangular ones in which they're stored.) This is needed so that we can do
        # matvecs against them, which can't be of variable length.
        # (We could maybe implement a variable-length matvec by using a while loop --
        # not clear that that would necessarily get good performance though. Not
        # benchmarked.)
        #

        y0_leaves = jtu.tree_leaves(y0)
        if len(y0_leaves) == 0:
            tableau_dtype = lxi.default_floating_dtype()
        else:
            tableau_dtype = jnp.result_type(*y0_leaves)

        def embed_a_lower(tab):
            tab_a_lower = np.zeros(
                (num_stages, num_stages), dtype=np.result_type(*tab.a_lower)
            )
            for i, a_lower_i in enumerate(tab.a_lower):
                tab_a_lower[i + 1, : i + 1] = a_lower_i
            return jnp.asarray(tab_a_lower, dtype=tableau_dtype)

        def embed_c(tab):
            tab_c = np.zeros(num_stages, dtype=tab.c.dtype)
            if tab.c1 is not None:
                tab_c[0] = tab.c1
            tab_c[1:] = tab.c
            return jnp.asarray(tab_c, dtype=jnp.result_type(t0, t1))

        tableaus_a_lower = t_map(embed_a_lower, tableaus)
        tableaus_c = t_map(embed_c, tableaus)

        if implicit_tableau is not None:
            implicit_diagonal = jnp.asarray(
                implicit_tableau.a_diagonal, dtype=tableau_dtype
            )
            implicit_predictor = np.zeros(
                (num_stages, num_stages),
                dtype=np.result_type(*implicit_tableau.a_predictor),
            )
            for i, a_predictor_i in enumerate(implicit_tableau.a_predictor):  # pyright: ignore
                implicit_predictor[i + 1, : i + 1] = a_predictor_i
            implicit_predictor = jnp.asarray(implicit_predictor, dtype=tableau_dtype)
            implicit_c = get_implicit(tableaus_c)

        if implicit_term is None:
            implicit_vf = _unused
            implicit_prod = _unused
            implicit_vf_prod = _unused
        else:
            if eval_fs:
                assert f0 is not _unused
                implicit_vf = eqx.filter_closure_convert(implicit_term.vf, t0, y0, args)
                implicit_prod = eqx.filter_closure_convert(
                    implicit_term.prod, get_implicit(f0), implicit_control
                )
                implicit_vf_prod = _unused
            else:
                implicit_vf = _unused
                implicit_prod = _unused
                implicit_vf_prod = eqx.filter_closure_convert(
                    implicit_term.vf_prod, t0, y0, args, implicit_control
                )

        #
        # Run the loop over stages. (This is what you signed up for, and it's taken us
        # several hundred lines of code just to get this far!)
        #

        def cond_stage(val):
            stage_index, *_ = val
            return stage_index < num_stages

        def rk_stage(val):
            stage_index, _, _, dyn_jac_f, dyn_jac_k, fs, ks, result = val
            jac_f = eqx.combine(dyn_jac_f, static_jac_f)
            jac_k = eqx.combine(dyn_jac_k, static_jac_k)
            old_result = result
            #
            # Start by getting the linear combination of previous stages.
            #
            a_lower_i = t_map(lambda tab: tab[stage_index], tableaus_a_lower)
            c_i = t_map(lambda tab: tab[stage_index], tableaus_c)
            # Unwrap buffers. This is only valid (=correct under autodiff) because we
            # follow a triangular pattern and don't read from a location before it is
            # written to, or write to the same location twice.
            # (The reads in the vector_tree_dots don't count, as the operands are zero.)
            if store_fs:
                assert fs is not _unused
                unsafe_fs = f_map(lambda x: x[...], fs)
                unsafe_ks = _unused
                increment = prod(t_map(vector_tree_dot, a_lower_i, unsafe_fs))
            else:
                assert ks is not _unused
                unsafe_fs = _unused
                unsafe_ks = ty_map(lambda x: x[...], ks)
                increment = t_map(vector_tree_dot, a_lower_i, unsafe_ks)
            yi_partial = y_map(_sum, y0, *t_leaves(increment))
            #
            # Find the y value at which to evaluate this stage.
            # If we have only explicit tableaus, then this is just the linear
            # combination we found above.
            # If we have an implicit tableau, then perform the implicit solve.
            # Note that we perform the solve in f-space or k-space; not y-space.
            #
            if implicit_tableau is None:
                implicit_fi = sentinel
                implicit_ki = sentinel
                yi = yi_partial
            else:
                implicit_diagonal_i = implicit_diagonal[stage_index]  # pyright: ignore
                implicit_predictor_i = implicit_predictor[stage_index]  # pyright: ignore
                implicit_c_i = implicit_c[stage_index]  # pyright: ignore
                # No floating point error
                implicit_ti = jnp.where(implicit_c_i == 1, t1, t0 + implicit_c_i * dt)
                if_first_stage = ft.partial(jnp.where, stage_index == 0)
                if eval_fs:
                    f_pred = get_implicit(
                        vector_tree_dot(implicit_predictor_i, unsafe_fs)
                    )
                    if not fsal:
                        # FSAL => explicit first stage so the choice of predictor
                        # doesn't matter.
                        f_pred = jtu.tree_map(if_first_stage, f0_for_jac, f_pred)
                    assert f0 is not _unused
                    f_implicit_args = (
                        stage_index,
                        implicit_diagonal_i,
                        implicit_vf,
                        implicit_prod,
                        implicit_ti,
                        yi_partial,
                        args,
                        implicit_control,
                    )
                    k_pred = _unused
                    k_implicit_args = _unused
                else:
                    f_pred = _unused
                    f_implicit_args = _unused
                    k_pred = vector_tree_dot(
                        implicit_predictor_i, get_implicit(unsafe_ks)
                    )
                    if not fsal:
                        # FSAL implies explicit first stage so the choice of predictor
                        # doesn't matter.
                        k_pred = jtu.tree_map(if_first_stage, k0_for_jac, k_pred)
                    k_implicit_args = (
                        stage_index,
                        implicit_diagonal_i,
                        implicit_vf_prod,
                        implicit_ti,
                        yi_partial,
                        args,
                        implicit_control,
                    )

                def eval_f_jac():
                    return self.root_finder.init(  # pyright: ignore
                        lambda y, a: (_implicit_relation_f(y, a), None),
                        lax.stop_gradient(f_pred),
                        _filter_stop_gradient(f_implicit_args),
                        options={},
                        f_struct=jax.eval_shape(lambda: f_pred),
                        aux_struct=None,
                        tags=frozenset(),
                    )

                def eval_k_jac():
                    return self.root_finder.init(  # pyright: ignore
                        lambda y, a: (_implicit_relation_k(y, a), None),
                        lax.stop_gradient(k_pred),
                        _filter_stop_gradient(k_implicit_args),
                        options={},
                        f_struct=jax.eval_shape(lambda: k_pred),
                        aux_struct=None,
                        tags=frozenset(),
                    )

                if self.calculate_jacobian == CalculateJacobian.every_stage:
                    if eval_fs:
                        jac_f = eval_f_jac()
                        jac_k = _unused
                    else:
                        jac_f = _unused
                        jac_k = eval_k_jac()
                else:
                    if self.calculate_jacobian == CalculateJacobian.first_stage:
                        assert len(set(implicit_tableau.a_diagonal)) == 1  # pyright: ignore
                        jac_stage_index = 0
                    else:
                        assert self.calculate_jacobian == CalculateJacobian.second_stage
                        assert implicit_tableau.a_diagonal[0] == 0  # pyright: ignore
                        assert len(set(implicit_tableau.a_diagonal[1:])) == 1  # pyright: ignore
                        jac_stage_index = 1
                        stage_index = eqxi.nonbatchable(stage_index)
                    # These `stop_gradients` are needed to work around the lack of
                    # symbolic zeros in `custom_vjp`s.
                    if eval_fs:
                        jac_f = _filter_stop_gradient(jac_f)
                        jac_f = _filter_maybe_cond(
                            stage_index == jac_stage_index, eval_f_jac, jac_f
                        )
                        jac_k = _unused
                    else:
                        jac_f = _unused
                        jac_k = _filter_stop_gradient(jac_k)
                        jac_k = _filter_maybe_cond(
                            stage_index == jac_stage_index, eval_k_jac, jac_k
                        )
                if eval_fs:
                    jac_f = eqxi.nondifferentiable(jac_f, name="jac_f")
                    nonlinear_sol = optx.root_find(
                        _implicit_relation_f,
                        self.root_finder,  # pyright: ignore
                        f_pred,
                        f_implicit_args,
                        options=dict(init_state=jac_f),
                        throw=False,
                        max_steps=self.root_find_max_steps,  # pyright: ignore
                    )
                    implicit_fi = nonlinear_sol.value
                    implicit_ki = _unused
                    implicit_inc = implicit_term.prod(implicit_fi, implicit_control)  # pyright: ignore
                else:
                    assert not fsal
                    jac_k = eqxi.nondifferentiable(jac_k, name="jac_k")
                    nonlinear_sol = optx.root_find(
                        _implicit_relation_k,
                        self.root_finder,  # pyright: ignore
                        k_pred,
                        k_implicit_args,
                        options=dict(init_state=jac_k),
                        throw=False,
                        max_steps=self.root_find_max_steps,  # pyright: ignore
                    )
                    implicit_fi = _unused
                    implicit_ki = implicit_inc = nonlinear_sol.value
                yi = y_map(
                    lambda a, b: a + implicit_diagonal_i * b, yi_partial, implicit_inc
                )
                result = update_result(result, RESULTS.promote(nonlinear_sol.result))
            #
            # Now evaluate our vector field at the value yi.
            # If we had an implicit tableau then we can skip evaluating the vector field
            # for that tableau, as we did the solve in f-space or k-space and already
            # have its value.
            #
            # No floating point error
            ti = t_map(lambda _c_i: jnp.where(_c_i == 1, t1, t0 + _c_i * dt), c_i)
            if eval_fs:
                assert not vf_expensive
                assert implicit_fi is not _unused
                fi = vf(ti, yi, implicit_val=implicit_fi)
                if store_fs:
                    ki = _unused
                else:
                    ki = prod(fi)
            else:
                assert implicit_ki is not _unused
                assert not store_fs
                fi = _unused
                ki = vf_prod(ti, yi, implicit_val=implicit_ki)
            #
            # Update our outputs
            #
            if fsal:
                assert fi is not _unused
                f1_for_fsal = fi
            else:
                f1_for_fsal = _unused
            if store_fs:
                assert fi is not _unused
                assert fs is not _unused
                fs = f_map(lambda x, xs: xs.at[stage_index].set(x), fi, fs)
            else:
                assert ki is not _unused
                assert ks is not _unused
                ks = ty_map(lambda x, xs: xs.at[stage_index].set(x), ki, ks)
            nonlocal const_result
            if const_result is const_result_sentinel:
                const_result = result is old_result
            else:
                const_result = const_result and (result is old_result)
            dyn_jac_f = eqx.filter(jac_f, eqx.is_array)
            dyn_jac_k = eqx.filter(jac_k, eqx.is_array)
            return (
                stage_index + 1,
                yi,
                f1_for_fsal,
                dyn_jac_f,
                dyn_jac_k,
                fs,
                ks,
                result,
            )

        def buffers(val):
            *_, fs, ks, _ = val
            return fs, ks

        if fsal:
            assert f0 is not _unused
            dummy_f = f0
        else:
            dummy_f = _unused
        if self.calculate_jacobian == CalculateJacobian.never:
            jac_f = _unused
            jac_k = _unused
        else:
            # Set the initial Jacobian to be the identity matrix.
            # For DIRK and SDIRK methods then the choice here doesn't matter; we compute
            # the Jacobian straight away.
            # For ESDIRK methods, this is the Jacobian of an explicit step.
            if eval_fs:
                assert f0 is not _unused
                f_implicit_args = (
                    jnp.array(0),
                    # zero diagonal == identity matrix as the Jacobian
                    jnp.array(0.0, dtype=implicit_diagonal.dtype),  # pyright: ignore
                    implicit_vf,
                    implicit_prod,
                    t0,
                    y0,
                    args,
                    implicit_control,
                )
                jac_f = self.root_finder.init(  # pyright: ignore
                    lambda y, a: (_implicit_relation_f(y, a), None),
                    jtu.tree_map(jnp.zeros_like, get_implicit(f0)),
                    _filter_stop_gradient(f_implicit_args),
                    options={},
                    f_struct=jax.eval_shape(lambda: get_implicit(f0)),
                    aux_struct=None,
                    tags=frozenset(),
                )
                jac_k = _unused
            else:
                k_implicit_args = (
                    jnp.array(0),
                    # zero diagonal == identity matrix as the Jacobian
                    jnp.array(0.0, dtype=implicit_diagonal.dtype),  # pyright: ignore
                    implicit_vf_prod,
                    t0,
                    y0,
                    args,
                    implicit_control,
                )
                jac_f = _unused
                jac_k = self.root_finder.init(  # pyright: ignore
                    lambda y, a: (_implicit_relation_k(y, a), None),
                    jtu.tree_map(jnp.zeros_like, y0),
                    _filter_stop_gradient(k_implicit_args),
                    options={},
                    f_struct=jax.eval_shape(lambda: y0),
                    aux_struct=None,
                    tags=frozenset(),
                )
        dyn_jac_f, static_jac_f = eqx.partition(jac_f, eqx.is_array)
        dyn_jac_k, static_jac_k = eqx.partition(jac_k, eqx.is_array)
        init_val = (
            init_stage_index,
            y0,
            dummy_f,
            dyn_jac_f,
            dyn_jac_k,
            fs,
            ks,
            RESULTS.successful,
        )
        const_result = const_result_sentinel = object()
        # Needs to be an `eqxi.while_loop` as:
        # (a) we may have variable length: e.g. an FSAL explicit RK scheme will have one
        #     more stage on the first step.
        # (b) to work around a limitation of JAX's autodiff being unable to express
        #     "triangular computations" (every stage depends on all previous stages)
        #     without spurious copies.
        final_val = eqxi.while_loop(
            cond_stage,
            rk_stage,
            init_val,
            max_steps=num_stages,
            buffers=buffers,
            kind="checkpointed" if self.scan_kind is None else self.scan_kind,
            checkpoints=num_stages,
            base=num_stages,
        )
        _, y1, f1_for_fsal, _, _, fs, ks, result = final_val
        assert const_result is not const_result_sentinel
        if const_result:
            result = RESULTS.successful

        #
        # Calculate outputs: the final `y1` from our step, any dense information, etc.
        #

        if store_fs:
            assert ks == _unused
            if fs is None:
                # Handle edge-case of y0=None
                ks = None
            else:
                ks = jax.vmap(prod)(fs)
        if any(not tableau.ssal for tableau in jtu.tree_leaves(tableaus)):

            def _increment(tab_i, k_i):
                return vector_tree_dot(
                    jnp.asarray(tab_i.b_sol, dtype=tableau_dtype), k_i
                )

            increment = t_map(_increment, tableaus, ks)
            y1 = y_map(_sum, y0, *t_leaves(increment))
        y_error = t_map(
            lambda tab, k: vector_tree_dot(
                jnp.asarray(tab.b_error, dtype=tableau_dtype), k
            ),
            tableaus,
            ks,
        )
        y_error = y_map(_sum, *t_leaves(y_error))
        y_error = jtu.tree_map(
            lambda _y_error: jnp.where(is_okay(result), _y_error, jnp.inf),
            y_error,
        )  # i.e. an implicit step failed to converge
        dense_info = dict(y0=y0, y1=y1, k=ks)
        if fsal:
            new_solver_state = False, f1_for_fsal
        else:
            new_solver_state = None
        return y1, y_error, dense_info, new_solver_state, result


class AbstractERK(AbstractRungeKutta):
    """Abstract base class for all Explicit Runge--Kutta solvers.

    Subclasses should include a class-level attribute `tableau`, an instance of
    [`diffrax.ButcherTableau`][].
    """

    calculate_jacobian: ClassVar[CalculateJacobian] = CalculateJacobian.never


class AbstractDIRK(AbstractRungeKutta, AbstractImplicitSolver):
    """Abstract base class for all Diagonal Implicit Runge--Kutta solvers.

    Subclasses should include a class-level attribute `tableau`, an instance of
    [`diffrax.ButcherTableau`][].
    """

    calculate_jacobian: ClassVar[CalculateJacobian] = CalculateJacobian.every_stage


class AbstractSDIRK(AbstractDIRK):
    """Abstract base class for all Singular Diagonal Implict Runge--Kutta solvers.

    Subclasses should include a class-level attribute `tableau`, an instance of
    [`diffrax.ButcherTableau`][].
    """

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if hasattr(cls, "tableau"):  # Abstract subclasses may not have a tableau.
            assert isinstance(cls.tableau, ButcherTableau)
            assert cls.tableau.a_diagonal is not None
            diagonal = cls.tableau.a_diagonal[0]
            assert (cls.tableau.a_diagonal == diagonal).all()

    calculate_jacobian: ClassVar[CalculateJacobian] = CalculateJacobian.first_stage


class AbstractESDIRK(AbstractDIRK):
    """Abstract base class for all Explicit Singular Diagonal Implicit Runge--Kutta
    solvers.

    Subclasses should include a class-level attribute `tableau`, an instance of
    [`diffrax.ButcherTableau`][].
    """

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if hasattr(cls, "tableau"):  # Abstract subclasses may not have a tableau.
            assert isinstance(cls.tableau, ButcherTableau)
            assert cls.tableau.a_diagonal is not None
            assert cls.tableau.a_diagonal[0] == 0
            diagonal = cls.tableau.a_diagonal[1]
            assert (cls.tableau.a_diagonal[1:] == diagonal).all()

    calculate_jacobian: ClassVar[CalculateJacobian] = CalculateJacobian.second_stage
