import abc
from dataclasses import dataclass, field
from typing import Optional, Tuple

import equinox as eqx
import jax
import jax.lax as lax
import jax.numpy as jnp
import numpy as np

from ..custom_types import Bool, DenseInfo, PyTree, Scalar
from ..misc import ContainerMeta, ω
from ..solution import is_okay, RESULTS, update_result
from ..term import AbstractTerm
from .base import AbstractAdaptiveSolver, AbstractImplicitSolver, vector_tree_dot


def _scan(*sequences):
    for x in sequences:
        if x is not None:
            length = len(x)
            break
    else:
        raise ValueError("Must have at least one non-None iterable")

    def _check(_x):
        assert len(_x) == length
        return _x

    sequences = [[None] * length if x is None else _check(x) for x in sequences]
    return zip(*sequences)


# Entries must be np.arrays, and not jnp.arrays, so that we can index into them during
# trace time.
@dataclass(frozen=True)
class ButcherTableau:
    """The Butcher tableau for an explicit or diagonal Runge--Kutta method."""

    # Explicit RK methods
    c: np.ndarray
    b_sol: np.ndarray
    b_error: np.ndarray
    a_lower: Tuple[np.ndarray]

    # Implicit RK methods
    a_diagonal: Optional[np.ndarray] = None
    a_predictor: Optional[Tuple[np.ndarray]] = None

    # Determine the use of fast-paths
    ssal: bool = field(init=False)
    fsal: bool = field(init=False)

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
            implicit = False
        else:
            assert self.a_predictor is not None
            implicit = True
        if implicit:
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
    [the developer documentation](../../devdocs/predictor_dirk). U#sed for diagonal
    implicit Runge--Kutta methods only.

Whether the solver exhibits either the FSAL or SSAL properties is determined
automatically.
"""


class CalculateJacobian(metaclass=ContainerMeta):
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


_SolverState = Optional[PyTree]


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

    scan_stages: bool = False

    term_structure = jax.tree_structure(0)

    @property
    @abc.abstractmethod
    def tableau(self) -> ButcherTableau:
        pass

    @property
    @abc.abstractmethod
    def calculate_jacobian(self) -> CalculateJacobian:
        pass

    def func(
        self,
        terms: AbstractTerm,
        t0: Scalar,
        y0: PyTree,
        args: PyTree,
    ) -> PyTree:
        return terms.vf(t0, y0, args)

    def init(
        self,
        terms: AbstractTerm,
        t0: Scalar,
        t1: Scalar,
        y0: PyTree,
        args: PyTree,
    ) -> _SolverState:
        vf_expensive = terms.is_vf_expensive(t0, t1, y0, args)
        fsal = self.tableau.fsal and not vf_expensive
        if fsal:
            return terms.vf(t0, y0, args)
        else:
            return None

    def step(
        self,
        terms: AbstractTerm,
        t0: Scalar,
        t1: Scalar,
        y0: PyTree,
        args: PyTree,
        solver_state: _SolverState,
        made_jump: Bool,
    ) -> Tuple[PyTree, PyTree, DenseInfo, _SolverState, RESULTS]:

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

        _vf_expensive = terms.is_vf_expensive(t0, t1, y0, args)
        _implicit_later_stages = self.tableau.a_diagonal is not None and any(
            self.tableau.a_diagonal[1:] != 0
        )
        implicit_first_stage = (
            self.tableau.a_diagonal is not None and self.tableau.a_diagonal[0] != 0
        )
        fsal = self.tableau.fsal and not _vf_expensive
        ssal = self.tableau.ssal
        if _implicit_later_stages and fsal:
            use_fs = True
        elif _vf_expensive:
            use_fs = False
        else:  # Choice not as important here; we use ks for minor efficiency reasons.
            use_fs = False
        del _vf_expensive, _implicit_later_stages

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
                k0 = lax.cond(
                    made_jump,
                    lambda _: terms.vf_prod(t0, y0, args, control),
                    lambda _: terms.prod(f0, control),  # noqa: F821
                    None,
                )
        else:
            if (
                self.calculate_jacobian == CalculateJacobian.every_step
                or implicit_first_stage
                or not self.scan_stages
            ):
                # The gamut of conditions under which we need to evaluate `f0` or `k0`.
                #
                # If we're computing the Jacobian at the start of the step, then we
                # need this as a linearisation point.
                #
                # If the first stage is implicit, then we need this as a predictor for
                # where to start iterating from.
                #
                # If we're not scanning stages then we're definitely not deferring this
                # evaluation to the scan loop, so get it done now.
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

        if use_fs or (fsal and self.scan_stages):
            # Only perform this trace if we have to; tracing can actually be a bit
            # expensive.
            f0_struct = eqx.filter_eval_shape(terms.vf, t0, y0, args)
        # else f0_struct deliberately left undefined, and is unused.

        num_stages = len(self.tableau.c) + 1
        if use_fs:
            fs = jax.tree_map(lambda f: jnp.empty((num_stages,) + f.shape), f0_struct)
            ks = None
        else:
            fs = None
            ks = jax.tree_map(lambda k: jnp.empty((num_stages,) + jnp.shape(k)), y0)

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
                if self.tableau.diagonal[0] == 1:
                    # No floating point error
                    t0_ = t1
                else:
                    t0_ = t0 + self.tableau.diagonal[0] * dt
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
                scan_first_stage = self.scan_stages
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

        if self.scan_stages:

            def _vector_tree_dot(_x, _y, _i):
                del _i
                return vector_tree_dot(_x, _y)

        else:

            def _vector_tree_dot(_x, _y, _i):
                return vector_tree_dot(_x, ω(_y)[:_i].ω)

        def eval_stage(_carry, _input):
            _, _, _fs, _ks, _result = _carry
            _i, _a_lower_i, _a_diagonal_i, _a_predictor_i, _c_i = _input

            #
            # Evaluate the linear combination of previous stages
            #

            if use_fs:
                _increment = _vector_tree_dot(_a_lower_i, _fs, _i)  # noqa: F821
                _increment = terms.prod(_increment, control)
            else:
                _increment = _vector_tree_dot(_a_lower_i, _ks, _i)  # noqa: F821
            _yi_partial = (y0**ω + _increment**ω).ω

            #
            # Is this an implicit or explicit stage?
            #

            if self.tableau.a_diagonal is None:
                _implicit_stage = False
            else:
                if self.scan_stages:
                    if scan_first_stage:  # noqa: F821
                        _diagonal = self.tableau.a_diagonal
                    else:
                        _diagonal = self.tableau.a_diagonal[1:]
                    _implicit_stage = any(_diagonal != 0)
                    if _implicit_stage and any(_diagonal == 0):
                        assert False, (
                            "Cannot have a mix of implicit and "
                            "explicit stages when scanning"
                        )
                    del _diagonal
                else:
                    _implicit_stage = _a_diagonal_i != 0

            #
            # Figure out if we're computing a vector field ("f") or a
            # vector-field-product ("k")
            #
            # Ask for fi if we're using fs; ask for ki if we're using ks. Makes sense!
            # In addition, ask for fi if we're on the last stage and are using
            # an FSAL scheme, as we'll be passing that on to the next step. If
            # we're scanning the stages then every stage uses the same logic so
            # override the last iteration check.
            #

            _last_iteration = _i == num_stages - 1
            _return_fi = use_fs or (fsal and (self.scan_stages or _last_iteration))
            _return_ki = not use_fs
            del _last_iteration

            #
            # Evaluate the stage
            #

            _ti = jnp.where(_c_i == 1, t1, t0 + _c_i * dt)  # No floating point error
            if _implicit_stage:
                assert _a_diagonal_i is not None
                # Predictor for where to start iterating from
                if _return_fi:
                    _f_pred = _vector_tree_dot(_a_predictor_i, fs, _i)  # noqa: F821
                else:
                    _k_pred = _vector_tree_dot(_a_predictor_i, ks, _i)  # noqa: F821
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
                _fs = ω(_fs).at[_i].set(ω(_fi)).ω
            else:
                _ks = ω(_ks).at[_i].set(ω(_ki)).ω
            if ssal:
                _yi_partial_out = _yi_partial
            else:
                _yi_partial_out = None
            if fsal:
                _fi_out = _fi
            else:
                _fi_out = None
            return (_yi_partial_out, _fi_out, _fs, _ks, _result), None

        if self.scan_stages:
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
            if ssal:
                y_dummy = y0
            else:
                y_dummy = None
            if fsal:
                f_dummy = jax.tree_map(
                    lambda x: jnp.zeros(x.shape, dtype=x.dtype), f0_struct
                )
            else:
                f_dummy = None
            (y1_partial, f1, fs, ks, result), _ = lax.scan(
                eval_stage,
                (y_dummy, f_dummy, fs, ks, result),
                (
                    np.arange(i_init, num_stages),
                    tableau_a_lower,
                    tableau_a_diagonal,
                    tableau_a_predictor,
                    tableau_c,
                ),
            )
            del y_dummy, f_dummy
        else:
            assert not scan_first_stage
            if self.tableau.a_diagonal is None:
                a_diagonal = None
            else:
                a_diagonal = self.tableau.a_diagonal[1:]
            for i, a_lower_i, a_diagonal_i, a_predictor_i, c_i in _scan(
                range(1, num_stages),
                self.tableau.a_lower,
                a_diagonal,
                self.tableau.a_predictor,
                self.tableau.c,
            ):
                (yi_partial, fi, fs, ks, result), _ = eval_stage(
                    (None, None, fs, ks, result),
                    (i, a_lower_i, a_diagonal_i, a_predictor_i, c_i),
                )
            y1_partial = yi_partial
            f1 = fi
            del a_diagonal, yi_partial, fi
        del scan_first_stage, _vector_tree_dot

        #
        # Compute step output
        #

        if ssal:
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
        y_error = jax.tree_map(
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
        if cls.tableau is not None:  # Abstract subclasses may not have a tableau.
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
        if cls.tableau is not None:  # Abstract subclasses may not have a tableau.
            assert cls.tableau.a_diagonal[0] == 0
            diagonal = cls.tableau.a_diagonal[1]
            assert (cls.tableau.a_diagonal[1:] == diagonal).all()

    calculate_jacobian = CalculateJacobian.every_step
