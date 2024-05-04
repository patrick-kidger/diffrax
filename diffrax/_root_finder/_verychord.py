from collections.abc import Callable
from typing import Any, cast

import equinox as eqx
import jax
import jax.lax as lax
import jax.numpy as jnp
import jax.tree_util as jtu
import lineax as lx
import lineax.internal as lxi
import optimistix as optx
from equinox.internal import ω
from jaxtyping import Array, Bool, PyTree, Scalar
from lineax.internal import complex_to_real_dtype

from .._custom_types import Y


ω = cast(Callable, ω)


def _small(diffsize: Scalar) -> Bool[Array, ""]:
    # TODO(kidger): make a more careful choice here -- the existence of this
    # function is pretty ad-hoc.
    resolution = 10 ** (2 - jnp.finfo(diffsize.dtype).precision)
    return diffsize < resolution


def _diverged(rate: Scalar) -> Bool[Array, ""]:
    return jnp.invert(jnp.isfinite(rate)) | (rate > 2)


def _converged(factor: Scalar, tol: float) -> Bool[Array, ""]:
    return (factor > 0) & (factor < tol)


class _VeryChordState(eqx.Module):
    linear_state: tuple[lx.AbstractLinearOperator, PyTree[Any]]
    diff: Y
    diffsize: Scalar
    diffsize_prev: Scalar
    result: optx.RESULTS
    step: Scalar


class _NoAux(eqx.Module):
    fn: Callable

    def __call__(self, y, args):
        out, aux = self.fn(y, args)
        del aux
        return out


class VeryChord(optx.AbstractRootFinder):
    """The Chord method of root finding.

    As `optimistix.Chord`, except that in Runge--Kutta methods, the linearisation point
    is recomputed per-step and not per-stage. (This is computationally cheaper.)

    !!! info "Advanced notes"

        In terms of how this matches the Optimistix API, this is done by supporting the
        option `self.init(..., options=dict(init_state=...))`, in which case it will
        directly return the provided state instead of computing it. This makes it
        possible to manually call `self.init` at an earlier point around the desired
        linearisation point.
    """

    rtol: float
    atol: float
    norm: Callable[[PyTree], Scalar] = optx.max_norm
    kappa: float = 1e-2
    linear_solver: lx.AbstractLinearSolver = lx.AutoLinearSolver(well_posed=None)

    def init(
        self,
        fn: Callable,
        y: Y,
        args: PyTree[Any],
        options: dict[str, Any],
        f_struct: PyTree[jax.ShapeDtypeStruct],
        aux_struct: PyTree[jax.ShapeDtypeStruct],
        tags: frozenset[object],
    ) -> _VeryChordState:
        try:
            init_state = options["init_state"]
        except KeyError:
            jac = lx.JacobianLinearOperator(_NoAux(fn), y, args, tags=tags)
            jac = lx.linearise(jac)
            init_later_state = self.linear_solver.init(jac, options={})
            dynamic, static = eqx.partition(init_later_state, eqx.is_array)
            dynamic = lax.stop_gradient(dynamic)
            init_later_state = eqx.combine(dynamic, static)
            linear_state = (jac, init_later_state)
            y_leaves = jtu.tree_leaves(y)
            if len(y_leaves) == 0:
                y_dtype = lxi.default_floating_dtype()
            else:
                y_dtype = jnp.result_type(*y_leaves)
            diff_dtype = complex_to_real_dtype(y_dtype)
            init_state = _VeryChordState(
                linear_state=linear_state,
                diff=jtu.tree_map(lambda x: jnp.full(x.shape, jnp.inf, x.dtype), y),
                diffsize=jnp.array(jnp.inf, dtype=diff_dtype),
                diffsize_prev=jnp.array(1.0, dtype=diff_dtype),
                result=optx.RESULTS.successful,
                step=jnp.array(0),
            )
        else:
            assert isinstance(init_state, _VeryChordState)
        return init_state

    def step(
        self,
        fn: Callable,
        y: Y,
        args: PyTree[Any],
        options: dict[str, Any],
        state: _VeryChordState,
        tags: frozenset[object],
    ) -> tuple[Y, _VeryChordState, Any]:
        del options, tags
        fx, aux = fn(y, args)
        jac, linear_state = state.linear_state
        linear_state = lax.stop_gradient(linear_state)
        sol = lx.linear_solve(
            jac, fx, self.linear_solver, state=linear_state, throw=False
        )
        diff = sol.value
        new_y = (y**ω - diff**ω).ω

        with jax.numpy_dtype_promotion("standard"):
            scale = (self.atol + self.rtol * ω(new_y).call(jnp.abs)).ω
            diffsize = self.norm((diff**ω / scale**ω).ω)
        new_state = _VeryChordState(
            linear_state=state.linear_state,
            diff=diff,
            diffsize=diffsize,
            diffsize_prev=state.diffsize,
            result=optx.RESULTS.promote(sol.result),
            step=state.step + 1,
        )
        return new_y, new_state, aux

    def terminate(
        self,
        fn: Callable,
        y: Y,
        args: PyTree,
        options: dict[str, Any],
        state: _VeryChordState,
        tags: frozenset[object],
    ):
        del fn, y, args, options, tags
        # TODO(kidger): perform only one iteration when solving a linear system!
        at_least_two = state.step >= 2
        rate = state.diffsize / state.diffsize_prev
        factor = state.diffsize * rate / (1 - rate)
        small = _small(state.diffsize)
        diverged = _diverged(rate)
        converged = _converged(factor, self.kappa)
        terminate = at_least_two & (small | diverged | converged)
        terminate_result = optx.RESULTS.where(
            jnp.invert(small) & (diverged | jnp.invert(converged)),
            optx.RESULTS.nonlinear_divergence,
            optx.RESULTS.successful,
        )
        linsolve_fail = state.result != optx.RESULTS.successful
        result = optx.RESULTS.where(linsolve_fail, state.result, terminate_result)
        terminate = linsolve_fail | terminate
        return terminate, result

    def postprocess(
        self,
        fn: Callable,
        y: Y,
        aux: Any,
        args: PyTree[Any],
        options: dict[str, Any],
        state: _VeryChordState,
        tags: frozenset[object],
        result: optx.RESULTS,
    ) -> tuple[Y, Any, dict[str, Any]]:
        return y, aux, {}


VeryChord.__init__.__doc__ = """**Arguments:**

- `rtol`: Relative tolerance for terminating the solve.
- `atol`: Absolute tolerance for terminating the solve.
- `norm`: The norm used to determine the difference between two iterates in the 
    convergence criteria. Should be any function `PyTree -> Scalar`, for example
    `optimistix.max_norm`.
- `kappa`: A tolerance for the early convergence check.
- `linear_solver`: The linear solver used to compute the Newton step.
"""
