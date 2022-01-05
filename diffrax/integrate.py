from typing import Optional

import equinox as eqx
import jax
import jax.numpy as jnp

from .custom_types import Array, Bool, Int, PyTree, Scalar
from .global_interpolation import DenseInterpolation
from .misc import (
    bounded_while_loop,
    branched_error_if,
    error_if,
    HadInplaceUpdate,
    ravel_pytree,
    unvmap_max,
)
from .saveat import SaveAt
from .solution import RESULTS, Solution
from .solver import AbstractSolver
from .step_size_controller import AbstractStepSizeController, ConstantStepSize


class _State(eqx.Module):
    # Evolving state during the solve
    y: Array["state"]  # noqa: F821
    tprev: Scalar
    tnext: Scalar
    made_jump: Bool
    solver_state: PyTree
    controller_state: PyTree
    result: RESULTS
    step: Int
    # Output that is .at[].set() updated during the solve (and their indices)
    saveat_ts_index: Scalar
    ts: Array["times"]  # noqa: F821
    ys: Array["times"]  # noqa: F821
    save_index: Int
    dense_ts: Optional[Array["times + 1"]]  # noqa: F821
    dense_infos: Optional[Array["times"]]  # noqa: F821
    dense_save_index: Int


class _InnerState(eqx.Module):
    saveat_ts_index: Int
    ts: Array
    ys: Array
    save_index: Array


def _init_diffeqsolve(
    solver,
    t0,
    t1,
    y0,
    args,
    dt0,
    saveat,
    stepsize_controller,
    solver_state,
    controller_state,
    max_steps,
):
    """
    - Performs error checking.
    - Normalises PyTree states down to just a flat Array.
    - Normalises time: if t0 > t1 then flip things around.
    - Initialise loop state.
    """

    if dt0 is not None:
        error_if((t1 - t0) * dt0 <= 0, "Must have (t1 - t0) * dt0 > 0")

    direction = jnp.where(t0 < t1, 1, -1)
    y, unravel_y = ravel_pytree(y0)
    solver = solver.wrap(t0, y0, args, direction)
    stepsize_controller = stepsize_controller.wrap(unravel_y, direction)
    t0 = t0 * direction
    t1 = t1 * direction
    if dt0 is not None:
        dt0 = dt0 * direction
    if saveat.ts is not None:
        saveat = eqx.tree_at(lambda s: s.ts, saveat, saveat.ts * direction)

    if saveat.ts is not None:
        error_if(
            saveat.ts[1:] < saveat.ts[:-1],
            "saveat.ts must be increasing or decreasing.",
        )
        error_if(
            (saveat.ts > t1) | (saveat.ts < t0), "saveat.ts must lie between t0 and t1."
        )

    tprev = t0
    if controller_state is None:
        (tnext, controller_state) = stepsize_controller.init(
            t0, t1, y, dt0, args, solver
        )
    else:
        error_if(dt0 is None, "Must provide `dt0` if providing `controller_state`.")
        tnext = t0 + dt0
    tnext = jnp.minimum(tnext, t1)

    if solver_state is None:
        solver_state = solver.init(t0, tnext, y, args)

    out_size = 0
    if saveat.t0:
        out_size += 1
    if saveat.ts is not None:
        out_size += len(saveat.ts)
    if saveat.steps:
        # We have no way of knowing how many steps we'll actually end up taking, and
        # XLA doesn't support dynamic shapes. So we just have to allocate the maximum
        # amount of steps we can possibly take.
        error_if(
            max_steps is None,
            "`max_steps=None` is incompatible with `saveat.steps=True`",
        )
        out_size += max_steps
    if saveat.t1 and not saveat.steps:
        out_size += 1

    step = 0
    saveat_ts_index = 0
    save_index = 0
    made_jump = jnp.array(False)
    ts = jnp.full(out_size, jnp.nan)
    ys = jnp.full((out_size, y.size), jnp.nan)
    result = jnp.array(RESULTS.successful)
    if saveat.dense:
        error_if(t0 == t1, "Cannot save dense output if t0 == t1")
        (
            _,
            _,
            dense_info,
            _,
            _,
        ) = solver.step(tprev, tnext, y, args, solver_state, made_jump)
        dense_ts = jnp.full(max_steps + 1, jnp.nan)
        error_if(
            max_steps is None,
            "`max_steps=None` is incompatible with `saveat.dense=True`",
        )
        _make_full = lambda x: jnp.full((max_steps,) + jnp.shape(x), jnp.nan)
        dense_infos = jax.tree_map(_make_full, dense_info)
        dense_save_index = 0
    else:
        dense_ts = None
        dense_infos = None
        dense_save_index = None

    state = _State(
        y=y,
        tprev=tprev,
        tnext=tnext,
        made_jump=made_jump,
        solver_state=solver_state,
        controller_state=controller_state,
        result=result,
        step=step,
        saveat_ts_index=saveat_ts_index,
        ts=ts,
        ys=ys,
        save_index=save_index,
        dense_ts=dense_ts,
        dense_infos=dense_infos,
        dense_save_index=dense_save_index,
    )

    return solver, stepsize_controller, saveat, args, unravel_y, direction, state


def _save(state, t):
    ts = state.ts
    ys = state.ys
    save_index = state.save_index
    y = state.y

    ts = ts.at[save_index].set(t)
    ys = ys.at[save_index].set(y)
    save_index = save_index + 1

    return eqx.tree_at(
        lambda s: (s.ts, s.ys, s.save_index), state, (ts, ys, save_index)
    )


@eqx.filter_jit
def diffeqsolve(
    solver: AbstractSolver,
    t0: Scalar,
    t1: Scalar,
    y0: PyTree,
    dt0: Optional[Scalar],
    args: Optional[PyTree] = None,
    *,
    saveat: SaveAt = SaveAt(t1=True),
    stepsize_controller: AbstractStepSizeController = ConstantStepSize(),
    max_steps: Optional[Scalar] = 4096,
    throw: bool = True,
    solver_state: Optional[PyTree] = None,
    controller_state: Optional[PyTree] = None,
) -> Solution:
    """Solves a differential equation.

    This function is the main entry point for solving all kinds of initial value
    problems, whether they are ODEs, SDEs, or CDEs.

    The differential equation is integrated from `t0` to `t1`.

    **Main arguments:**

    These are the arguments most commonly used day-to-day.

    - `solver`: The solver for the differential equation. The vector field of the
        differential equation is specified when instantiating the solver.
    - `t0`: The start of the region of integration.
    - `t1`: The end of the region of integration.
    - `y0`: The initial value. This can be any PyTree of JAX arrays. (Or types that
        can be coerced to JAX arrays, like Python floats.)
    - `dt0`: The step size to use for the first step. If using fixed step sizes then
        this will also be the step size for all other steps. (Except the last one,
        which may be slightly smaller and clipped to `t1`.) If set as `None` then the
        initial step size will be determined automatically if possible.
    - `args`: Any additional arguments to pass to the vector field.
    - `saveat`: What times to save the solution of the differential equation. Defaults
        to just the last time `t1`. (Keyword-only argument.)
    - `stepsize_controller`: How to change the step size as the integration progresses.
        Defaults to using a fixed constant step size. (Keyword-only argument.)

    **Other arguments:**

    These arguments are infrequently used, and for most purposes you shouldn't need to
    understand these. All of these are keyword-only arguments.

    - `max_steps`: The maximum number of steps to take before quitting the computation
        unconditionally.

        Can also be set to `None` to allow an arbitrary number of steps, although this
        will disable backpropagation via discretise-then-optimise (backpropagation via
        optimise-then-discretise will still work), and also disables
        `saveat.steps=True` and `saveat.dense=True`.

        Note that using larger values of `max_steps` will start to increase compilation
        time, so try to use the smallest value that is reasonable for your problem.

    - `throw`: Whether to raise an exception if the integration fails for any reason.

        If `True` then an integration failure will either raise a `ValueError` (when
        not using `jax.jit`) or print a warning message (when using `jax.jit`).

        If `False` then the returned solution object will have a `result` field
        indicating whether any failures occurred.

        Possible failures include for example hitting `max_steps`, or the problem
        becoming too stiff to integrate. (For most purposes these failures are
        unusual.)

        !!! note

            Note that when `jax.vmap`-ing a differential equation solve, then
            `throw=True` means that an exception will be raised if any batch element
            fails. You may prefer to set `throw=False` and inspect the `result` field
            of the returned solution object, to determine which batch elements
            succeeded and which failed.

    - `solver_state`: Some initial state for the solver. Can be useful when for example
        using a reversible solver to recompute a solution.

    - `controller_state`: Some initial state for the step size controller.

    **Returns:**

    Returns a [`diffrax.Solution`][] object specifying the solution to the differential
    equation.

    **Raises:**

    - `ValueError` for bad inputs.
    - `RuntimeError` if `throw=True`, not using `jax.jit`, and the integration fails
        (e.g. hitting the maximum number of steps).

    !!! note
        It is possible to have `t1 < t0`, in which case integration proceeds backwards
        in time.
    """

    #
    # Initial set-up
    #

    if dt0 is not None:
        # Allow setting t0 as an int with dt0 as a float. (We need consistent
        # types for JAX to be happy with the bounded_while_loop below.)
        dtype = jnp.result_type(t0, t1, dt0)
        t0 = jnp.asarray(t0, dtype=dtype)
        t1 = jnp.asarray(t1, dtype=dtype)
        dt0 = jnp.asarray(dt0, dtype=dtype)

    (
        solver,
        stepsize_controller,
        saveat,
        args,
        unravel_y,
        direction,
        init_state,
    ) = _init_diffeqsolve(
        solver,
        t0,
        t1,
        y0,
        args,
        dt0,
        saveat,
        stepsize_controller,
        solver_state,
        controller_state,
        max_steps,
    )

    if saveat.t0:
        init_state = _save(init_state, t0)

    if saveat.dense:
        dense_ts = init_state.dense_ts
        dense_ts = dense_ts.at[0].set(t0)
        init_state = eqx.tree_at(lambda s: s.dense_ts, init_state, dense_ts)

    #
    # Main loop
    #

    def cond_fun(state):
        return (state.tprev < t1) & (state.result == RESULTS.successful)

    def body_fun(state, inplace):

        #
        # Actually do some differential equation solving! Make numerical steps, adapt
        # step sizes, all that jazz.
        #

        (y, y_error, dense_info, solver_state, solver_result,) = solver.step(
            state.tprev, state.tnext, state.y, args, state.solver_state, state.made_jump
        )

        (
            keep_step,
            tprev,
            tnext,
            made_jump,
            controller_state,
            stepsize_controller_result,
        ) = stepsize_controller.adapt_step_size(
            state.tprev,
            state.tnext,
            state.y,
            y,
            args,
            y_error,
            solver.order,
            state.controller_state,
        )

        #
        # Do some book-keeping.
        #

        # The 1e-6 tolerance means that we don't end up with too-small intervals for
        # dense output, which then gives numerically unstable answers due to floating
        # point errors.
        tnext = jnp.where(tnext > t1 - 1e-6, t1, tnext)
        tprev = jnp.minimum(tprev, t1)

        # The other parts of the mutable state are kept/not-kept (based on whether the
        # step was accepted) by the stepsize controller. But it doesn't get access to
        # these parts, so we do them here.
        keep = lambda a, b: jnp.where(keep_step, a, b)
        y = keep(y, state.y)
        solver_state = jax.tree_map(keep, solver_state, state.solver_state)
        made_jump = keep(made_jump, state.made_jump)
        solver_result = keep(solver_result, RESULTS.successful)

        # TODO: if we ever support events, then they should go in here.
        # In particular the thing to be careful about is in the `if saveat.steps`
        # branch below, where we want to make sure that it is the value of `y` at
        # `tprev` that is actually saved. (And not just the value of `y` at the
        # previous step's `tnext`, i.e. immediately before the jump.)

        # Store the first unsuccessful result we get whilst iterating (if any).
        result = state.result
        result = jnp.where(result == RESULTS.successful, solver_result, result)
        result = jnp.where(
            result == RESULTS.successful, stepsize_controller_result, result
        )

        # Count the number of steps, just for statistical purposes.
        step = state.step + 1

        #
        # Store the output produced from this numerical step.
        # This is a bit involved, and uses the `inplace` function passed as an argument
        # to this body function.
        # This is because we need to make in-place updates to store our results, but
        # doing is a bit of a hassle inside `bounded_while_loop`. (See its docstring
        # for details.)
        #

        saveat_ts_index = state.saveat_ts_index
        ts = state.ts
        ys = state.ys
        save_index = state.save_index
        dense_ts = state.dense_ts
        dense_infos = state.dense_infos
        dense_save_index = state.dense_save_index
        made_inplace_update = False

        if saveat.ts is not None:
            made_inplace_update = True

            def _saveat_get(_saveat_ts_index):
                return saveat.ts[jnp.minimum(_saveat_ts_index, len(saveat.ts) - 1)]

            def _cond_fun(_state):
                _saveat_ts_index = _state.saveat_ts_index
                _saveat_t = _saveat_get(_saveat_ts_index)
                return (_saveat_t <= state.tnext) & (_saveat_ts_index < len(saveat.ts))

            def _body_fun(_state, _inplace):
                _saveat_ts_index = _state.saveat_ts_index
                _ts = _state.ts
                _ys = _state.ys
                _save_index = _state.save_index

                _inplace = _inplace.merge(inplace)

                def _maybe_inplace(x, i, u):
                    return _inplace(x).at[i].set(jnp.where(keep_step, u, x[i]))

                _interpolator = solver.interpolation_cls(
                    t0=state.tprev, t1=state.tnext, **dense_info
                )

                _saveat_t = _saveat_get(_saveat_ts_index)
                _saveat_y = _interpolator.evaluate(_saveat_t)
                _saveat_ts_index = _saveat_ts_index + 1

                _ts = _maybe_inplace(_ts, _save_index, _saveat_t)
                _ys = _maybe_inplace(_ys, _save_index, _saveat_y)
                _save_index = _save_index + 1

                return _InnerState(
                    saveat_ts_index=_saveat_ts_index,
                    ts=_ts,
                    ys=_ys,
                    save_index=_save_index,
                )

            init_inner_state = _InnerState(
                saveat_ts_index=saveat_ts_index, ts=ts, ys=ys, save_index=save_index
            )
            final_inner_state = bounded_while_loop(
                _cond_fun, _body_fun, init_inner_state, max_steps=len(saveat.ts)
            )

            saveat_ts_index = final_inner_state.saveat_ts_index
            ts = final_inner_state.ts
            ys = final_inner_state.ys
            save_index = final_inner_state.save_index

        def maybe_inplace(x, i, u):
            return inplace(x).at[i].set(jnp.where(keep_step, u, x[i]))

        if saveat.steps:
            made_inplace_update = True
            ts = maybe_inplace(ts, save_index, tprev)
            ys = maybe_inplace(ys, save_index, y)
            save_index = save_index + keep_step

        if saveat.dense:
            made_inplace_update = True
            dense_ts = maybe_inplace(dense_ts, dense_save_index, tprev)
            dense_infos = jax.tree_map(
                lambda x, u: maybe_inplace(x, dense_save_index, u),
                dense_infos,
                dense_info,
            )
            dense_save_index = dense_save_index + keep_step

        if made_inplace_update:
            ts = HadInplaceUpdate(ts)
            ys = HadInplaceUpdate(ys)
            dense_ts = HadInplaceUpdate(dense_ts)
            dense_infos = jax.tree_map(HadInplaceUpdate, dense_infos)

        new_state = _State(
            y=y,
            tprev=tprev,
            tnext=tnext,
            made_jump=made_jump,
            solver_state=solver_state,
            controller_state=controller_state,
            result=result,
            step=step,
            saveat_ts_index=saveat_ts_index,
            ts=ts,
            ys=ys,
            save_index=save_index,
            dense_ts=dense_ts,
            dense_infos=dense_infos,
            dense_save_index=dense_save_index,
        )

        return new_state

    final_state = bounded_while_loop(cond_fun, body_fun, init_state, max_steps)
    result = jnp.where(
        cond_fun(final_state), RESULTS.max_steps_reached, final_state.result
    )

    #
    # Finish up
    #

    error_index = unvmap_max(result)
    branched_error_if(
        throw & (result != RESULTS.successful),
        error_index,
        RESULTS.reverse_lookup,
        RuntimeError,
    )

    # saveat.steps will include the final timepoint anyway
    if saveat.t1 and not saveat.steps:
        final_state = _save(final_state, t1)

    if saveat.t0 or saveat.t1 or saveat.steps or (saveat.ts is not None):
        ts = final_state.ts
        ts = jnp.where(direction == 1, ts, -ts[::-1])
        ys = jax.vmap(unravel_y)(final_state.ys)
    else:
        ts = None
        ys = None
    if saveat.controller_state:
        controller_state = final_state.controller_state
    else:
        controller_state = None
    if saveat.solver_state:
        solver_state = final_state.solver_state
    else:
        solver_state = None
    if saveat.dense:
        interpolation = DenseInterpolation(
            ts=final_state.dense_ts,
            ts_size=final_state.dense_save_index,
            interpolation_cls=solver.interpolation_cls,
            infos=final_state.dense_infos,
            unravel_y=unravel_y,
            direction=direction,
        )
    else:
        interpolation = None

    stats = {"num_steps": final_state.step}

    return Solution(
        t0=t0,
        t1=t1,
        ts=ts,
        ys=ys,
        controller_state=controller_state,
        solver_state=solver_state,
        interpolation=interpolation,
        stats=stats,
        result=result,
    )
