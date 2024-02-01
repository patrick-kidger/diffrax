import abc
import importlib.util
import threading
from typing import Generic, TypeVar

import equinox as eqx
import equinox.internal as eqxi
import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import io_callback
from jaxtyping import Array, PyTree

from ._custom_types import FloatScalarLike, IntScalarLike, RealScalarLike


_State = TypeVar("_State", bound=PyTree[Array])


class AbstractProgressMeter(eqx.Module, Generic[_State]):
    """Progress meters used to indicate how far along a solve is. Typically these
    perform some kind of printout as the solve progresses.
    """

    @abc.abstractmethod
    def init(self) -> _State:
        """Initialises the state for a new progress meter.

        **Arguments:**

        Nothing.

        **Returns:**

        The initial state for the progress meter.
        """

    @abc.abstractmethod
    def step(self, state: _State, progress: FloatScalarLike) -> _State:
        """Updates the progress meter. Called on every numerical step of a differential
        equation solve.

        **Arguments:**

        - `state`: the state from the previous step.
        - `progress`: how far along the solve is, as a number in `[0, 1]`.

        **Returns:**

        The updated state. In addition, the meter is expected to update as a
        side-effect.
        """

    @abc.abstractmethod
    def close(self, state: _State):
        """Closes the progress meter. Called at the end of a differential equation
        solve.

        **Arguments:**

        - `state`: the final state from the end of the solve.

        *Returns:**

        None.
        """


class NoProgressMeter(AbstractProgressMeter):
    """Indicates that no progress meter should be displayed during the solve."""

    def init(self) -> None:
        return None

    def step(self, state, progress: FloatScalarLike) -> None:
        return state

    def close(self, state):
        pass


NoProgressMeter.__init__.__doc__ = """**Arguments:**

Nothing.
"""


def _unvmap_min(x):  # No `eqxi.unvmap_min` at the moment.
    return -eqxi.unvmap_max(-x)


class _TextProgressMeterState(eqx.Module):
    progress: FloatScalarLike


def _print_percent_callback(progress):
    print(f"{100 * progress.item():.2f}%")


def _print_percent(progress):
    # `io_callback` would be preferable here, to indicate that it provides an output,
    # but that's not supported in vmap-of-while.
    progress = eqxi.nonbatchable(progress)  # check we'll only call the callback once.
    jax.debug.callback(_print_percent_callback, progress, ordered=True)
    return progress


class TextProgressMeter(AbstractProgressMeter):
    """A text progress meter, printing out e.g.:
    ```
    0.00%
    2.00%
    5.30%
    ...
    100.00%
    ```
    """

    minimum_increase: RealScalarLike = 0.02

    def init(self) -> _TextProgressMeterState:
        _print_percent(0.0)
        return _TextProgressMeterState(progress=jnp.array(0.0))

    def step(
        self, state: _TextProgressMeterState, progress: FloatScalarLike
    ) -> _TextProgressMeterState:
        # When `diffeqsolve(..., t0=..., t1=...)` are batched, then both
        # `state.progress` and `progress` will pick up a batch tracer.
        # (For the former, because the condition for the while-loop-over-steps becomes
        # batched, so necessarily everything in the body of the loop is as well.)
        #
        # We take a `min` over `progress` and a `max` over `state.progress`, as we want
        # to report the progress made over the worst batch element.
        state_progress = eqxi.unvmap_max(state.progress)
        del state
        progress = _unvmap_min(progress)
        pred = eqxi.nonbatchable(progress - state_progress > self.minimum_increase)

        # We only print if the progress has increased by at least `minimum_increase` to
        # avoid flooding the user with too many updates.
        next_progress = jax.lax.cond(
            pred,
            _print_percent,
            lambda _: state_progress,
            progress,
        )

        return _TextProgressMeterState(progress=next_progress)

    def close(self, state: _TextProgressMeterState):
        # As in `step`, we `unvmap` to handle batched state.
        # This means we only call the callback once.
        progress = _unvmap_min(state.progress)
        # Consumes `progress` without using it, to get the order of callbacks correct.
        progress = jax.debug.callback(
            lambda _: print("100.00%"), progress, ordered=True
        )


TextProgressMeter.__init__.__doc__ = """**Arguments:**

- `minimum_increase`: the minimum amount the progress has to have increased in order to
    print out a new line. The progress starts at 0 at the beginning of the solve, and
    increases to 1 at the end of the solve. Defaults to `0.02`, so that a new line is
    printed each time the progress increases another 2%.
"""


class _TqdmProgressMeterState(eqx.Module):
    progress_meter_id: IntScalarLike
    step: IntScalarLike


class TqdmProgressMeter(AbstractProgressMeter):
    """Uses tqdm to display a progress bar for the solve."""

    refresh_steps: int = 20

    def __check_init__(self):
        if importlib.util.find_spec("tqdm") is None:
            raise ValueError(
                "Cannot use `diffrax.TqdmProgressMeter` without `tqdm` installed. "
                "Install it via `pip install tqdm`."
            )

    def init(self) -> _TqdmProgressMeterState:
        # Not `pure_callback` because it's not a deterministic function of its input
        # arguments.
        # Not `debug.callback` because it has a return value.
        progress_meter_id = io_callback(
            _progress_meter_manager.init, jax.ShapeDtypeStruct((), jnp.int32)
        )
        progress_meter_id = eqxi.nonbatchable(progress_meter_id)
        return _TqdmProgressMeterState(
            progress_meter_id=progress_meter_id, step=jnp.array(0)
        )

    def step(
        self,
        state: _TqdmProgressMeterState,
        progress: FloatScalarLike,
    ) -> _TqdmProgressMeterState:
        # As in `TextProgressMeter`, then `state` may pick up a batch tracer from a
        # batched condition, so we need to handle that.
        #
        # In practice it should always be the case that this remains constant over the
        # solve, so we can just do a max to extract the single value we want.
        progress_meter_id = eqxi.unvmap_max(state.progress_meter_id)
        # What happens here is that all batch values for `state.step` start off in sync,
        # and then eventually will freeze their values as that batch element finishes
        # its solve. So take a `max` to get the true number of overall solve steps for
        # the batched system.
        step = eqxi.unvmap_max(state.step)
        del state
        # Track the slowest batch element.
        progress = _unvmap_min(progress)

        def update_progress_bar():
            # `io_callback` would be preferable here (to indicate the side-effect), but
            # that's not supported in vmap-of-while. (Even when none of the inputs to
            # the callback are batched.)
            jax.debug.callback(
                _progress_meter_manager.step, progress, progress_meter_id, ordered=True
            )

        # Here we update every `refresh_rate` steps in order to limit expensive
        # callbacks.
        jax.lax.cond(
            eqxi.nonbatchable(step % self.refresh_steps == 0),
            update_progress_bar,
            lambda: None,
        )

        return _TqdmProgressMeterState(
            progress_meter_id=progress_meter_id, step=step + 1
        )

    def close(self, state: _TqdmProgressMeterState):
        # `unvmap_max` as in `step`.
        progress_meter_id = eqxi.unvmap_max(state.progress_meter_id)
        # Pass in `step` to thread the order correctly. (`ordered=True` seems sketchy.
        # At the very least it doesn't also hold the order wrt
        # `jax.debug.callback(..., ordered=True)`.)
        # In addition, unvmap it to be sure the callback is only called once.
        step = eqxi.unvmap_max(state.step)
        del state
        io_callback(
            lambda idx, _: _progress_meter_manager.close(idx),
            None,
            progress_meter_id,
            step,
        )


TqdmProgressMeter.__init__.__doc__ = """**Arguments:**

- `refresh_steps`: the number of numerical steps between refreshing the bar. Used to
    limit how frequently the (potentially computationally expensive) bar update is
    performed.
"""


class _TqdmProgressMeterManager:
    """Host-side progress meter manager for TqdmProgressMeter."""

    def __init__(self):
        self.idx = 0
        self.bars = {}
        # Not sure how important a lock really is, but included just in case.
        self.lock = threading.Lock()

    def init(self) -> IntScalarLike:
        with self.lock:
            import tqdm  # pyright: ignore

            bar_format = (
                "{percentage:.2f}%|{bar}| [{elapsed}<{remaining}, {rate_fmt}{postfix}]"
            )
            bar = tqdm.tqdm(
                total=100,
                unit="%",
                bar_format=bar_format,
            )
            self.idx += 1
            self.bars[self.idx] = bar
            return np.array(self.idx, dtype=jnp.int32)

    def step(self, progress: FloatScalarLike, idx: IntScalarLike):
        with self.lock:
            bar = self.bars[int(idx)]
            bar.n = round(100 * float(progress), 2)
            bar.update(n=0)

    def close(self, idx: IntScalarLike):
        with self.lock:
            idx = int(idx)
            bar = self.bars[idx]
            bar.n = 100.0
            bar.update(n=0)
            bar.close()
            del self.bars[idx]


_progress_meter_manager = _TqdmProgressMeterManager()
