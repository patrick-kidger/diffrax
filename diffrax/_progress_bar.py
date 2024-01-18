import abc

import equinox as eqx
import jax
import jax.numpy as jnp
from jax.experimental import io_callback
from jaxtyping import PyTree

from ._custom_types import FloatScalarLike, IntScalarLike


class AbstractProgressBar(eqx.Module):
    @abc.abstractmethod
    def init(self) -> PyTree:
        """Creates a new progress bar and returns its unique identifier"""

    @abc.abstractmethod
    def step(self, state: PyTree, progress: FloatScalarLike) -> PyTree:
        """Updates the progress bar and returns the last printed progress value"""

    @abc.abstractmethod
    def close(self, state: PyTree):
        """Closes the progress bar"""


class NoProgressBar(AbstractProgressBar):
    def init(self) -> None:
        return None

    def step(self, state, progress: FloatScalarLike) -> None:
        return state

    def close(self, state):
        pass


class _TextProgressBarState(eqx.Module):
    progress: FloatScalarLike


class TextProgressBar(AbstractProgressBar):
    minimum_increase: float = 2.0

    def init(self) -> _TextProgressBarState:
        return _TextProgressBarState(progress=0.0)

    def step(
        self, state: _TextProgressBarState, progress: FloatScalarLike
    ) -> _TextProgressBarState:
        def update_aux(progress):
            # currently
            # jax.debug.print("{progress:.2f}%", progress=progress)
            # does not work, so we have to round the progress ourselves
            jax.debug.print("{}%", jnp.round(progress, 1), ordered=True)
            return progress

        # we only print if the progress has increased by at least
        # `minimum_increase` to avoid flooding the user
        # with too many updates
        next_progress = jax.lax.cond(
            progress - state.progress > self.minimum_increase,
            update_aux,
            lambda p: state.progress,
            progress,
        )

        return _TextProgressBarState(progress=next_progress)

    def close(self, state: _TextProgressBarState):
        jax.debug.print("100%", ordered=True)


TextProgressBar.__init__.__doc__ = """**Arguments:**

- `minimum_increase`: minimum increase of progress bar in percent
"""


class _TqdmProgressBarState(eqx.Module):
    progress_bar_id: IntScalarLike
    step: IntScalarLike


class TqdmProgressBar(AbstractProgressBar):
    refresh_rate: int = 20

    def init(self) -> _TqdmProgressBarState:
        progress_bar_id = io_callback(
            _progress_bar_manager.init, jax.ShapeDtypeStruct(tuple(), jnp.int32)
        )

        return _TqdmProgressBarState(progress_bar_id=progress_bar_id, step=0)

    def step(
        self,
        state: _TqdmProgressBarState,
        progress: FloatScalarLike,
    ) -> _TqdmProgressBarState:
        def update_aux(progress_bar_id, progress):
            io_callback(
                _progress_bar_manager.step,
                None,
                progress,
                progress_bar_id,
                ordered=True,
            )

        # here we update every `refresh_rate` steps in order
        # to limit `io_callback` expensive calls
        jax.lax.cond(
            state.step % self.refresh_rate == 0,
            update_aux,
            lambda _, p: None,
            state.progress_bar_id,
            progress,
        )

        return _TqdmProgressBarState(
            step=state.step + 1, progress_bar_id=state.progress_bar_id
        )

    def close(self, state: _TqdmProgressBarState):
        io_callback(
            _progress_bar_manager.close, None, state.progress_bar_id, ordered=True
        )


TqdmProgressBar.__init__.__doc__ = """**Arguments:**

- `refresh_rate`: number of steps between refreshes
"""


class _TqdmProgressBarManager:
    """Host-side progress bar manager for TqdmProgressBar"""

    def __init__(self):
        self.idx = 0
        self.bars = {}

    def init(self) -> IntScalarLike:
        import tqdm

        self.idx += 1
        bar = tqdm.tqdm(total=100)
        self.bars[self.idx] = bar
        return jnp.array(self.idx, dtype=jnp.int32)

    def step(self, progress: FloatScalarLike, idx: IntScalarLike):
        bar = self.bars[int(idx)]
        bar.n = float(progress)
        bar.update()

    def close(self, idx: IntScalarLike):
        idx = int(idx)
        bar = self.bars[int(idx)]
        bar.n = float(100.0)
        bar.update()
        bar.close()
        del self.bars[idx]


_progress_bar_manager = _TqdmProgressBarManager()
