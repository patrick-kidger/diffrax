# Progress meters

As the solve progresses, progress meters offer the ability to have some kind of output indicating how far along the solve has progressed. For example, to display a text output every now and again, or to fill a [tqdm](https://github.com/tqdm/tqdm) progress bar.

??? abstract "`diffrax.AbstractProgressMeter`"

    An abstract base class for all progress meters.

    **Methods:**

    - `init()`
    - `step()`
    - `close()`

---

### `diffrax.NoProgressMeter`

A progress meter that does nothing.

### `diffrax.TextProgressMeter`

A progress meter that prints text to the console.

### `diffrax.TqdmProgressMeter`

A progress meter that displays a tqdm progress bar.
