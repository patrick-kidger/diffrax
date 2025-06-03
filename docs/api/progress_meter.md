# Progress meters

As the solve progresses, progress meters offer the ability to have some kind of output indicating how far along the solve has progressed. For example, to display a text output every now and again, or to fill a [tqdm](https://github.com/tqdm/tqdm) progress bar.

??? abstract "`diffrax.AbstractProgressMeter`"

    ::: diffrax.AbstractProgressMeter
        options:
            members:
                - init
                - step
                - close

---

::: diffrax.NoProgressMeter
    options:
        members:
            - __init__

::: diffrax.TextProgressMeter
    options:
        members:
            - __init__

::: diffrax.TqdmProgressMeter
    options:
        members:
            - __init__
