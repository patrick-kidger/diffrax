# Step size controllers

The list of step size controllers is as follows.


??? "`diffrax.AbstractStepSizeController`"

    All of the classes implement the following interface specified by [`diffrax.AbstractStepSizeController`][].

    The exact details of this interface are only really useful if you're using the [Manual stepping](../usage/manual-stepping.md) interface; otherwise this is all just internal to the library.

    ::: diffrax.AbstractStepSizeController
        selection:
            members:
                - wrap
                - init
                - adapt_step_size

---

::: diffrax.ConstantStepSize
    selection:
        members: false

::: diffrax.StepTo
    selection:
        members:
            - __init__
            - ts

::: diffrax.IController
    selection:
        members:
            - __init__
            - rtol
            - atol
            - dtmin
            - dtmax
            - force_dtmin
            - step_ts
            - jump_ts
            - ifactor
            - dfactor
            - norm
            - safety
