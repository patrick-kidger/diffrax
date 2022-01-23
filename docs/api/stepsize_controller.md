# Step size controllers

The list of step size controllers is as follows. The most common cases are fixed step sizes with [`diffrax.ConstantStepSize`][] and adaptive step sizes with [`diffrax.PIDController`][].


??? abstract "`diffrax.AbstractStepSizeController`"

    All of the classes implement the following interface specified by [`diffrax.AbstractStepSizeController`][].

    The exact details of this interface are only really useful if you're using the [Manual stepping](../usage/manual-stepping.md) interface; otherwise this is all just internal to the library.

    ::: diffrax.AbstractStepSizeController
        selection:
            members:
                - wrap
                - wrap_solver
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

::: diffrax.PIDController
    selection:
        members:
            - __init__
            - pcoeff
            - icoeff
            - dcoeff
            - rtol
            - atol
            - dtmin
            - dtmax
            - force_dtmin
            - step_ts
            - jump_ts
            - factormin
            - factormax
            - norm
            - safety
