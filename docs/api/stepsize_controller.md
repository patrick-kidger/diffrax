# Step size controllers

The list of step size controllers is as follows. The most common cases are fixed step sizes with [`diffrax.ConstantStepSize`][] and adaptive step sizes with [`diffrax.PIDController`][].

!!! warning

    To perform adaptive stepping with SDEs requires [commutative noise](../usage/how-to-choose-a-solver.md#stochastic-differential-equations). Note that this commutativity condition is not checked.


??? abstract "Abtract base classes"

    All of the classes implement the following interface specified by [`diffrax.AbstractStepSizeController`][].

    The exact details of this interface are only really useful if you're using the [Manual stepping](../usage/manual-stepping.md) interface; otherwise this is all just internal to the library.

    ::: diffrax.AbstractStepSizeController
        selection:
            members:
                - wrap
                - init
                - adapt_step_size

    ::: diffrax.AbstractAdaptiveStepSizeController
        selection:
            members:
                - rtol
                - atol

---

::: diffrax.ConstantStepSize
    selection:
        members: false

::: diffrax.StepTo
    selection:
        members:
            - __init__

::: diffrax.PIDController
    selection:
        members:
            - __init__
