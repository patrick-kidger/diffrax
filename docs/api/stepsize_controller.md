# Step size controllers

The list of step size controllers is as follows. The most common cases are fixed step sizes with [`diffrax.ConstantStepSize`][] and adaptive step sizes with [`diffrax.PIDController`][].

!!! warning
        
    When solving SDEs with an adaptive step controller, then three requirements
    have to be fulfilled in order for the solution to be guaranteed to converge to
    the correct result:
    
    - the Brownian motion has to be generated using [`diffrax.VirtualBrownianTree`][],
    - the solver must satisfy certain conditions (in practice all SDE solvers except
    [`diffrax.Euler`][] satisfy these),
    - either
    a) the SDE must have [commutative noise](../usage/how-to-choose-a-solver.md#stochastic-differential-equations)
    OR
    b) the SDE is evaluated at all times at which the Brownian motion (BM) is
    evaluated; since the BM is also evaluated at steps that are rejected by the step
    controller, we must later evaluate the SDE at these times as well 
    (i.e. revisit rejected steps). This can be done using [`diffrax.JumpStepWrapper`].
    
    Note that these conditions are not checked by Diffrax.

    For more details about the convergence of adaptive solutions to SDEs, please refer to
    
    ```bibtex
    @misc{foster2024convergenceadaptiveapproximationsstochastic,
        title={On the convergence of adaptive approximations for stochastic differential equations}, 
        author={James Foster and Andraž Jelinčič},
        year={2024},
        eprint={2311.14201},
        archivePrefix={arXiv},
        primaryClass={math.NA},
        url={https://arxiv.org/abs/2311.14201}, 
    }
    ```


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

::: diffrax.JumpStepWrapper
    selection:
        members:
            - __init__