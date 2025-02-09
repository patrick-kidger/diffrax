# Step size controllers

The list of step size controllers is as follows. The most common cases are fixed step sizes with [`diffrax.ConstantStepSize`][] and adaptive step sizes with [`diffrax.PIDController`][].

?? warning "Adaptive SDEs"
        
    When solving SDEs with an adaptive step controller, then three requirements must be met for the solution to converge to the correct result:
    
    1. the Brownian motion must be generated with [`diffrax.VirtualBrownianTree`][];
    2. the solver must satisfy certain technical conditions (in practice all SDE solvers except [`diffrax.Euler`][] satisfy these),
    3. the SDE must either have [commutative noise](../usage/how-to-choose-a-solver.md#stochastic-differential-equations), or `ClipStepSizeController(..., store_rejected_steps=...)` must be used.
    
    Conditions 1 and 2 are checked by Diffrax. Condition 3 is not (as there is no easy way to verify commutativity of the noise).

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
                - norm

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

::: diffrax.ClipStepSizeController
    selection:
        members:
            - __init__