# Solvers

The complete list of solvers, categorised by type, is as follows.

!!! info "Term structure"

    The type of solver chosen determines how the `terms` argument of `diffeqsolve` should be laid out. Most of them demand that it should be a single `AbstractTerm`. But for example [`diffrax.SemiImplicitEuler`][] demands that it be a 2-tuple `(AbstractTerm, AbstractTerm)`, to represent the two vector fields that solver uses.

    If it is different from this default, then you can find the appropriate structure documented below, and available programmatically under `<solver>.term_structure`.

!!! info "Stochastic differential equations"

    Little distinction is made between solvers for different kinds of differential equation, like between ODEs and SDEs. Diffrax's term system allows for treating them all in a unified way.

    Those ODE solvers that make sense as SDE solvers are documented as such below. For the common case of an SDE with drift and Brownian-motion-driven diffusion, they can be used by combining drift and diffusion into a single term:

    ```python
    drift = lambda t, y, args: -y
    diffusion = lambda t, y, args: y[..., None]
    bm = UnsafeBrownianPath(shape=(1,), key=...)
    terms = MultiTerm(ODETerm(drift), ControlTerm(diffusion, bm))
    diffeqsolve(terms, solver=Euler(), ...)
    ```

    In addition there are some [SDE-specific solvers](#sde-only-solvers).


??? abstract "`diffrax.AbstractSolver`"

    All of the classes implement the following interface specified by [`diffrax.AbstractSolver`][].

    The exact details of this interface are only really useful if you're using the [Manual stepping](../usage/manual-stepping.md) interface or defining your own solvers; otherwise this is all just internal to the library.

    ::: diffrax.AbstractSolver
        selection:
            members:
                - order
                - strong_order
                - error_order
                - term_structure
                - init
                - step
                - func_for_init

---

### Explicit Runge--Kutta (ERK) methods

::: diffrax.Euler
    selection:
        members: false

::: diffrax.Heun
    selection:
        members: false

::: diffrax.Midpoint
    selection:
        members: false

::: diffrax.Ralston
    selection:
        members: false

::: diffrax.Fehlberg2
    selection:
        members: false

::: diffrax.Bosh3
    selection:
        members: false

::: diffrax.Tsit5
    selection:
        members: false

::: diffrax.Dopri5
    selection:
        members: false

::: diffrax.Dopri8
    selection:
        members: false

---

### Implicit Runge--Kutta (IRK) methods

::: diffrax.ImplicitEuler
    selection:
        members: false

::: diffrax.Kvaerno3
    selection:
        members: false

::: diffrax.Kvaerno4
    selection:
        members: false

::: diffrax.Kvaerno5
    selection:
        members: false

---

### Symplectic methods

??? info "Term and state structure"

    The state of the system (the initial value of which is given by `y0` to [`diffrax.diffeqsolve`][]) must be a 2-tuple (of PyTrees). The terms (given by the value of `terms` to [`diffrax.diffeqsolve`][]) must be a 2-tuple of `AbstractTerms`.
    
    Letting `v, w = y0` and `f, g = terms`, then `v` is updated according to
    `f(t, w, args) * dt` and `w` is updated according to `g(t, v, args) * dt`.

::: diffrax.SemiImplicitEuler
    selection:
        members: false

---

### Reversible methods

::: diffrax.ReversibleHeun
    selection:
        members: false

---

### Linear multistep methods

::: diffrax.LeapfrogMidpoint
    selection:
        members: false

---

### SDE-only solvers

!!! tip "Other SDE solvers"

    Many low-order ODE solvers can also be used as SDE solvers:

    **Itô:**

    - [`diffrax.Euler`][]

    **Stratonovich:**

    - [`diffrax.Heun`][]
    - [`diffrax.Midpoint`][]
    - [`diffrax.ReversibleHeun`][]

!!! info "Term structure"

    For these SDE-specific solvers, the terms (given by the value of `terms` to [`diffrax.diffeqsolve`][]) must be a 2-tuple `(AbstractTerm, AbstractTerm)`, representing the drift and diffusion respectively. Typically that means `(ODETerm(...), ControlTerm(..., ...))`.

::: diffrax.EulerHeun
    selection:
        members: false

::: diffrax.ItoMilstein
    selection:
        members: false

::: diffrax.StratonovichMilstein
    selection:
        members: false

---

### Wrapper solvers

::: diffrax.HalfSolver
    selection:
        members: false
