# Solvers

## Solver classes

The complete list of solvers, categorised by type, is as follows.

!!! note

    The type of solver chosen determines how the `terms` field of `diffeqsolve` should be laid out. Most of them demand that it should be a single `AbstractTerm`. But for example [`diffrax.SemiImplicitEuler`][] demands that it by a 2-tuple `(AbstractTerm, AbstractTerm)`, to represent the two vector fields that solver uses.

    If it is different from this default, then you can find the appropriate structure under `<solver>.term_structure`.

??? info "Stochastic differential equations"

    No distinction is made between solvers for different kinds of differential equation, like between ODEs and SDEs. Diffrax's term system allows for treating them all in a unified way.

    For the common case of an SDE with drift and Brownian-motion-driven diffusion, they can be used as

    ```python
    drift = lambda t, y, args: -y
    diffusion = lambda t, y, args: y[..., None]
    bm = UnsafeBrownianPath(shape=(1,), key=...)
    terms = MultiTerm(ODETerm(drift), ControlTerm(diffusion, bm))
    diffeqsolve(terms, ..., solver=Euler())
    ```

    In which the various terms are combined together into a single term, via [`diffrax.MultiTerm`][].

    As a general rule, any first or second order ODE solver may be used to solve an SDE. (Higher solvers will work perfectly fine, but won't produce more accurate results, so their extra computational work is unnecessary.)


??? abstract "`diffrax.AbstractSolver`"

    All of the classes implement the following interface specified by [`diffrax.AbstractSolver`][].

    The exact details of this interface are only really useful if you're using the [Manual stepping](../usage/manual-stepping.md) interface; otherwise this is all just internal to the library.

    ::: diffrax.AbstractSolver
        selection:
            members:
                - order
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
