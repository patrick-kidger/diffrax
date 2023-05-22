# SDE solvers

See also [How to choose a solver](../../usage/how-to-choose-a-solver.md#stochastic-differential-equations).

!!! info "Term structure"

    The type of solver chosen determines how the `terms` argument of `diffeqsolve` should be laid out. Most of them operate in the same way whether they are solving an ODE or an SDE, and as such expected that it should be a single `AbstractTerm`. For SDEs that typically means a [`diffrax.MultiTerm`][] wrapping together a drift ([`diffrax.ODETerm`][]) and diffusion ([`diffrax.ControlTerm`][]). (Although you could also include any other term, e.g. an exogenous forcing term, if you wished.) For example:

    ```python
    drift = lambda t, y, args: -y
    diffusion = lambda t, y, args: y[..., None]
    bm = UnsafeBrownianPath(shape=(1,), key=...)
    terms = MultiTerm(ODETerm(drift), ControlTerm(diffusion, bm))
    diffeqsolve(terms, solver=Euler(), ...)
    ```

    Some solvers are SDE-specific. For these, such as for example [`diffrax.StratonovichMilstein`][], then `terms` must specifically be of the form `MultiTerm(ODETerm(...), SomeOtherTerm(...))` (Typically `SomeOTherTerm` will be a `ControlTerm` or `WeaklyDiagonalControlTerm`) representing the drift and diffusion specifically.

    For those SDE-specific solvers then this is documented below, and the term structure is available programmatically under `<solver>.term_structure`.

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

!!! info

    In addition to the solvers above, then most higher-order ODE solvers can actually also be used as SDE solvers. They will typically converge to the Stratonovich solution. In practice this is computationally wasteful as they will not obtain more accurate solutions when applied to SDEs.

---

### Reversible methods

These are reversible in the same way as when applied to ODEs. [See here.](./ode_solvers.md#reversible-methods)

::: diffrax.ReversibleHeun
    selection:
        members: false

---

### SDE-only solvers

!!! info "Term structure"

    These solvers are SDE-specific. For these, `terms` must specifically be of the form `MultiTerm(ODETerm(...), SomeOtherTerm(...))` (Typically `SomeOTherTerm` will be a `ControlTerm` or `WeaklyDiagonalControlTerm`) representing the drift and diffusion specifically.


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
        members:
            - __init__
