# Abstract solvers

All of the solvers (both ODE and SDE solvers) implement the following interface specified by [`diffrax.AbstractSolver`][].

The exact details of this interface are only really useful if you're using the [Manual stepping](../../usage/manual-stepping.md) interface or defining your own solvers; otherwise this is all just internal to the library.

Also see [Extending Diffrax](../../usage/extending.md) for more information on defining your own solvers.

In addition [`diffrax.AbstractSolver`][] has several subclasses that you can use to mark your custom solver as exhibiting particular behaviour.

---

::: diffrax.AbstractSolver
    selection:
        members:
            - order
            - strong_order
            - error_order
            - init
            - step
            - func

---

::: diffrax.AbstractImplicitSolver
    selection:
        members:
          - __init__

---

::: diffrax.AbstractAdaptiveSolver
    selection:
        members: false

---

::: diffrax.AbstractItoSolver
    selection:
        members: false

---

::: diffrax.AbstractStratonovichSolver
    selection:
        members: false

---

::: diffrax.AbstractWrappedSolver
    selection:
        members:
            - __init__

---

### Abstract Runge--Kutta solvers

::: diffrax.AbstractRungeKutta
    selection:
        members: false

::: diffrax.AbstractERK
    selection:
        members: false

::: diffrax.AbstractDIRK
    selection:
        members: false

::: diffrax.AbstractSDIRK
    selection:
        members: false

::: diffrax.AbstractESDIRK
    selection:
        members: false

::: diffrax.ButcherTableau
    selection:
        members:
            - __init__

::: diffrax.CalculateJacobian
    selection:
        members: false
