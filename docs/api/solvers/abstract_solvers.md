# Abstract solvers

All of the solvers (both ODE and SDE solvers) implement the following interface specified by [`diffrax.AbstractSolver`][].

The exact details of this interface are only really useful if you're using the [Manual stepping](../../usage/manual-stepping.md) interface or defining your own solvers; otherwise this is all just internal to the library.

Also see [Extending Diffrax](../../usage/extending.md) for more information on defining your own solvers.

In addition [`diffrax.AbstractSolver`][] has several subclasses that you can use to mark your custom solver as exhibiting particular behaviour.

---

::: diffrax.AbstractSolver
    options:
        members:
            - order
            - strong_order
            - error_order
            - init
            - step
            - func

---

::: diffrax.AbstractImplicitSolver
    options:
        members:
            - does_not_exist

---

::: diffrax.AbstractAdaptiveSolver
    options:
        members:
            - does_not_exist

---

::: diffrax.AbstractItoSolver
    options:
        members:
            - does_not_exist

---

::: diffrax.AbstractStratonovichSolver
    options:
        members:
            - does_not_exist

---

::: diffrax.AbstractWrappedSolver
    options:
        members:
            - does_not_exist

---

### Abstract Runge--Kutta solvers

::: diffrax.AbstractRungeKutta
    options:
        members:
            - does_not_exist

::: diffrax.AbstractERK
    options:
        members:
            - does_not_exist

::: diffrax.AbstractDIRK
    options:
        members:
            - does_not_exist

::: diffrax.AbstractSDIRK
    options:
        members:
            - does_not_exist

::: diffrax.AbstractESDIRK
    options:
        members:
            - does_not_exist

::: diffrax.ButcherTableau
    options:
        members:
            - __init__

::: diffrax.CalculateJacobian
    options:
        members: false

---

### Abstract Stochastic Runge--Kutta (SRK) solvers

::: diffrax.AbstractSRK
    options:
        members:
            - does_not_exist

::: diffrax.StochasticButcherTableau
    options:
        members:
            - __init__

::: diffrax.AbstractFosterLangevinSRK
    options:
        members:
            - does_not_exist
