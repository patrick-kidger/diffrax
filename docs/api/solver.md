# Solvers

## Solver classes

All of the classes implement the following interface specified by [`diffrax.AbstractSolver`][].

(The exact details of this interface are only really useful if you're using the [Manual stepping](../usage/manual-stepping.md) interface; otherwise this is all just internal to the library.)

??? "`diffrax.AbstractSolver`"

    ::: diffrax.AbstractSolver
        selection:
            members:
                - order
                - wrap
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

## Convenience wrappers

The following are convenience wrappers for the above solvers, for the common case of solving single-term ODEs and SDEs.

---

### Ordinary differential equations

!!! example
    All of the following are used as:
    ```python
    vector_field = lambda t, y, args: -y
    solver = euler(vector_field)
    ```

    This is equivalent to:
    ```python
    vector_field = lambda t, y, args: -y
    ode_term = ODETerm(vector_field)
    solver = Euler(ode_term)
    ```

::: diffrax.euler

::: diffrax.implicit_euler

::: diffrax.heun

::: diffrax.fehlberg2

::: diffrax.bosh3

::: diffrax.kvaerno3

::: diffrax.kvaerno4

::: diffrax.kvaerno5

::: diffrax.tsit5

::: diffrax.dopri5

::: diffrax.dopri8

---

### Stochastic differential equations

!!! note
    Generally speaking any first or second order ODE solver can be used as an SDE
    solver. Sometimes they then go by slightly different names -- for example Euler
    becomes Euler--Maruyama.

!!! example
    All of the following are used as:
    ```python
    drift = lambda t, y, args: -y
    diffusion = lambda t, y, args: y[..., None]  # 1-dimensional Brownian motion
    bm = UnsafeBrownianPath(shape=(1,), key=...)
    solver = euler_maruyama(drift, diffusion, bm)
    ```

    This is equivalent to:
    ```python
    drift = lambda t, y, args: -y
    diffusion = lambda t, y, args: y[..., None]
    bm = UnsafeBrownianPath(shape=(1,), key=...)
    drift_term = ODETerm(drift)
    diffusion_term = ControlTerm(diffusion, bm)
    multi_term = MultiTerm((drift_term, diffusion_term))
    solver = Euler(multi_term)
    ```

::: diffrax.euler_maruyama

::: diffrax.implicit_euler_maruyama

::: diffrax.heun
