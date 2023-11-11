# ODE solvers

See also [How to choose a solver](../../usage/how-to-choose-a-solver.md#ordinary-differential-equations).

!!! info "Term structure"

    The type of solver chosen determines how the `terms` argument of `diffeqsolve` should be laid out. Most of them demand that it should be a single `AbstractTerm`. But for example [`diffrax.SemiImplicitEuler`][] demands that it be a 2-tuple `(AbstractTerm, AbstractTerm)`, to represent the two vector fields that solver uses.

    If it is different from this default, then you can find the appropriate structure documented below, and available programmatically under `<solver>.term_structure`.

---

### Explicit Runge--Kutta (ERK) methods

These methods are suitable for most problems.

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

These methods are suitable for stiff problems.

Each of these takes a `root_finder` argument at initialisation, defaulting to a Newton solver, which is used to solve the implicit problem at each step. See the page on [root finders](../nonlinear_solver.md).

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

### IMEX methods

These "implicit-explicit" methods are suitable for problems of the form $\frac{\mathrm{d}y}{\mathrm{d}t} = f(t, y(t)) + g(t, y(t))$, where $f$ is the non-stiff part (explicit integration) and $g$ is the stiff part (implicit integration).

??? info "Term structure"

    These methods should be called with `terms=MultiTerm(explicit_term, implicit_term)`.

::: diffrax.Sil3
    selection:
        members: false

::: diffrax.KenCarp3
    selection:
        members: false

::: diffrax.KenCarp4
    selection:
        members: false

::: diffrax.KenCarp5
    selection:
        members: false

---

### Symplectic methods

These methods are suitable for problems with symplectic structure; that is to say those ODEs of the form

$\frac{\mathrm{d}v}{\mathrm{d}t}(t) = f(t, w(t))$

$\frac{\mathrm{d}w}{\mathrm{d}t}(t) = g(t, v(t))$

In particular this includes Hamiltonian systems.

??? info "Term and state structure"

    The state of the system (the initial value of which is given by `y0` to [`diffrax.diffeqsolve`][]) must be a 2-tuple (of PyTrees). The terms (given by the value of `terms` to [`diffrax.diffeqsolve`][]) must be a 2-tuple of `AbstractTerms`.

    Letting `v, w = y0` and `f, g = terms`, then `v` is updated according to `f(t, w, args)` and `w` is updated according to `g(t, v, args)`.

    See also this [Wikipedia page](https://en.wikipedia.org/wiki/Semi-implicit_Euler_method#Setting).

::: diffrax.SemiImplicitEuler
    selection:
        members: false

---

### Reversible methods

These methods can be run "in reverse": solving from an initial condition `y0` to obtain some terminal value `y1`, it is possible to reconstruct `y0` from `y1` with zero truncation error. (There will still be a small amount of floating point error.) This can be done via `SaveAt(solver_state=True)` to save the final solver state, and then passing it as `diffeqsolve(..., solver_state=solver_state)` on the backwards-in-time pass.

In addition all [symplectic methods](#symplectic-methods) are reversible, as are some linear multistep methods. (Below are the non-symplectic reversible solvers.)

::: diffrax.ReversibleHeun
    selection:
        members: false

---

### Linear multistep methods

::: diffrax.LeapfrogMidpoint
    selection:
        members: false
