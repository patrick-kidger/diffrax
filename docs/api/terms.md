# Terms

One of the advanced features of Diffrax is its *term* system. When we write down e.g. a stochastic differential equation

$\mathrm{d}y(t) = f(t, y(t))\mathrm{d}t + g(t, y(t))\mathrm{d}w(t)$

then we have two "terms": a drift and a diffusion. Each of these terms has two parts: a *vector field* ($f$ or $g$) and a *control* ($\mathrm{d}t$ or $\mathrm{d}w(t)$). There is also an implicit assumption about how vector field and control interact: $f$ and $\mathrm{d}t$ interact as a vector-scalar product. $g$ and $\mathrm{d}w(t)$ interact as a matrix-vector product. (This interaction is always linear.)

"Terms" are thus the building blocks of differential equations. In Diffrax, the above SDE has its drift described by [`diffrax.ODETerm`][] and the diffusion described by a [`diffrax.ControlTerm`][].

!!! example

    As a simpler example, consider the ODE $\frac{\mathrm{d}{y}}{\mathrm{d}t} = f(t, y(t))$. Then this has vector field $f$, control $\mathrm{d}t$, and their interaction is a vector-scalar product. This can be described as a single [`diffrax.ODETerm`][].

!!! example

    Consider the pair of equations (as commonly arising from Hamiltonian systems):

    $\frac{\mathrm{d}x}{\mathrm{d}t}(t) = f(t, y(t)),\qquad\frac{\mathrm{d}y}{\mathrm{d}t}(t) = g(t, x(t))$

    These can be described as a 2-tuple of [`diffrax.ODETerm`][]`s.

The very first argument to [`diffrax.diffeqsolve`][] should be some PyTree of terms. This is interpreted by the solver in the appropriate way.

- For example [`diffrax.Euler`][] expects a single term: it solves an ODE represented via `ODETerm(...)`, or an SDE represented via `MultiTerm(ODETerm(...), ControlTerm(...))`.
- Meanwhile [`diffrax.SemiImplicitEuler`][] solves the paired (Hamiltonian) system given in the example above, and expects a 2-tuple of terms representing each piece.
- Some SDE-specific solvers (e.g. [`diffrax.StratonovichMilstein`][] need to be able to see the distinction between the drift and diffusion, and expect a 2-tuple of terms representing the drift and diffusion respectively.

??? abstract "`diffrax.AbstractTerm`"

    ::: diffrax.AbstractTerm
        selection:
            members:
                - vf
                - contr
                - prod
                - vf_prod
                - is_vf_expensive

---

!!! note
    You can create your own terms if appropriate: e.g. if a diffusion matrix has some particular structure, and you want to use a specialised more efficient matrix-vector product algorithm in `prod`. For example this is what [`diffrax.WeaklyDiagonalControlTerm`][] does, as compared to just [`diffrax.ControlTerm`][].

::: diffrax.ODETerm
    selection:
        members:
            - __init__

::: diffrax.ControlTerm
    selection:
        members:
            - __init__
            - to_ode

::: diffrax.WeaklyDiagonalControlTerm
    selection:
        members:
            - __init__
            - to_ode

::: diffrax.MultiTerm
    selection:
        members:
            - __init__
