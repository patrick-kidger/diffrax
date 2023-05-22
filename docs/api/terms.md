# Terms

One of the advanced features of Diffrax is its *term* system. When we write down e.g. a stochastic differential equation

$\mathrm{d}y(t) = f(t, y(t))\mathrm{d}t + g(t, y(t))\mathrm{d}w(t)$

then we have two "terms": a drift and a diffusion. Each of these terms has two parts: a *vector field* ($f$ or $g$) and a *control* ($\mathrm{d}t$ or $\mathrm{d}w(t)$). There is also an implicit assumption about how the vector field and control interact: $f$ and $\mathrm{d}t$ interact as a vector-scalar product. $g$ and $\mathrm{d}w(t)$ interact as a matrix-vector product. (This interaction is always linear.)

"Terms" are thus the building blocks of differential equations.

!!! example

    Consider the ODE $\frac{\mathrm{d}{y}}{\mathrm{d}t} = f(t, y(t))$. Then this has vector field $f$, control $\mathrm{d}t$, and their interaction is a vector-scalar product. This can be described as a single [`diffrax.ODETerm`][].

If multiple terms affect the same evolving state, then they should be grouped into a single [`diffrax.MultiTerm`][].

!!! example

    An SDE would have its drift described by [`diffrax.ODETerm`][] and the diffusion described by a [`diffrax.ControlTerm`][]. As these affect the same evolving state variable, they should be passed to the solver as `MultiTerm(ODETerm(...), ControlTerm(...))`.

If terms affect different pieces of the state, then they should be placed in some PyTree structure. (The exact structure will depend on what the solver accepts.)

!!! example

    Consider the pair of equations (as commonly arising from Hamiltonian systems):

    $\frac{\mathrm{d}x}{\mathrm{d}t}(t) = f(t, y(t)),\qquad\frac{\mathrm{d}y}{\mathrm{d}t}(t) = g(t, x(t))$

    These would be passed to the solver as the 2-tuple of `(ODETerm(...), ODETerm(...))`.

Each solver is capable of handling certain classes of problems, as described by their `solver.term_structure`.

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
