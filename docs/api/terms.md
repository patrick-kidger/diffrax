# Terms

One of the advanced features of Diffrax is its *term* system. When we write down e.g. a stochastic differential equation

$\mathrm{d}y(t) = f(t, y(t))\mathrm{d}t + g(t, y(t))\mathrm{d}w(t)$

then we have two "terms": a drift and a diffusion. Each of these terms has two parts: a *vector field* ($f$ or $g$) and a *control* ($\mathrm{d}t$ or $\mathrm{d}w(t)$). In addition (often not represented in mathematical notation), there is also a choice of how the vector field and control interact: $f$ and $\mathrm{d}t$ interact as a vector-scalar product. $g$ and $\mathrm{d}w(t)$ interact as a matrix-vector product. (In general this interaction is always bilinear.)

"Terms" are thus the building blocks of differential equations.

!!! example

    Consider the ODE $\frac{\mathrm{d}{y}}{\mathrm{d}t} = f(t, y(t))$. Then this has vector field $f$, control $\mathrm{d}t$, and their interaction is a vector-scalar product. This can be described as a single [`diffrax.ODETerm`][].

#### Adding multiple terms, such as SDEs

We can add multiple terms together by grouping them into a single [`diffrax.MultiTerm`][].

!!! example

    The SDE above would have its drift described by [`diffrax.ODETerm`][] and the diffusion described by a [`diffrax.ControlTerm`][]. As these affect the same evolving state variable, they should be passed to the solver as `MultiTerm(ODETerm(...), ControlTerm(...))`.

#### Independent terms, such as Hamiltonian systems

If terms affect different pieces of the state, then they should be placed in some PyTree structure.

!!! example

    Consider the pair of equations (as commonly arising from Hamiltonian systems):

    $\frac{\mathrm{d}x}{\mathrm{d}t}(t) = f(t, y(t)),\qquad\frac{\mathrm{d}y}{\mathrm{d}t}(t) = g(t, x(t))$

    These would be passed to the solver as the 2-tuple of `(ODETerm(...), ODETerm(...))`.

#### What each solver accepts

Each solver in Diffrax will specify what kinds of problems it can handle, as described by their `.term_structure` attribute. Not all solvers are able to handle all problems!

Some example term structures include:

1. `solver.term_structure = AbstractTerm`

    In this case the solver can handle a simple ODE as descibed above: `ODETerm` is a subclass of `AbstractTerm`.

    It can also handle SDEs: `MultiTerm(ODETerm(...), ControlTerm(...))` includes everything wrapped into a single term (the `MultiTerm`), and at that point this defines an interface the solver knows how to handle.

    Most solvers in Diffrax have this term structure.

2. `solver.term_structure = MultiTerm[tuple[ODETerm, ControlTerm]]`

    In this case the solver specifically handles just SDEs of the form `MultiTerm(ODETerm(...), ControlTerm(...))`; nothing else is compatible.

    Some SDE-specific solvers have this term structure.

3. `solver.term_structure = (AbstractTerm, AbstractTerm)`

    In this case the solver is used to solve ODEs like the Hamiltonian system described above: we have a PyTree of terms, each of which is treated individually.

---

??? abstract "`diffrax.AbstractTerm`"

    ::: diffrax.AbstractTerm
        selection:
            members:
                - vf
                - contr
                - prod
                - vf_prod
                - is_vf_expensive

??? note "Defining your own term types"

    For advanced users: you can create your own terms if appropriate. For example if your diffusion is matrix, itself computed as a matrix-matrix product, then you may wish to define a custom term and specify its [`diffrax.AbstractTerm.vf_prod`][] method. By overriding this method you could express the contraction of the vector field - control as a matrix-(matix-vector) product, which is more efficient than the default (matrix-matrix)-vector product.


---

::: diffrax.ODETerm
    selection:
        members:
            - __init__

::: diffrax.ControlTerm
    selection:
        members:
            - __init__
            - to_ode

::: diffrax.MultiTerm
    selection:
        members:
            - __init__


---

#### Underdamped Langevin terms

These are special terms which describe the Underdamped Langevin diffusion (ULD),
which takes the form 

\begin{align*}
    \mathrm{d} x(t) &= v(t) \, \mathrm{d}t \\
    \mathrm{d} v(t) &= - \gamma \, v(t) \, \mathrm{d}t - u \,
    \nabla \! f( x(t) ) \, \mathrm{d}t + \sqrt{2 \gamma u} \, \mathrm{d} w(t),
\end{align*}

where $x(t), v(t) \in \mathbb{R}^d$ represent the position
and velocity, $w$ is a Brownian motion in $\mathbb{R}^d$,
$f: \mathbb{R}^d \rightarrow \mathbb{R}$ is a potential function, and
$\gamma , u \in \mathbb{R}^{d \times d}$ are diagonal matrices governing
the friction and the damping of the system.

These terms enable the use of ULD-specific solvers which can be found 
[here](./solvers/sde_solvers.md#underdamped-langevin-solvers). Note that these ULD solvers will only work if given
terms with structure `MultiTerm(UnderdampedLangevinDriftTerm(gamma, u, grad_f), UnderdampedLangevinDiffusionTerm(gamma, u, bm))`,
where `bm` is an [`diffrax.AbstractBrownianPath`][] and the same values of `gammma` and `u` are passed to both terms.

::: diffrax.UnderdampedLangevinDriftTerm
    selection:
        members:
            - __init__

::: diffrax.UnderdampedLangevinDiffusionTerm
    selection:
        members:
            - __init__