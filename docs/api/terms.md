# Terms

One of the really neat things about Diffrax is its ability to solve ODEs, SDEs and CDEs all in the same way, using the same solvers. This simplifies the internals of the library a lot. (It also means that, if you really wanted to, you can do advanced things like solve an ODE using one numerical method whilst simultaneously solving an SDE using some other numerical method.)

Diffrax makes this happen through the use of its *term* system. When we write down e.g. a stochastic differential equation

$\mathrm{d}y(t) = f(t, y(t))\mathrm{d}t + g(t, y(t))\mathrm{d}w(t)$

then we have two *terms*: a drift and a diffusion. Each of these terms has two parts: a *vector field* ($f$ or $g$) and a *control* ($\mathrm{d}t$ or $\mathrm{d}w(t)$). There is also an implicit assumption about how vector field and control interact: $f$ and $\mathrm{d}t$ interact as a vector-scalar product. $g$ and $\mathrm{d}w(t)$ interact as a matrix-vector product.

Diffrax offers a way to represent all of the above, and all of the solvers accept terms as arguments.

!!! note
    Representing differential equations like this is new in Diffrax [computationally speaking, that is -- the mathematics has been around for a while]. The explanations given in the following documentation are therefore relatively untested -- let us know if you find them unclear!

---

Every term implements the following interface:

??? "`diffrax.AbstractTerm`"

    ::: diffrax.AbstractTerm
        selection:
            members:
                - vf
                - contr
                - prod
                - vf_prod
                - func_for_init

---

Several common cases are already specified.

!!! note
    You can create your own terms if appropriate: e.g. if a diffusion matrix has some particular structure, and you want to use a specialised more efficient matrix-vector product algorithm in `prod`.

::: diffrax.ODETerm
    selection:
        members: false

::: diffrax.ControlTerm
    selection:
        members:
            - to_ode

::: diffrax.MultiTerm
    selection:
        members: false

