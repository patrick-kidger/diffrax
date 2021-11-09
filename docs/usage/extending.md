# Extending Diffrax

It's completely possible to extend Diffrax with your own custom solvers, step size controllers, and so on.

The main points of extension are as follows:

- **Custom solvers** should inherit from [`diffrax.AbstractSolver`][].
    - If you are making a new Runge--Kutta method then this is particularly easy; you can use the existing base classes Diffrax already uses for its own Runge--Kutta methods.
    - For explicit-Runge--Kutta methods (ERK) then inherit from `diffrax.AbstractERK`.
    - For general diagonal-implicit-Runge--Kutta methods (DIRK) then inherit from `diffrax.AbstractDIRK`.
    - For singly-diagonal-implicit-Runge--Kutta methods (SDIRK) then inherit from `diffrax.AbstractSDIRK`.
    - For explicit-singly-diagonal-implicit-Runge--Kutta methods (ESDIRK) then inherit from `diffrax.AbstractESDIRK`.
    - (Fully-implicit-Runge--Kutta methods (FIRK) don't have a convenient base class yet.)
    - In each case it is then enough to simply set the right tableau, interpolation scheme and so on. Look up the code for the existing solvers and copy what they do.

- **Custom step size controllers** should inherit from [`diffrax.AbstractStepSizeController`][].

- **Custom Brownian motion simulations** should inherit from [`diffrax.AbstractBrownianPath`][].

- **Custom controls** (e.g. **custom interpolation schemes** analogous to [`diffrax.CubicInterpolation`][]) should inherit from [`diffrax.AbstractPath`][].

- **Custom nonlinear solvers** (used in implicit methods) should inherit from [`diffrax.AbstractNonlinearSolver`][].

- **Custom terms** should inherit from [`diffrax.AbstractTerm`][].
    - For example, if the vector field - control interaction is a matrix-vector product, but the matrix is known to have special structure, then you may wish to create a custom term that can calculate this interaction more efficiently than is given by a full matrix-vector product.

In each case we recommend looking up existing solvers/etc. in Diffrax to understand how to implement them.

!!! tip "Contributions"

    If you implement a technique that you'd like to see merged into the main Diffrax library then open a [pull request on GitHub](https://github.com/patrick-kidger/diffrax). We're very happy to take contributions.

!!! warning

    In practice the APIs provided by these abstract base classes is now pretty stable. In principle however they may still be subject to change.
