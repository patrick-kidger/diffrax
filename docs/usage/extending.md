# Extending Diffrax

It's completely possible to extend Diffrax with your own custom solvers, step size controllers, and so on.

The main points of extension are as follows:

- **Custom solvers** should inherit from [`diffrax.AbstractSolver`][].
    - If you are making a new Runge--Kutta method then this is particularly easy; you can use the existing base classes Diffrax already uses for its own Runge--Kutta methods, and supply them with an appropriate `diffrax.ButcherTableau`.
        - For explicit-Runge--Kutta methods (ERK) then inherit from `diffrax.AbstractERK`.
        - For general diagonal-implicit-Runge--Kutta methods (DIRK) then inherit from `diffrax.AbstractDIRK`.
        - For singly-diagonal-implicit-Runge--Kutta methods (SDIRK) then inherit from `diffrax.AbstractSDIRK`.
        - For explicit-singly-diagonal-implicit-Runge--Kutta methods (ESDIRK) then inherit from `diffrax.AbstractESDIRK`.
    - Several abstract base classes are available to specify the behaviour of the solver:
        - `diffrax.AbstractImplicitSolver` are those solvers that solve implicit problems (and therefore take a `root_finder` argument).
        - `diffrax.AbstractAdaptiveSolver` are those solvers capable of providing error estimates (and thus can be used with adaptive step size controllers).
        - `diffrax.AbstractItoSolver` and `diffrax.AbstractStratonovichSolver` are used to specify which SDE solution a particular solver is known to converge to.
        - `diffrax.AbstractWrappedSolver` indicates that the solver is used to wrap another solver, and so e.g. it will be treated as an implicit solver/etc. if the wrapped solver is also an implicit solver/etc.

- **Custom step size controllers** should inherit from [`diffrax.AbstractStepSizeController`][].
    - The abstract base class `diffrax.AbstractAdaptiveStepSizeController` can be used to specify that this controller uses error estimates to adapt step sizes.

- **Custom Brownian motion simulations** should inherit from [`diffrax.AbstractBrownianPath`][].

- **Custom controls** (e.g. **custom interpolation schemes** analogous to [`diffrax.CubicInterpolation`][]) should inherit from [`diffrax.AbstractPath`][].

- **Custom terms** should inherit from [`diffrax.AbstractTerm`][].
    - For example, if the vector field - control interaction is a matrix-vector product, but the matrix is known to have special structure, then you may wish to create a custom term that can calculate this interaction more efficiently than is given by a full matrix-vector product. For example this is done with [`diffrax.WeaklyDiagonalControlTerm`][] as compared to [`diffrax.ControlTerm`][].

In each case we recommend looking up existing solvers/etc. in Diffrax to understand how to implement them.

!!! tip "Contributions"

    If you implement a technique that you'd like to see merged into the main Diffrax library then open a [pull request on GitHub](https://github.com/patrick-kidger/diffrax). We're very happy to take contributions.
