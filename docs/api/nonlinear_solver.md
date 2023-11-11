# Root finders

Some differential equation solvers -- in particular implicit solvers -- have to solve an implicit root-finding problem at every step. Such differential equation solvers thus rely on a particular choice of root-finding subroutine, which is passed as a `root_finder` argument.

[Optimistix](https://github.com/patrick-kidger/optimistix) is the JAX library for solving root-finding problems, so all root finders are subclasses of `optimistix.AbstractRootFinder`.
Here's a quick example:

!!! Example

    ```python
    import diffrax as dfx
    import optimistix as optx

    root_finder = optx.Newton(rtol=1e-8, atol=1e-8)
    solver = dfx.Kvaerno5(root_finder=root_finder)
    dfx.diffeqsolve(..., solver, ...)
    ```

In addition to the solvers provided by Optimistix, then Diffrax provides some additional differential-equation-specific functionality, namely [`diffrax.VeryChord`][] and [`diffrax.with_stepsize_controller_tols`][]. The former is a variation of the [chord method](https://docs.kidger.site/optimistix/api/root_find/#optimistix.Chord) that is slightly more efficient for most differential equation solvers. The latter sets the convergence tolerances of the root-finding algorithm to whatever tolerances were used with the adaptive stepsize controller (i.e. `diffeqsolve(..., stepsize_controller=diffrax.PIDController(rtol=..., atol=...))`).

As such the default root-finding algorithm for most solvers in Diffrax is `with_stepsize_controller_tols(VeryChord)()`.

---

::: diffrax.VeryChord
    selection:
        members:
            - __init__

---

::: diffrax.with_stepsize_controller_tols
