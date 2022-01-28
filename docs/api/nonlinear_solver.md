# Nonlinear solvers

Some differential equation solvers -- in particular implicit solvers -- have to solve an implicit nonlinear problem at every step. Such differential equation solvers take an instance of a nonlinear solver as an argument.

??? abstract "`diffrax.AbstractNonlinearSolver`"

    ::: diffrax.AbstractNonlinearSolver
        selection:
            members:
                - __call__
                - jac

---

::: diffrax.NewtonNonlinearSolver
    selection:
        members:
            - __init__
            - __call__
            - jac
