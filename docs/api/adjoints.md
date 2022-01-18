# Adjoints

There are multiple ways to backpropagate through a differential equation (to compute the gradient of the solution with respect to its initial condition and any parameters).

!!! info

    Why are there multiple ways of backpropagating through a differential equation? Suppose we are given an ODE

    $\frac{\mathrm{d}y}{\mathrm{d}t} = f(t, y(t))$

    on $[t_0, t_1]$, with initial condition $y(0) = y_0$. So $y(t)$ is the (unknown) exact solution, to which we will compute some numerical approxiation $y_N \approx y(t_1)$.

    We may directly apply autodifferentiation to calculate $\frac{\mathrm{d}y_N}{\mathrm{d}y_0}$, by backpropagating through the internals of the solver. This is known a "discretise then optimise", is the default in Diffrax, and corresponds to [`diffrax.RecursiveCheckpointAdjoint`][] below.

    Alternatively we may compute $\frac{\mathrm{d}y(t_1)}{\mathrm{d}y_0}$ analytically -- in doing so we obtain a backwards-in-time ODE that we must numerically solve to obtain the desired gradients. This is known as "optimise then discretise", and corresponds to [`diffrax.BacksolveAdjoint`][] below.

??? abstract "`diffrax.AbstractSolver`"

    ::: diffrax.AbstractAdjoint
        selection:
            members:
                - loop

---

::: diffrax.RecursiveCheckpointAdjoint
    selection:
        members: false

::: diffrax.NoAdjoint
    selection:
        members: false

::: diffrax.BacksolveAdjoint
    selection:
        members:
            - __init__

!!! tip

    To use adjoint seminorms as described in ["Hey, that's not an ODE": Faster ODE Adjoints via Seminorms](https://arxiv.org/abs/2009.09457), then this can be accomplished by simply passing the appropriate norm for the reverse pass:
    ```python
    adjoint_stepsize_controller = diffrax.IController(norm=diffrax.adjoint_rms_seminorm)
    adjoint = diffrax.BacksolveAdjoint(stepsize_controller=adjoint_stepsize_controller)
    diffrax.diffeqsolve(..., adjoint=adjoint)
    ```
    Note that this will mean that any `stepsize_controller` specified for the forward pass will not be automatically used for the backward pass (as `adjoint_stepsize_controller` overrides it), so you should specify any custom `rtol`, `atol` for the backward pass as well.
