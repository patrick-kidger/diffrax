# Delays

Delays allow to model a broader class of differential equations, Delay Differential Equations (DDEs). 
Compared to ODEs, DDEs vector fields need a new argument `history` that integrate delayed states.

At the moment only DDEs with known delays is supported.

!!! example
    The equation's RHS $y'(t) = y(t) + y(t-1) - y(t-2)$ is modelled by
    ```
    def vf(t,y,args,history):
        return y + history[0] - history[1]
    ```

The first element of a `PyTree` of delayed states in a vector field's definition would be `history[0]`. If the state is multidimensional, then `history[0][i]` refers to the $i$th dimension of the first delayed state.
    


::: diffrax.Delays
    selection:
        members:
            - __init__

!!! example
    Pytree of three `Delays.delays` : 
    ``` 
    delays=[lambda t, y, args: 1.0, (lambda t, y ,args: min(t,2), lambda t, y, args : 1/2 * jnp.cos(y))]
    ``` 
 
!!! info
    If `recurrent_checking=True`, then at integration step, a so-called artificial event function $g$ checks for any new discontinuity jumps unconditionally (which are $g_i$'s new odd multiplicity roots). Let $\lambda_i$ be the combined intial discontinuity jumps and the ones found up to the current integration step.

    $\begin{align}
    g_i(t,y(t)) = t - \tau(t,y(t)) - \lambda_i, \quad i = \dots, -2,-1,-,1,2,\dots \\
    \end{align}$

    This can be prohibitely expensive but in some cases can speed up the integration and its accuracy. 
    We refer to this [paper](http://www.cs.toronto.edu/pub/reports/na/hzpEnrightNA09Preprint.pdf) for more information on how to detect new discontinuity jumps in a DDE.

!!! note
    If the history function is continuous and the initial time point induces a discontinuity `max_discontinuities = jnp.array([t0])`.

