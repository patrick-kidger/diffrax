# Paths

Many objects that arise when solving differential equations are *paths*. That is to say, they are some piecewise continuous function $f : [t_0, t_1] \to \mathbb{R}^d$.

Diffrax represents this general notion via the [`diffrax.AbstractPath`][] class. For example, Brownian motion is a subclass of this. So are the interpolations used to drive neural controlled differential equations.

If you need to create your own path (e.g. to drive a CDE) then you can do so by subclassing [`diffrax.AbstractPath`][].

??? abstract "`diffrax.AbstractPath`"

    ::: diffrax.AbstractPath
        selection:
            members:
                - t0
                - t1
                - evaluate
                - derivative
