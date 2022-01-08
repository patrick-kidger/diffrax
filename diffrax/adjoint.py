import equinox as eqx


class AbstractAdjoint(eqx.Module):
    """Abstract base class for all adjoint methods."""


class RecursiveCheckpointAdjoint(AbstractAdjoint):
    """Backpropagate through [`diffrax.diffeqsolve`][] by differentiating the numerical
    solution directly. This is sometimes known as "discretise-then-optimise", or
    described as "backpropagation through the solver".

    For most problems this is the preferred technique for backpropagating through a
    differential equation.

    A binomial checkpointing scheme is used so that memory usage is low.
    """


class BacksolveAdjoint(AbstractAdjoint):
    """Backpropagate through [`diffrax.diffeqsolve`][] by solving the continuous
    adjoint equations backwards-in-time. This is also sometimes known as
    "optimise-then-discretise", the "continuous adjoint method" or simply the "adjoint
    method".

    This method implies very low memory usage, but is usually relatively slow, and the
    computed gradients will only be approximate. As such other methods are generally
    preferred unless memory pressure is a concern.

    !!! note

        This was popularised by [this paper](https://arxiv.org/abs/1806.07366). For
        this reason it is sometimes erroneously believed to be a better method for
        backpropagation than the other choices available.

    !!! warning

        Note that using this method prevents computing forward-mode autoderivatives of
        [`diffrax.diffeqsolve`][]. (That is to say, `jax.jvp` will not work.)
    """


class NoAdjoint(AbstractAdjoint):
    """Disable backpropagation through [`diffrax.diffeqsolve`][].

    Forward-mode autodifferentiation (`jax.jvp`) will continue to work as normal.

    If you do not need to differentiate the results of [`diffrax.diffeqsolve`][] then
    this may sometimes improve the speed at which the differential equation is solved.
    """
