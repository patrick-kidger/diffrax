from collections.abc import Callable
from typing import Any, cast

import equinox as eqx
import equinox.internal as eqxi
import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Int, PyTree, Scalar

from .._custom_types import Aux, Fn, Y
from .._minimise import AbstractMinimiser
from .._misc import cauchy_termination, max_norm, verbose_print
from .._solution import RESULTS


class _OptaxState(eqx.Module, strict=True):
    step: Int[Array, ""]
    f: Scalar
    opt_state: Any
    terminate: Bool[Array, ""]


class OptaxMinimiser(AbstractMinimiser[Y, Aux, _OptaxState], strict=True):
    """A wrapper to use Optax first-order gradient-based optimisers with
    [`optimistix.minimise`][].
    """

    optim: "optax.GradientTransformation"  # pyright: ignore  # noqa: F821
    rtol: float
    atol: float
    norm: Callable[[PyTree], Scalar]
    verbose: frozenset[str]

    def __init__(
        self,
        optim,
        rtol: float,
        atol: float,
        norm: Callable[[PyTree], Scalar] = max_norm,
        verbose: frozenset[str] = frozenset(),
    ):
        """**Arguments:**

        - `optim`: The Optax optimiser to use.
        - `rtol`: Relative tolerance for terminating the solve. Keyword only argument.
        - `atol`: Absolute tolerance for terminating the solve. Keyword only argument.
        - `norm`: The norm used to determine the difference between two iterates in the
            convergence criteria. Should be any function `PyTree -> Scalar`. Optimistix
            includes three built-in norms: [`optimistix.max_norm`][],
            [`optimistix.rms_norm`][], and [`optimistix.two_norm`][]. Keyword only
            argument.
        - `verbose`: Whether to print out extra information about how the solve is
            proceeding. Should be a frozenset of strings, specifying what information to
            print out. Valid entries are `step`, `loss`, `y`. For example
            `verbose=frozenset({"step", "loss"})`.
        """
        # See https://github.com/deepmind/optax/issues/577: Optax has an issue in which
        # it doesn't use pytrees correctly.
        self.optim = eqxi.closure_to_pytree(optim)
        self.rtol = rtol
        self.atol = atol
        self.norm = norm
        self.verbose = verbose

    def init(
        self,
        fn: Fn[Y, Scalar, Aux],
        y: Y,
        args: PyTree,
        options: dict[str, Any],
        f_struct: PyTree[jax.ShapeDtypeStruct],
        aux_struct: PyTree[jax.ShapeDtypeStruct],
        tags: frozenset[object],
    ) -> _OptaxState:
        del fn, args, options, aux_struct
        opt_state = self.optim.init(y)
        maxval = jnp.array(jnp.finfo(f_struct.dtype).max, f_struct.dtype)
        return _OptaxState(
            step=jnp.array(0), f=maxval, opt_state=opt_state, terminate=jnp.array(False)
        )

    def step(
        self,
        fn: Fn[Y, Scalar, Aux],
        y: Y,
        args: PyTree,
        options: dict[str, Any],
        state: _OptaxState,
        tags: frozenset[object],
    ) -> tuple[Y, _OptaxState, Aux]:
        del options
        (f, aux), grads = eqx.filter_value_and_grad(fn, has_aux=True)(y, args)
        f = cast(Array, f)
        if len(self.verbose) > 0:
            verbose_print(
                ("step" in self.verbose, "Step", state.step),
                ("loss" in self.verbose, "Loss", f),
                ("y" in self.verbose, "y", y),
            )
        updates, new_opt_state = self.optim.update(grads, state.opt_state, y)
        new_y = eqx.apply_updates(y, updates)
        terminate = cauchy_termination(
            self.rtol,
            self.atol,
            self.norm,
            y,
            updates,
            f,
            f - state.f,
        )
        new_state = _OptaxState(
            step=state.step + 1, f=f, opt_state=new_opt_state, terminate=terminate
        )
        return new_y, new_state, aux

    def terminate(
        self,
        fn: Fn[Y, Scalar, Aux],
        y: Y,
        args: PyTree,
        options: dict[str, Any],
        state: _OptaxState,
        tags: frozenset[object],
    ) -> tuple[Bool[Array, ""], RESULTS]:
        del fn, args, options
        return state.terminate, RESULTS.successful

    def postprocess(
        self,
        fn: Fn[Y, Scalar, Aux],
        y: Y,
        aux: Aux,
        args: PyTree,
        options: dict[str, Any],
        state: _OptaxState,
        tags: frozenset[object],
        result: RESULTS,
    ) -> tuple[Y, Aux, dict[str, Any]]:
        return y, aux, {}
