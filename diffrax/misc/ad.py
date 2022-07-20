import functools as ft
from typing import Any

import equinox as eqx
import jax
import jax.flatten_util as fu
import jax.interpreters.ad as ad
import jax.interpreters.batching as batching
import jax.interpreters.mlir as mlir
import jax.interpreters.xla as xla
import jax.lax as lax
import jax.numpy as jnp

from ..custom_types import PyTree


# TODO: this will sometimes return False on a perturbed array, see JAX issue #9567.
# Correspondingly it should *not be used* until that is fixed.
# (The only use is in nondifferentiable_input, below, which will simply not raise
# errors quite as frequently as it should do -- not too bad.)
def is_perturbed(x: Any) -> bool:
    if isinstance(x, jax.ad.JVPTracer):
        return True
    elif isinstance(x, jax.core.Tracer):
        return any(is_perturbed(attr) for name, attr in x._contents())
    else:
        return False


def nondifferentiable_input(x: PyTree, name: str) -> None:
    if any(is_perturbed(xi) for xi in jax.tree_leaves(x)):
        raise ValueError(f"Cannot differentiate {name}.")


_nondifferentiable_output_p = jax.core.Primitive("nondifferentiable_output")


def _nondifferentiable_output_batch(x, batch_axes):
    (x,) = x
    (batch_axes,) = batch_axes
    return nondifferentiable_output(x), batch_axes


def _nondifferentiable_output_jvp(primals, tangents):
    (primals,) = primals
    (tangents,) = tangents
    return nondifferentiable_output(primals), nondifferentiable_output(tangents)


def _nondifferentiable_output_transpose(cts_in, _):
    if isinstance(cts_in, ad.Zero):
        return ad.Zero  # the class, not an instance
    else:
        raise RuntimeError(
            "Reverse-mode autodifferentiation is disabled for this operation."
        )


_nondifferentiable_output_p.def_impl(lambda x: x)
_nondifferentiable_output_p.def_abstract_eval(lambda x: x)
batching.primitive_batchers[
    _nondifferentiable_output_p
] = _nondifferentiable_output_batch
if hasattr(xla, "lower_fun"):
    xla.register_translation(
        _nondifferentiable_output_p,
        xla.lower_fun(lambda x: x, multiple_results=False, new_style=True),
    )
mlir.register_lowering(
    _nondifferentiable_output_p,
    mlir.lower_fun(lambda x: x, multiple_results=False),
)
ad.primitive_jvps[_nondifferentiable_output_p] = _nondifferentiable_output_jvp
ad.primitive_transposes[
    _nondifferentiable_output_p
] = _nondifferentiable_output_transpose


def nondifferentiable_output(x: PyTree) -> PyTree:
    return _nondifferentiable_output_p.bind(x)


class fixed_custom_jvp:
    """As jax.custom_jvp but works around JAX issue #9374."""

    def __init__(self, fn, nondiff_argnums=()):
        assert set(nondiff_argnums) == set(range(len(nondiff_argnums)))

        def fn_wrapper(nondiff_args_nontracer, nondiff_args_tracer, diff_args):
            nondiff_args = eqx.combine(nondiff_args_nontracer, nondiff_args_tracer)
            return fn(*nondiff_args, *diff_args)

        self.fn = jax.custom_jvp(fn_wrapper, nondiff_argnums=(0,))
        self.cutoff = max(nondiff_argnums, default=-1) + 1
        self.fn_jvp = None

    def defjvp(self, fn_jvp):
        def fn_jvp_wrapper(nondiff_args_nontracer, combined_args, tang_combined_args):
            nondiff_args_tracer, diff_args = combined_args
            _, tang_diff_args = tang_combined_args
            nondiff_args = eqx.combine(nondiff_args_nontracer, nondiff_args_tracer)
            return fn_jvp(*nondiff_args, diff_args, tang_diff_args)

        self.fn.defjvp(fn_jvp_wrapper)

    def __call__(self, *args):
        is_tracer = lambda x: isinstance(x, jax.core.Tracer)
        nondiff_args = args[: self.cutoff]
        diff_args = args[self.cutoff :]
        nondiff_args_tracer, nondiff_args_nontracer = eqx.partition(
            nondiff_args, is_tracer
        )
        nondiff_args_tracer = jax.tree_map(lax.stop_gradient, nondiff_args_tracer)
        return self.fn(nondiff_args_nontracer, nondiff_args_tracer, diff_args)


# TODO: I think the jacfwd and the jvp can probably be combined, as they both
# basically do the same thing. That might improve efficiency via parallelism.
def implicit_jvp(fn_primal, fn_rewrite, args, closure):
    """
    Takes a function `fn_primal : (args, closure) -> (root, residual)` and a function
    `fn_rewrite : (root, residual, args, closure) -> arb`.

    Has primals `fn_primal(args, closure)[0]` with auxiliary information
    `fn_primal(args, closure)[1]`.
    Has tangents `-(d(fn_rewrite)/d(root))^-1 d(fn_rewrite)/d(args)`, evaluated at
    `(root, residual, args, closure)`.

    This is used for rewriting gradients via the implicit function theorem.

    Note that due to limitations with JAX's custom autodiff, both `fn_primal` and
    `fn_rewrite` should be global functions (i.e. they should not capture any JAX array
    via closure, even if it does not participate in autodiff).
    """
    diff_args, nondiff_args = eqx.partition(args, eqx.is_inexact_array)
    root, residual = _implicit_backprop(
        fn_primal, fn_rewrite, nondiff_args, closure, diff_args
    )
    # Trim off the zero tangents we added to `residual`.
    return root, jax.tree_map(lax.stop_gradient, residual)


@ft.partial(fixed_custom_jvp, nondiff_argnums=(0, 1, 2, 3))
def _implicit_backprop(fn_primal, fn_rewrite, nondiff_args, closure, diff_args):
    del fn_rewrite
    args = eqx.combine(diff_args, nondiff_args)
    return fn_primal(args, closure)


@_implicit_backprop.defjvp
def _implicit_backprop_jvp(
    fn_primal, fn_rewrite, nondiff_args, closure, diff_args, tang_diff_args
):
    (diff_args,) = diff_args
    (tang_diff_args,) = tang_diff_args
    root, residual = _implicit_backprop(
        fn_primal, fn_rewrite, nondiff_args, closure, diff_args
    )

    flat_root, unflatten_root = fu.ravel_pytree(root)
    args = eqx.combine(nondiff_args, diff_args)

    def _for_jac(_root):
        _root = unflatten_root(_root)
        _out = fn_rewrite(_root, residual, args, closure)
        _out, _ = fu.ravel_pytree(_out)
        return _out

    jac_flat_root = jax.jacfwd(_for_jac)(flat_root)

    flat_diff_args, unflatten_diff_args = fu.ravel_pytree(diff_args)
    flat_tang_diff_args, _ = fu.ravel_pytree(tang_diff_args)

    def _for_jvp(_diff_args):
        _diff_args = unflatten_diff_args(_diff_args)
        _args = eqx.combine(nondiff_args, _diff_args)
        _out = fn_rewrite(root, residual, _args, closure)
        _out, _ = fu.ravel_pytree(_out)
        return _out

    _, jvp_flat_diff_args = jax.jvp(_for_jvp, (flat_diff_args,), (flat_tang_diff_args,))

    tang_root = -jnp.linalg.solve(jac_flat_root, jvp_flat_diff_args)
    tang_root = unflatten_root(tang_root)
    tang_residual = jax.tree_map(jnp.zeros_like, residual)
    return (root, residual), (tang_root, tang_residual)
