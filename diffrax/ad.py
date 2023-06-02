import functools as ft

import equinox as eqx
import jax
import jax.flatten_util as fu
import jax.lax as lax
import jax.numpy as jnp
import jax.tree_util as jtu


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
        nondiff_args_tracer = jtu.tree_map(lax.stop_gradient, nondiff_args_tracer)
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
    return root, jtu.tree_map(lax.stop_gradient, residual)


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
    tang_residual = jtu.tree_map(jnp.zeros_like, residual)
    return (root, residual), (tang_root, tang_residual)
