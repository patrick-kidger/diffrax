from typing import Any

import equinox as eqx
import jax
import jax.interpreters.ad as ad
import jax.interpreters.batching as batching
import jax.interpreters.xla as xla
import jax.lax as lax

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
xla.register_translation(
    _nondifferentiable_output_p,
    xla.lower_fun(lambda x: x, multiple_results=False, new_style=True),
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
