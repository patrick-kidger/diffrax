###########
#
# This example combines neural differential equations with regularised evolution to
# discover the equations
#
# dx        y(t)
# --(t) = --------
# dt      1 + y(t)
#
# dy       -x(t)
# --(t) = --------
# dt      1 + x(t)
#
# directly from data.
#
###########

import math
import tempfile
from typing import List

import equinox as eqx
import fire
import jax
import jax.numpy as jnp
import neural_ode
import optax
import pysr
import sympy


def quantise(expr, quantise_to):
    if isinstance(expr, sympy.Float):
        return expr.func(round(float(expr) / quantise_to) * quantise_to)
    elif isinstance(expr, sympy.Symbol):
        return expr
    else:
        return expr.func(*[quantise(arg, quantise_to) for arg in expr.args])


class SymbolicFn(eqx.Module):
    fn: callable
    parameters: jnp.ndarray

    def __call__(self, x):
        # Dummy batch/unbatching. PySR assumes its JAX'd symbolic functions act on
        # tensors with a single batch dimension.
        return jnp.squeeze(self.fn(x[None], self.parameters))


class Stack(eqx.Module):
    modules: List[eqx.Module]

    def __call__(self, x):
        return jnp.stack([module(x) for module in self.modules], axis=-1)


def expr_size(expr):
    return sum(expr_size(v) for v in expr.args) + 1


def _replace_parameters(expr, parameters, i_ref):
    if isinstance(expr, sympy.Float):
        i_ref[0] += 1
        return expr.func(parameters[i_ref[0]])
    elif isinstance(expr, sympy.Symbol):
        return expr
    else:
        return expr.func(
            *[_replace_parameters(arg, parameters, i_ref) for arg in expr.args]
        )


def replace_parameters(expr, parameters):
    i_ref = [-1]  # Distinctly sketchy approach to making this conversion.
    return _replace_parameters(expr, parameters, i_ref)


def main(
    neural_dataset_size=256,
    neural_batch_size=32,
    neural_lr=3e-3,
    neural_steps=5000,
    neural_width_size=64,
    neural_depth=2,
    symbolic_dataset_size=2000,
    symbolic_num_populations=100,
    symbolic_population_size=20,
    symbolic_migration_steps=10,
    symbolic_mutation_steps=50,
    symbolic_descent_steps=50,
    pareto_coefficient=2,
    fine_tuning_steps=500,
    fine_tuning_lr=3e-3,
    quantise_to=0.01,
    seed=5678,
):
    ###########
    # First obtain a neural approximation to the dynamics.
    ###########
    ts, ys, model, _ = neural_ode.main(
        dataset_size=neural_dataset_size,
        batch_size=neural_batch_size,
        lr=neural_lr,
        steps=neural_steps,
        width_size=neural_width_size,
        depth=neural_depth,
        seed=seed,
        plot=False,
    )

    ###########
    # Now symbolically regress across the learnt vector field, to obtain a Pareto
    # frontier of symbolic equations, that trade loss against complexity of the
    # equation.
    ###########
    vector_field = model.solver.term.vector_field.impl
    dataset_size, length_size, data_size = ys.shape
    in_ = ys.reshape(dataset_size * length_size, data_size)[:symbolic_dataset_size]
    out = jax.vmap(vector_field)(in_)
    with tempfile.TemporaryDirectory() as tempdir:
        pareto_frontier = pysr.pysr(
            in_,
            out,
            niterations=symbolic_migration_steps,
            ncyclesperiteration=symbolic_mutation_steps,
            populations=symbolic_num_populations,
            npop=symbolic_population_size,
            optimizer_iterations=symbolic_descent_steps,
            optimizer_nrestarts=1,
            procs=1,
            tempdir=tempdir,
            temp_equation_file=True,
            output_jax_format=True,
        )

    ###########
    # We now select the `best' equation from this frontier.
    # PySR actually has a built-in way of doing this (`parsimony`) if you want.
    ###########
    expressions = []
    symbolic_fns = []
    for pareto_frontier_i, out_i in zip(pareto_frontier, jnp.rollaxis(out, 1)):
        best_expression = None
        best_symbolic_fn = None
        best_expr_size = None
        best_expr_value = None
        for expr in pareto_frontier_i.itertuples():
            symbolic_fn = SymbolicFn(
                expr.jax_format["callable"], expr.jax_format["parameters"]
            )
            loss = jnp.mean((jax.vmap(symbolic_fn)(in_) - out_i) ** 2)
            if best_expression is None:
                best_expression = expr.sympy_format
                best_symbolic_fn = symbolic_fn
                best_expr_size = expr_size(expr.sympy_format)
                best_expr_value = math.log(loss, pareto_coefficient) + best_expr_size
            else:
                _expr_size = expr_size(expr.sympy_format)
                expr_value = math.log(loss, pareto_coefficient) + _expr_size
                if expr_value < best_expr_value or (
                    (expr_value == best_expr_value) and (_expr_size < best_expr_size)
                ):
                    best_expression = expr.sympy_format
                    best_symbolic_fn = symbolic_fn
                    best_expr_size = _expr_size
                    best_expr_value = expr_value
        expressions.append(best_expression)
        symbolic_fns.append(best_symbolic_fn)

    ###########
    # Now the constants in this expression have been optimised for regressing across
    # the neural vector field. This was good enough to obtain the symbolic expression,
    # but won't quite be perfect -- some of the constants will be slightly off.
    #
    # To fix this we now plug our symbolic function back into the original (neural)
    # model and apply gradient descent.
    ###########
    symbolic_fn = Stack(symbolic_fns)
    symbolic_model = eqx.tree_at(
        lambda m: m.solver.term.vector_field.impl,
        model,
        symbolic_fn,
        replace_subtree=True,
    )

    @eqx.filter_grad
    def grad_loss(symbolic_model):
        vmap_model = jax.vmap(symbolic_model, in_axes=(None, 0))
        pred_ys = vmap_model(ts, ys[:, 0])
        return jnp.mean((ys - pred_ys) ** 2)

    optim = optax.adam(fine_tuning_lr)
    opt_state = optim.init(eqx.filter(symbolic_model, eqx.is_array))
    for _ in range(fine_tuning_steps):
        grads = grad_loss(symbolic_model)
        updates, opt_state = optim.update(grads, opt_state)
        symbolic_model = eqx.apply_updates(symbolic_model, updates)

    ###########
    # Finally we round each constant to the nearest multiple of `quantise_to`.
    ###########
    trained_expressions = []
    for module, expression in zip(
        symbolic_model.solver.term.vector_field.impl.modules, expressions
    ):
        expression = replace_parameters(expression, module.parameters.tolist())
        expression = quantise(expression, quantise_to)
        trained_expressions.append(expression)

    print(f"Expressions found: {trained_expressions}")


if __name__ == "__main__":
    fire.Fire(main)
