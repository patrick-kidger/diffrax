import math
import tempfile

import equinox as eqx
import fire
import jax
import jax.numpy as jnp
import neural_ode
import optax
import pysr
import sympy


class SymbolicFn(eqx.Module):
    fn: callable
    parameters: jnp.ndarray

    def __call__(self, x):
        return self.fn(x, self.parameters)


def optimise(fn, in_, out, steps, lr):
    def loss(fn):
        # No vmap -- PySR creates functions that operate on batches.
        return jnp.mean((fn(in_) - out) ** 2)

    optim = optax.sgd(lr)
    opt_state = optim.init(eqx.filter(fn, eqx.is_array))

    @eqx.filter_jit
    def step(fn, opt_state):
        grads = eqx.filter_grad(loss)(fn)
        updates, opt_state = optim.update(grads, opt_state)
        fn = eqx.apply_updates(fn, updates)
        return fn, opt_state

    for _ in range(steps):
        fn, opt_state = step(fn, opt_state)

    return fn, loss(fn).item()


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


def quantise(expr, quantise_to):
    if isinstance(expr, sympy.Float):
        return expr.func(round(float(expr) / quantise_to) * quantise_to)
    elif isinstance(expr, sympy.Symbol):
        return expr
    else:
        return expr.func(*[quantise(arg, quantise_to) for arg in expr.args])


def expr_size(expr):
    return sum(expr_size(v) for v in expr.args) + 1


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
    fine_tuning_steps=500,
    fine_tuning_lr=3e-4,
    fine_tuning_quantise_to=0.01,
    pareto_coefficient=2,
    seed=5678,
):
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

    vector_field = model.solver.term.vector_field.mlp
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

    expressions = []
    for pareto_frontier_i, out_i in zip(pareto_frontier, jnp.rollaxis(out, 1)):
        best_expr = None
        best_expr_size = None
        best_expr_value = None
        for symbolic_fn in pareto_frontier_i.itertuples():
            expr = symbolic_fn.sympy_format
            parameters = symbolic_fn.jax_format["parameters"]
            symbolic_fn = symbolic_fn.jax_format["callable"]
            symbolic_fn = SymbolicFn(symbolic_fn, parameters)
            symbolic_fn, loss = optimise(
                symbolic_fn, in_, out_i, fine_tuning_steps, fine_tuning_lr
            )
            expr = replace_parameters(expr, symbolic_fn.parameters.tolist())
            expr = quantise(expr, fine_tuning_quantise_to).simplify()
            if best_expr is None:
                best_expr = expr
                best_expr_size = expr_size(expr)
                best_expr_value = math.log(loss, pareto_coefficient) + best_expr_size
            else:
                _expr_size = expr_size(expr)
                expr_value = math.log(loss, pareto_coefficient) + _expr_size
                if expr_value < best_expr_value or (
                    (expr_value == best_expr_value) and (_expr_size < best_expr_size)
                ):
                    best_expr = expr
                    best_expr_size = _expr_size
                    best_expr_value = expr_value
        expressions.append(best_expr)

    print(f"Expressions found: {expressions}")


if __name__ == "__main__":
    fire.Fire(main)
