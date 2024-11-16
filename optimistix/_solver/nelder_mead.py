import functools as ft
from collections.abc import Callable
from typing import Any, cast, Generic

import equinox as eqx
import jax
import jax.lax as lax
import jax.numpy as jnp
import jax.tree_util as jtu
from equinox.internal import ω
from jaxtyping import Array, ArrayLike, Bool, PyTree, Scalar

from .._custom_types import Aux, Fn, Y
from .._minimise import AbstractMinimiser
from .._misc import max_norm, tree_full_like, tree_where
from .._solution import RESULTS


class _NMStats(eqx.Module, strict=True):
    n_reflect: Scalar
    n_inner_contract: Scalar
    n_outer_contract: Scalar
    n_expand: Scalar
    n_shrink: Scalar


class _NelderMeadState(eqx.Module, Generic[Y, Aux], strict=True):
    """
    Information to update and store the simplex of the Nelder Mead update. If
    `dim` is the dimension of the problem, we expect there to be
    `n_vertices = dim + 1` vertices. We expect the leading axis of each leaf
    to be of length `n_vertices`, and the sum of the rest of the axes of all leaves
    together to be `dim`.

    - `simplex`: a PyTree with leading axis of leaves `n_vertices` and sum of the
        rest of the axes of all leaves `dim`.
    - `f_simplex`: a 1-dimensional array of size `n_vertices`.
        The values of the problem function evaluated on each vertex of
        simplex.
    - `best`: A tuple of shape (Scalar, PyTree, Scalar). The tuple contains
        (`f(best_vertex)`, `best_vertex`, index of `best_vertex`) where
        `best_vertex` is the vertex which minimises `f` among all vertices in
        `simplex`.
    - `worst`: A tuple of shape (Scalar, PyTree, Scalar). The tuple contains
        (`f(worst_vertex)`, `worst_vertex`, index of `worst_vertex`) where
        `worst_vertex` is the vertex which maximises `f` among all vertices in
        `simplex`.
    -`second_worst`: A scalar, which is `f(second_worst_vertex)` where
        `second_worst_vertex` is the vertex which maximises `f` among all vertices
        in `simplex` with `worst_vertex` removed.
    - `step`: A scalar. How many steps have been taken so far.
    - `stats`: A `_NMStats` PyTree. This tracks information about the Nelder Mead
        algorithm. Specifically, how many times each of the operations reflect,
        expand, inner contract, outer contract, and shrink are performed.
    - `result`: a [`optimistix.RESULTS`][] object which indicates if we have diverged
        during the course of optimisation.
    - `first_pass`: A bool which indicates if this is the first call to Nelder Mead
        which allows for extra setup. This ultimately exists to save on compilation
        time.
    """

    simplex: PyTree
    f_simplex: Array
    best: tuple[Scalar, Y, Scalar]
    worst: tuple[Scalar, Y, Scalar]
    second_worst_val: Scalar
    stats: _NMStats
    result: RESULTS
    first_pass: Bool[ArrayLike, ""]
    aux: Aux
    step: Scalar


def _update_stats(
    stats,
    reflect: Bool[Array, ""],
    inner_contract: Bool[Array, ""],
    outer_contract: Bool[Array, ""],
    expand: Bool[Array, ""],
    shrink: Bool[Array, ""],
) -> _NMStats:
    return _NMStats(
        stats.n_reflect + jnp.where(reflect, 1, 0),
        stats.n_inner_contract + jnp.where(inner_contract, 1, 0),
        stats.n_outer_contract + jnp.where(outer_contract, 1, 0),
        stats.n_expand + jnp.where(expand, 1, 0),
        stats.n_shrink + jnp.where(shrink, 1, 0),
    )


class NelderMead(AbstractMinimiser[Y, Aux, _NelderMeadState[Y, Aux]], strict=True):
    """The Nelder-Mead minimisation algorithm. (Downhill simplex derivative-free
    method.)

    This algorithm is notable in that it only uses function evaluations, and does not
    need gradient evaluations.

    This is usually an "algorithm of last resort". Gradient-based algorithms are usually
    much faster, and be more likely to converge to a minima.

    Comparable to `scipy.optimize.minimize(method="Nelder-Mead")`.
    """

    rtol: float
    atol: float
    norm: Callable[[PyTree], Scalar] = max_norm
    rdelta: float = 5e-2
    adelta: float = 2.5e-4

    def init(
        self,
        fn: Fn[Y, Scalar, Aux],
        y: Y,
        args: PyTree,
        options: dict[str, Any],
        f_struct: PyTree[jax.ShapeDtypeStruct],
        aux_struct: PyTree[jax.ShapeDtypeStruct],
        tags: frozenset[object],
    ) -> _NelderMeadState[Y, Aux]:
        aux = tree_full_like(aux_struct, 0)
        y0_simplex = options.get("y0_simplex", False)

        if y0_simplex:
            n_vertices = {x.shape[0] for x in jtu.tree_leaves(y)}
            if len(n_vertices) == 1:
                [n_vertices] = n_vertices
            else:
                raise ValueError(
                    "The PyTree must form a valid simplex. \
                    Got different leading dimension (number of vertices) \
                    for each leaf"
                )
            size = jtu.tree_reduce(
                lambda x, y: x + y, jtu.tree_map(lambda x: x[1:].size, y)
            )
            if n_vertices != size + 1:
                raise ValueError(
                    "The PyTree must form a valid simplex. Got"
                    f"{n_vertices} vertices but dimension {size}."
                )
            simplex = y
        else:
            #
            # The standard approach to creating the initial simplex from a single vector
            # is to add a small constant times each unit vector to the initial vector.
            # The constant is different if the `y` is 0 in the direction of the
            # unit vector. Just because this is standard, does not mean it's well
            # justified. We add rdelta * y[i] + adelta y[i] in the ith unit direction.
            #
            size = jtu.tree_reduce(lambda x, y: x + y, jtu.tree_map(jnp.size, y))
            n_vertices = size + 1
            leaves, treedef = jtu.tree_flatten(y)
            running_size = 0
            new_leaves = []

            for index, leaf in enumerate(leaves):
                # TODO(raderj): This is a difficult bit of Pytree manipulation, and
                # needs to either be simplified if possible or explained, maybe with a
                # follow-along example.
                leaf_size = jnp.size(leaf)
                broadcast_leaves = jnp.repeat(leaf[None, ...], size + 1, axis=0)
                indices = jnp.arange(
                    running_size + 1, running_size + leaf_size + 1, dtype=jnp.int16
                )
                relative_indices = jnp.unravel_index(
                    indices - running_size, shape=leaf.shape
                )
                indices = jnp.unravel_index(indices, shape=broadcast_leaves.shape)
                broadcast_leaves = broadcast_leaves.at[indices].set(
                    broadcast_leaves[indices]
                    + self.adelta
                    + self.rdelta * leaf[relative_indices]
                )
                running_size = running_size + leaf_size
                new_leaves.append(broadcast_leaves)
            simplex = jtu.tree_unflatten(treedef, new_leaves)

        f_simplex = jnp.full(n_vertices, jnp.inf, dtype=f_struct.dtype)
        # Shrink will be called the first time step is called, so remove one from stats
        # at init.
        stats = _NMStats(
            jnp.array(0), jnp.array(0), jnp.array(0), jnp.array(0), jnp.array(-1)
        )

        return _NelderMeadState(
            simplex=simplex,
            f_simplex=f_simplex,
            best=(
                jnp.array(0.0),
                jtu.tree_map(lambda x: x[0], simplex),
                jnp.array(0, dtype=jnp.int32),
            ),
            worst=(
                jnp.array(0.0),
                jtu.tree_map(lambda x: x[0], simplex),
                jnp.array(0, dtype=jnp.int32),
            ),
            second_worst_val=jnp.array(0.0),
            stats=stats,
            result=RESULTS.successful,
            first_pass=jnp.array(True),
            aux=aux,
            step=jnp.array(0),
        )

    def step(
        self,
        fn: Fn[Y, Scalar, Aux],
        y: Y,
        args: PyTree,
        options: dict[str, Any],
        state: _NelderMeadState[Y, Aux],
        tags: frozenset[object],
    ) -> tuple[Y, _NelderMeadState[Y, Aux], Aux]:
        # TODO(raderj): update to line search api.
        reflect_const = 2
        expand_const = 3
        out_const = 1.5
        in_const = 0.5
        shrink_const = 0.5

        f_best, best, best_index = state.best
        f_worst, worst, worst_index = state.worst
        f_second_worst = state.second_worst_val
        stats = state.stats
        (n_vertices,) = state.f_simplex.shape

        def init_step(state):
            simplex = state.simplex
            f_new_vertex = jnp.array(jnp.inf)
            return (
                state,
                simplex,
                jtu.tree_map(lambda _, x: x[0], y, simplex),
                f_new_vertex,
                state.stats,
                state.aux,
            )

        def main_step(state):
            # TODO(raderj): Calculate the centroid and search dir based upon
            # the prior one.
            search_direction = jtu.tree_map(
                lambda a, b: a - b[None], state.simplex, worst
            )
            search_direction = (
                (ω(search_direction) / (n_vertices - 1))
                .call(ft.partial(jnp.sum, axis=0))
                .ω
            )
            reflection = (worst**ω + reflect_const * search_direction**ω).ω

            def eval_new_vertices(vertex_carry, i):
                vertex, (f_vertex, _), stats = vertex_carry

                def internal_eval(f_vertex, stats):
                    expand = f_vertex < f_best
                    inner_contract = f_vertex > f_worst
                    contract = f_vertex > f_second_worst
                    outer_contract = jnp.invert(expand | contract)
                    reflect = (f_vertex > f_best) & (f_vertex < f_second_worst)
                    contract_const = jnp.where(inner_contract, in_const, out_const)
                    #
                    # NOTE: Throughout, we use worst + const * search_direction, not
                    # centroid + const * search direction! The latter
                    # is ubiquitious in Nelder-Mead implementations, but is a
                    # mistake. Nearly every line search assumes:
                    #
                    # a) `delta > 0` for `delta` the line search param.
                    # b) `f(x + delta dir) < f(x)` for suitably small `delta`
                    #
                    # both of these are violated when we start the line search at
                    # centroid (indeed, inner contraction usually requires
                    # multiplication by a negative constant). However, if we start
                    # the search at worst both these issues are avoided.
                    #
                    expanded = (worst**ω + expand_const * search_direction**ω).ω
                    contracted = (worst**ω + contract_const * search_direction**ω).ω
                    new_vertex = tree_where(
                        expand,
                        expanded,
                        vertex,
                    )
                    new_vertex = tree_where(
                        contract,
                        contracted,
                        new_vertex,
                    )
                    stats = _update_stats(
                        stats,
                        reflect,
                        inner_contract,
                        outer_contract,
                        expand,
                        shrink=jnp.array(False),
                    )
                    return new_vertex, stats

                out, stats = lax.cond(
                    i == 1,
                    internal_eval,
                    lambda x, y: (vertex, stats),
                    *(f_vertex, stats),
                )
                return (out, fn(out, args), stats), None

            #
            # We'd like to avoid making two calls to problem.fn in the step to
            # avoid compiling the potentially large problem.fn twice. Instead, we
            # wrap the two evaluations in a single scan and pass different inputs
            # each time.
            #
            # the first iteration of scan will have lax.cond(False) and will return
            # f(reflection) for use in the second iteration which calls internal_eval
            # which uses f(reflection) to determine the next vertex, and returns the
            # vertex along with f(new_vertex) and stats.
            #
            # TODO(raderj): pull out this entire thing into line  api.
            #
            # TODO(raderj): eqxi.scan_trick.
            #
            (new_vertex, (f_new_vertex, aux), stats), _ = lax.scan(
                eval_new_vertices,
                (reflection, (jnp.array(0.0), state.aux), state.stats),
                jnp.arange(2),
            )
            return (state, state.simplex, new_vertex, f_new_vertex, stats, aux)

        state, simplex, new_vertex, f_new_vertex, stats, aux = lax.cond(
            state.first_pass, init_step, main_step, state
        )
        new_best = f_new_vertex < f_best
        best = tree_where(new_best, new_vertex, best)
        f_best = jnp.where(new_best, f_new_vertex, f_best)
        shrink = cast(
            Array, jnp.where(state.first_pass, jnp.array(True), f_new_vertex > f_worst)
        )
        stats = _update_stats(
            stats,
            reflect=jnp.array(False),
            inner_contract=jnp.array(False),
            outer_contract=jnp.array(False),
            expand=jnp.array(False),
            shrink=shrink,
        )

        def shrink_simplex(best, new_vertex, simplex, first_pass):
            # This is just `best + shrink_const * (simplex - best)` but returns
            # a buffer. `best` is passed t
            shrink_simplex = jtu.tree_map(
                lambda a, b: a.at[...].set(b[None] + shrink_const * (a - b[None])),
                simplex,
                best,
            )
            # if it is the first pass and we just wanted to use shrink_simplex to
            # compute f_simplex, return simplex
            simplex = tree_where(first_pass, simplex, shrink_simplex)
            unwrapped_simplex = jtu.tree_map(lambda _, x: x[...], y, simplex)
            f_simplex, _ = jax.vmap(lambda x: fn(x, args))(unwrapped_simplex)
            return f_simplex, simplex

        def update_simplex(best, new_vertex, simplex, first_pass):
            simplex = jtu.tree_map(
                lambda _, a, b: a.at[worst_index].set(b), best, simplex, new_vertex
            )
            f_simplex = state.f_simplex.at[worst_index].set(f_new_vertex)
            return f_simplex, simplex

        f_new_simplex, new_simplex = lax.cond(
            shrink,
            shrink_simplex,
            update_simplex,
            *(best, new_vertex, state.simplex, state.first_pass),
        )
        #
        # TODO(raderj): only 1 value is updated when not shrinking. This implies
        # that in most cases, rather than do a top_k search in log time, we can
        # just compare f_next_vector and f_best and choose best between those two
        # in constant time. Implement this. A similar thing could likely be done with
        # worst and second worst, with recomputation occurring only when f_new < f_worst
        # but f_new > f_second_worst (otherwise, there will have been a shrink).
        #
        (f_best_neg,), (best_index,) = lax.top_k(-f_new_simplex, 1)
        f_best = -f_best_neg
        (f_worst, f_second_worst), (worst_index, _) = lax.top_k(f_new_simplex, 2)
        _structured_index = lambda a, b: jtu.tree_map(lambda _, x: x[b], y, a)
        best = _structured_index(simplex, best_index)
        worst = _structured_index(simplex, worst_index)
        new_state = _NelderMeadState(
            simplex=new_simplex,
            f_simplex=f_new_simplex,
            best=(f_best, best, best_index),
            worst=(f_worst, worst, worst_index),
            second_worst_val=f_second_worst,
            stats=stats,
            result=state.result,
            first_pass=jnp.array(False),
            aux=aux,
            step=state.step + 1,
        )
        try:
            y0_simplex = options["y0_simplex"]
        except KeyError:
            y0_simplex = False

        if y0_simplex:
            out = simplex
        else:
            out = best

        return out, new_state, aux

    def terminate(
        self,
        fn: Fn[Y, Scalar, Aux],
        y: Y,
        args: PyTree,
        options: dict[str, Any],
        state: _NelderMeadState[Y, Aux],
        tags: frozenset[object],
    ) -> tuple[Bool[Array, ""], RESULTS]:
        # TODO(raderj): only check terminate every k
        f_best, best, best_index = state.best
        x_scale = (self.atol + self.rtol * ω(best)[None].call(jnp.abs)).ω
        x_diff = jtu.tree_map(lambda a, b: jnp.abs(a - b[None]), state.simplex, best)
        x_converged = self.norm((x_diff**ω / x_scale**ω).ω) < 1
        f_scale = (self.atol + self.rtol * ω(f_best).call(jnp.abs)).ω
        f_diff = (state.f_simplex**ω - f_best**ω).call(jnp.abs).ω
        f_converged = self.norm((f_diff**ω / f_scale**ω).ω) < 1
        #
        # minpack does a further test here where it takes for each unit vector e_i a
        # perturbation "delta" and asserts that f(x + delta e_i) > f(x) and
        # f(x - delta e_i) > f(x). ie. a rough assertion that it is indeed a local
        # minimum. If it fails the algo resets completely. thus process scales as
        # O(dim(y) * T(f)), where T(f) is the cost of evaluating f.
        #
        converged = x_converged & f_converged
        diverged = jnp.any(jnp.invert(jnp.isfinite(f_best)))
        terminate = converged | diverged
        result = RESULTS.where(
            diverged, RESULTS.nonlinear_divergence, RESULTS.successful
        )
        return terminate, result

    def postprocess(
        self,
        fn: Fn[Y, Scalar, Aux],
        y: Y,
        aux: Aux,
        args: PyTree,
        options: dict[str, Any],
        state: _NelderMeadState,
        tags: frozenset[object],
        result: RESULTS,
    ) -> tuple[Y, Aux, dict[str, Any]]:
        stats = dict(
            num_reflections=state.stats.n_reflect,
            num_inner_contractions=state.stats.n_inner_contract,
            num_outer_contractions=state.stats.n_outer_contract,
            num_expansions=state.stats.n_expand,
            num_shrinkages=state.stats.n_shrink,
        )
        return y, aux, stats


NelderMead.__init__.__doc__ = """**Arguments:**

- `rtol`: Relative tolerance for terminating the solve.
- `atol`: Absolute tolerance for terminating the solve.
- `norm`: The norm used to determine the difference between two iterates in the 
    convergence criteria. Should be any function `PyTree -> Scalar`. Optimistix
    includes three built-in norms: [`optimistix.max_norm`][],
    [`optimistix.rms_norm`][], and [`optimistix.two_norm`][].
- `rdelta`: Nelder-Mead creates an initial simplex by appending a scaled identity 
    matrix to `y`. The `i`th element of this matrix is `rdelta * y_i + adelta`.
    That is, this is the relative size for creating the initial simplex.
- `adelta`: Nelder-Mead creates an initial simplex by appending a scaled identity 
    matrix to `y`. The `i`th element of this matrix is `rdelta * y_i + adelta`.
    That is, this is the absolute size for creating the initial simplex.
"""
