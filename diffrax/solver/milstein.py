from typing import Tuple

import jax
import jax.numpy as jnp

from ..custom_types import Bool, DenseInfo, PyTree, Scalar
from ..local_interpolation import LocalLinearInterpolation
from ..misc import ω
from ..solution import RESULTS
from ..term import AbstractTerm
from .base import AbstractItoSolver, AbstractStratonovichSolver


_ErrorEstimate = None
_SolverState = None


#
# The best online reference I've found for commutative-noise Milstein is
# https://www.performancetrading.it/Documents/KsStrong/KsS_Milstein.htm
#
# (It only gives it for Ito, but you can get Stratonovich by just ignoring the -dt
# correction term.)
#


class StratonovichMilstein(AbstractStratonovichSolver):
    r"""Milstein's method; Stratonovich version.

    Used to solve SDEs, and converges to the Stratonovich solution.

    !!! warning

        Requires [commutative noise](https://docs.kidger.site/diffrax/usage/how-to-choose-a-solver/#stochastic-differential-equations).
        Note that this commutativity condition is not checked.
    """  # noqa: E501

    term_structure = jax.tree_structure((0, 0))
    interpolation_cls = LocalLinearInterpolation

    def order(self, terms):
        raise ValueError("`StratonovichMilstein` should not used to solve ODEs.")

    def strong_order(self, terms):
        return 1  # assuming commutative noise

    def step(
        self,
        terms: Tuple[AbstractTerm, AbstractTerm],
        t0: Scalar,
        t1: Scalar,
        y0: PyTree,
        args: PyTree,
        solver_state: _SolverState,
        made_jump: Bool,
    ) -> Tuple[PyTree, _ErrorEstimate, DenseInfo, _SolverState, RESULTS]:
        del solver_state, made_jump
        drift, diffusion = terms
        dt = drift.contr(t0, t1)
        dw = diffusion.contr(t0, t1)

        f0_prod = drift.vf_prod(t0, y0, args, dt)
        g0_prod = diffusion.vf_prod(t0, y0, args, dw)

        def _to_jvp(_y0):
            return diffusion.vf_prod(t0, _y0, args, dw)

        _, v0_prod = jax.jvp(_to_jvp, (y0,), (g0_prod,))
        y1 = (y0**ω + f0_prod**ω + g0_prod**ω + 0.5 * v0_prod**ω).ω

        dense_info = dict(y0=y0, y1=y1)
        return y1, None, dense_info, None, RESULTS.successful

    def func(
        self,
        terms: Tuple[AbstractTerm, AbstractTerm],
        t0: Scalar,
        y0: PyTree,
        args: PyTree,
    ) -> PyTree:
        drift, diffusion = terms
        return drift.vf(t0, y0, args), diffusion.vf(t0, y0, args)


class ItoMilstein(AbstractItoSolver):
    r"""Milstein's method; Itô version.

    Used to solve SDEs, and converges to the Itô solution.

    !!! warning

        Requires [commutative noise](https://docs.kidger.site/diffrax/usage/how-to-choose-a-solver/#stochastic-differential-equations).
        Note that this commutativity condition is not checked.
    """  # noqa: E501

    term_structure = jax.tree_structure((0, 0))
    interpolation_cls = LocalLinearInterpolation

    def order(self, terms):
        raise ValueError("`StratonovichMilstein` should not used to solve ODEs.")

    def strong_order(self, terms):
        return 1  # assuming commutative noise

    def step(
        self,
        terms: Tuple[AbstractTerm, AbstractTerm],
        t0: Scalar,
        t1: Scalar,
        y0: PyTree,
        args: PyTree,
        solver_state: _SolverState,
        made_jump: Bool,
    ) -> Tuple[PyTree, _ErrorEstimate, DenseInfo, _SolverState, RESULTS]:
        del solver_state, made_jump
        drift, diffusion = terms
        Δt = drift.contr(t0, t1)
        Δw = diffusion.contr(t0, t1)

        #
        # So this is a bit involved, largely because of the generality that the rest of
        # the libary supports. (In particular arbitrary PyTrees, and arbitrary (linear)
        # `AbstractTerm.prod`)
        #
        # The expression for Ito Milstein is
        #
        # y1_{i} = y0_{i} + f0_{i} Δt + g0_{i j} Δw_{j} + 0.5 g0_{k j1} g0_{i j2, k} ΔwΔw_{j1 j2}  # noqa: E501
        #
        # where
        #
        # f0_{i} is the (vector-valued) drift at y0;
        # g0_{i j} is the (matrix-valued) diffusion at y0;
        # the index after the comma denotes a derivative;
        # ΔwΔw_{j1 j2} = Δw_{j1} Δw_{j2} if j1 != j2;
        # ΔwΔw_{j j} = Δw_{j} Δw_{j} - Δt.
        #
        # In particular note that that "-Δt" means ΔwΔw is not rank-1. This is what
        # makes the Stratonovich case so much simpler: the only mathematical difference
        # there is that the "-Δt" isn't present, but then dwdw decomposes, which
        # simplifies the computation immensely.
        #

        #
        # First we get f0 and g0 -- this is basically just the same as Euler's method.
        # We do need to materialise the diffusion g0 (and not just the diffusion-noise
        # product) as we need it below.
        #
        f0_prod = drift.vf_prod(t0, y0, args, Δt)
        g0 = diffusion.vf(t0, y0, args)
        g0_prod = diffusion.prod(g0, Δw)

        #
        # Next we construct ΔwΔw. (Which is symmetric.)
        #
        # To make sense of what's going on here, we're going to start referring to the
        # "structure" of an object. In this case ΔwΔw has "structure"
        #
        # (tree(Δw), tree(Δw), leaf(Δw), leaf(Δw))
        #
        # denoting the fact that it has a PyTree structure equivalent to placing a
        # Δw-structured PyTree on every leaf of a PyTree structured also like Δw, and
        # that for the leaf over the overall PyTree obtained by going to the i-th leaf
        # of the first PyTree structure and the j-th (sub-)leaf of the second PyTree
        # structure, will have a shape equivalent to the outer product of the i-th leaf
        # of Δw and the j-th leaf of Δw.
        #
        # Confused yet? Good.
        #
        # Basically, the structure is equivalent to the "outer product of two PyTrees".
        # This should make sense! In our mathematical notation above, ΔwΔw had a shape
        # equivalent to the outer product of Δw and Δw.
        #
        # We're going to use this notation extensively.
        #
        leaves_Δw, tree_Δw = jax.tree_flatten(Δw)
        leaves_ΔwΔw = []
        for i1, l1 in enumerate(leaves_Δw):
            for i2, l2 in enumerate(leaves_Δw):
                leaf = jnp.tensordot(l1[..., None], l2[None, ...], axes=1)
                if i1 == i2:
                    eye = jnp.eye(l1.size).reshape(l1.shape + l1.shape)
                    leaf = leaf - Δt * eye
                leaves_ΔwΔw.append(leaf)
        tree_ΔwΔw = tree_Δw.compose(tree_Δw)
        ΔwΔw = jax.tree_unflatten(tree_ΔwΔw, leaves_ΔwΔw)
        # ΔwΔw has structure (tree(Δw), tree(Δw), leaf(Δw), leaf(Δw))

        #
        # Next we construct g0_{k j1} g0_{i j2, k}.
        # Note the contraction over k: this is a JVP.
        # We denote this quantity v0 in the code below.
        #

        def _to_vjp(_y0):
            # _y0 has structure (tree(y0), leaf(y0))
            _out = diffusion.vf(t0, _y0, args)
            # _out has structure (tree(g0), leaf(g0))
            return _out

        def _to_vmap(_g0):
            # _g0 has structure (tree(y0), leaf(y0))
            _, _jvp = jax.jvp(_to_vjp, (y0,), (_g0,))
            # jvp has structure (tree(g0), leaf(g0))
            _jvp_matrix = jax.jacfwd(lambda _Δw: diffusion.prod(_jvp, _Δw))(Δw)
            # _jvp_matrix has structure (tree(y0), tree(Δw), leaf(y0), leaf(Δw))
            return _jvp_matrix

        # Aha! A new complexity.
        #
        # So the structure (tree(g0), leaf(g0)) is isomorphic to the structure
        # (tree(y0), tree(Δw), leaf(y0), leaf(Δw)).
        #
        # [
        # After all, the "default" product between a diffusion g0 and a control Δw is
        # simply a matrix-vector product. So if the PyTrees are trivial (for
        # simplicity), and y0 is a tensor with shape (d0, ..., dN) and Δw has shape
        # (e0, ..., eM), then the diffusion matrix must have shape
        # (d0, ..., dN, e0, ..., eM). That's what we mean by (leaf(y0), leaf(Δw)). And
        # then of course we just have the same pattern in the PyTree structure.
        # ]
        #
        # However it needn't actually be represented in that expanded structure. We
        # allow it to take any structure, and trust in (the overload of)
        # `AbstractTerm.prod` to actually interpret it for us.
        #
        # However here we actually need access to the expanded structure. Fortunately
        # we can get access to it (and materialise it -- hopefully not too inefficient)
        # by seeking the Jacobian of the *linear* operation Δw -> prod(..., Δw)
        # The Jacobian of a linear operation just being the matrix representation of
        # that operation, after all -- and as above, it's the matrix representation
        # that we seek.
        #
        # TODO: this implies a runtime overhead on XLA:CPU and a compiletime overhead
        # on XLA:GPU, see JAX issue #9215.

        def _to_treemap(_Δw, _g0):
            # _Δw has structure (leaf(Δw),)
            # _g0 has structure (tree(y0), leaf(y0), leaf(Δw))
            __to_vmap = _to_vmap
            for _ in range(jnp.ndim(_Δw)):
                __to_vmap = jax.vmap(__to_vmap, in_axes=-1, out_axes=-1)
            out = __to_vmap(_g0)
            # _out has structure (tree(y0), tree(Δw), leaf(y0), leaf(Δw), leaf(Δw))
            return out

        y_treedef = jax.tree_structure(y0)
        Δw_treedef = jax.tree_structure(Δw)
        # g0 has structure (tree(g0), leaf(g0))
        # Which we now transform into its isomorphic matrix form, as above.
        g0_matrix = jax.jacfwd(lambda _Δw: diffusion.prod(g0, _Δw))(Δw)
        # g0_matrix has structure (tree(y0), tree(Δw), leaf(y0), leaf(Δw))
        g0_matrix = jax.tree_transpose(y_treedef, Δw_treedef, g0_matrix)
        # g0_matrix has structure (tree(Δw), tree(y0), leaf(y0), leaf(Δw))
        v0_matrix = jax.tree_map(_to_treemap, Δw, g0_matrix)
        # v0_matrix has structure (tree(Δw), tree(y0), tree(Δw), leaf(y0), leaf(Δw), leaf(Δw))  # noqa: E501
        v0_matrix = jax.tree_transpose(
            Δw_treedef, y_treedef.compose(Δw_treedef), v0_matrix
        )
        # v0_matrix has structure (tree(y0), tree(Δw), tree(Δw), leaf(y0), leaf(Δw), leaf(Δw))  # noqa: E501

        #
        # Now we need to contract g0_{k j1} g0_{i j2, k} against ΔwΔw_{j1 j2}.
        #

        def __dot(_v0, _ΔwΔw):
            # _v0 has structure (leaf(y0), leaf(Δw), leaf(Δw))
            # _ΔwΔw has structure (leaf(Δw), leaf(Δw))
            _out = jnp.tensordot(_v0, _ΔwΔw, axes=jnp.ndim(_ΔwΔw))
            # _out has structure (leaf(y0),)
            return _out

        def _dot(_, _v0):
            # _v0 has structure (tree(Δw), tree(Δw), leaf(y0), leaf(Δw), leaf(Δw))
            # ΔwΔw has structure (tree(Δw), tree(Δw), leaf(Δw), leaf(Δw))
            _dotted = jax.tree_map(__dot, _v0, ΔwΔw)
            # _dotted has structure (tree(Δw), tree(Δw), leaf(y0))
            _out = sum(jax.tree_leaves(_dotted))
            # _out has structure (leaf(y0),)
            return _out

        # v0_matrix has structure (tree(y0), tree(Δw), tree(Δw), leaf(y0), leaf(Δw), leaf(Δw))  # noqa: E501
        v0_prod = jax.tree_map(_dot, y0, v0_matrix)
        # v0_prod has structure (tree(y0), leaf(y0))

        #
        # Finally we get to add everything together. Phew.
        #
        y1 = (y0**ω + f0_prod**ω + g0_prod**ω + 0.5 * v0_prod**ω).ω

        #
        # A couple of final notes:
        #
        # - Note how contract over j1 and j2 simultaneously. This is directly analogous
        #   to the task of computing trace(AB), which in Einstein notation is
        #   A_{i j} B_{j k} δ_{i k}. If would be tempting to compute this by doing a
        #   matrix-matrix product AB, and then taking a trace. But that involves
        #   materialising all of AB, just so we can ignore most of it and sum the
        #   diagonal.
        #   Rewriting the Einstein notation as A_{i j} B_{j i}, we see a more efficient
        #   way of computing this quantity: by doing an element-wise multiplication of
        #   A against B-transpose, and then summing every entry.
        #   Moral of the story: if the summation indices in a tensor network form a
        #   cycle, then you're doing something trace-like, and should try and contract
        #   over two indices at the same time.
        #   In this case, we contract over j1 and j2 simultaneously, between
        #   ΔwΔw_{j1 j2} and g0_{k j1} g0_{i j2, k}.
        #
        # - Note that in the overall expression g0_{k j1} g0_{i j2, k} ΔwΔw_{j1 j2},
        #   we have summation indices j1, j2, k. If we were to have contracted down j1
        #   or j2 too first, then we couldn't have pulled the double-contraction trick
        #   discussed in the first bullet point. This is because we'd have a
        #   contraction against a "top" and a "bottom" index in the Jacobian
        #   g0_{i j2, k}. We can do the former as a VJP and the latter as JVP, but we
        #   can't do both at the same time without materialising the full Jacobian.
        #   In other words, we could only do a double-contraction over (j1, j2), but
        #   not (j1, k) or (j2, k), and this is why we contract over k first.
        #
        # - Note that in those expressions featuring two lots of Δw-like-structure,
        #   that we never try to distinguish one Δw-like-structure from the other
        #   Δw-like-structure. This is because ΔwΔw is symmetric.
        #

        dense_info = dict(y0=y0, y1=y1)
        return y1, None, dense_info, None, RESULTS.successful

    def func(
        self,
        terms: Tuple[AbstractTerm, AbstractTerm],
        t0: Scalar,
        y0: PyTree,
        args: PyTree,
    ) -> PyTree:
        drift, diffusion = terms
        return drift.vf(t0, y0, args), diffusion.vf(t0, y0, args)
