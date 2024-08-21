import abc
from collections.abc import Callable
from typing import Any, Generic, TypeVar

import equinox as eqx
import equinox.internal as eqxi
import jax
import jax.lax as lax
import jax.numpy as jnp
import jax.tree_util as jtu
from equinox import AbstractVar
from jax import vmap
from jaxtyping import PyTree

from .._custom_types import (
    AbstractBrownianIncrement,
    BoolScalarLike,
    DenseInfo,
    RealScalarLike,
)
from .._local_interpolation import LocalLinearInterpolation
from .._solution import RESULTS
from .._term import (
    AbstractTerm,
    LangevinDiffusionTerm,
    LangevinDriftTerm,
    LangevinLeaf,
    LangevinTuple,
    LangevinX,
    MultiTerm,
    WrapTerm,
)
from .base import AbstractItoSolver, AbstractStratonovichSolver


_ErrorEstimate = TypeVar("_ErrorEstimate", None, LangevinTuple)
_LangevinArgs = tuple[LangevinX, LangevinX, Callable[[LangevinX], LangevinX]]


def _get_args_from_terms(
    terms: MultiTerm[tuple[AbstractTerm[Any, RealScalarLike], AbstractTerm]],
) -> tuple[PyTree, PyTree, Callable[[LangevinX], LangevinX]]:
    drift, diffusion = terms.terms
    if isinstance(drift, WrapTerm):
        assert isinstance(diffusion, WrapTerm)
        drift = drift.term
        diffusion = diffusion.term

    assert isinstance(drift, LangevinDriftTerm)
    assert isinstance(diffusion, LangevinDiffusionTerm)
    gamma = drift.gamma
    u = drift.u
    f = drift.grad_f
    return gamma, u, f


def _broadcast_pytree(source, target_tree):
    # Broadcasts the source PyTree to the shape and PyTree structure of
    # target_tree_shape. Requires that source is a prefix tree of target_tree
    # This is used to case gamma and u to the shape of x0 and v0
    def _inner_broadcast(_src_arr, _inner_target_tree):
        _arr = jnp.asarray(_src_arr)

        def fun(_leaf):
            return jnp.asarray(
                jnp.broadcast_to(_arr, _leaf.shape), dtype=jnp.result_type(_leaf)
            )

        return jtu.tree_map(fun, _inner_target_tree)

    return jtu.tree_map(_inner_broadcast, source, target_tree)


# CONCERNING COEFFICIENTS:
# The coefficients used in a step of this SRK depend on
# the time increment h, and the parameter gamma.
# Assuming the modelled SDE stays the same (i.e. gamma is fixed),
# then these coefficients must be recomputed each time h changes.
# Furthermore, for very small h, directly computing the coefficients
# via the function below can cause large floating point errors.
# Hence, we pre-compute the Taylor expansion of the SRK coefficients
# around h=0. Then we can compute the SRK coefficients either via
# the Taylor expansion, or via direct computation.
# In short the Taylor coefficients give a Taylor expansion with which
# one can compute the SRK coefficients more precisely for a small h.


class AbstractCoeffs(eqx.Module):
    dtype: AbstractVar[jnp.dtype]


_Coeffs = TypeVar("_Coeffs", bound=AbstractCoeffs)


class SolverState(eqx.Module, Generic[_Coeffs]):
    gamma: LangevinX
    u: LangevinX
    h: RealScalarLike
    taylor_coeffs: PyTree[_Coeffs, "LangevinX"]
    coeffs: _Coeffs
    rho: LangevinX
    prev_f: LangevinX


class AbstractFosterLangevinSRK(
    AbstractStratonovichSolver[SolverState],
    AbstractItoSolver[SolverState],
    Generic[_Coeffs, _ErrorEstimate],
):
    r"""Abstract class for Stochastic Runge Kutta methods specifically designed
    for Underdamped Langevin Diffusion of the form

    \begin{align*}
        \mathrm{d} x(t) &= v(t) \, \mathrm{d}t \\
        \mathrm{d} v(t) &= - \gamma \, v(t) \, \mathrm{d}t - u \,
        \nabla \! f( x(t) ) \, \mathrm{d}t + \sqrt{2 \gamma u} \, \mathrm{d} w(t),
    \end{align*}

    where $x(t), v(t) \in \mathbb{R}^d$ represent the position
    and velocity, $w$ is a Brownian motion in $\mathbb{R}^d$,
    $f: \mathbb{R}^d \rightarrow \mathbb{R}$ is a potential function, and
    $\gamma , u \in \mathbb{R}^{d \times d}$ are diagonal matrices governing
    the friction and the inertia of the system.

    Solvers which inherit from this class include [`diffrax.ALIGN`][],
    [`diffrax.ShOULD`][], and [`diffrax.QUIC_SORT`][].
    """

    term_structure = MultiTerm[tuple[LangevinDriftTerm, LangevinDiffusionTerm]]
    interpolation_cls = LocalLinearInterpolation
    minimal_levy_area: eqx.AbstractClassVar[type[AbstractBrownianIncrement]]
    taylor_threshold: AbstractVar[RealScalarLike]

    @abc.abstractmethod
    def _directly_compute_coeffs_leaf(
        self, h: RealScalarLike, c: LangevinLeaf
    ) -> _Coeffs:
        r"""This method specifies how to compute the SRK coefficients directly
        (as opposed to via Taylor expansion). This function is then mapped over the
        PyTree of gamma to compute the coefficients for the entire system.

        **Arguments:**

        - `h`: The time increment.
        - `c`: A leaf of gamma.

        **Returns:**

        The SRK coefficients for the given leaf of gamma.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def _tay_coeffs_single(self, c: LangevinLeaf) -> _Coeffs:
        r"""This method specifies how to compute the Taylor coefficients for a
        single leaf of gamma. These coefficients are then used to compute the SRK
        coefficients using the Taylor expansion. This function is then mapped over
        the PyTree of gamma to compute the coefficients for the entire system.

        **Arguments:**

        - `c`: A leaf of gamma.

        **Returns:**

        The Taylor coefficients for the given leaf of gamma.
        """
        raise NotImplementedError

    @staticmethod
    def _eval_taylor(h: RealScalarLike, tay_coeffs: _Coeffs) -> _Coeffs:
        r"""Computes the SRK coefficients via the Taylor expansion, using the
        precomputed Taylor coefficients and the step-size h."""
        h = jnp.asarray(h, dtype=tay_coeffs.dtype)

        def eval_taylor_fun(tay_leaf):
            # jnp.polyval performs the Taylor expansion along
            # the first axis, so we need to move the trailing
            # axis to 0th position
            transposed = jnp.moveaxis(tay_leaf, -1, 0)
            assert transposed.shape[0] == 6
            return jnp.polyval(transposed, h, unroll=6)

        return jtu.tree_map(eval_taylor_fun, tay_coeffs)

    def _recompute_coeffs(
        self, h: RealScalarLike, gamma: LangevinX, tay_coeffs: PyTree[_Coeffs]
    ) -> _Coeffs:
        r"""When h changes, the SRK coefficients (which depend on h) are recomputed
        using this function."""
        # Inner will record the tree structure of the coefficients
        inner = sentinel = object()

        def recompute_coeffs_leaf(c: LangevinLeaf, _tay_coeffs: _Coeffs):
            # c is a leaf of gamma
            # Depending on the size of h*gamma choose whether the Taylor expansion or
            # direct computation is more accurate.
            cond = h * c < self.taylor_threshold
            tay_out = self._eval_taylor(h, _tay_coeffs)
            if cond.ndim < jtu.tree_leaves(tay_out)[0].ndim:
                # This happens when c is a scalar
                cond = jnp.expand_dims(cond, axis=-1)

            def select_tay_or_direct():
                if jnp.ndim(c) == 0:
                    direct_out = self._directly_compute_coeffs_leaf(h, c)
                else:
                    fun = lambda _c: self._directly_compute_coeffs_leaf(h, _c)
                    direct_out = vmap(fun)(c)

                def _choose(tay_leaf, direct_leaf):
                    assert tay_leaf.ndim == direct_leaf.ndim == cond.ndim, (
                        f"tay_leaf.ndim: {tay_leaf.ndim},"
                        f" direct_leaf.ndim: {direct_leaf.ndim},"
                        f" cond.ndim: {cond.ndim}"
                    )
                    return jnp.where(cond, tay_leaf, direct_leaf)

                return jtu.tree_map(_choose, tay_out, direct_out)

            # If all entries of h*gamma are below threshold, only compute tay_out
            # otherwise, compute both tay_out and direct_out and select the
            # correct one for each dimension
            out = lax.cond(
                eqxi.unvmap_all(cond),
                lambda: tay_out,
                select_tay_or_direct,
            )

            # The inner tree structure is just the structure of _Coeffs,
            # but we need to record it for the tree transpose
            nonlocal inner
            if inner is sentinel:
                inner = jtu.tree_structure(out)
            else:
                assert (
                    jtu.tree_structure(out) == inner
                ), f"Expected {inner}, got {jtu.tree_structure(out)}"

            return out

        tree_with_coeffs = jtu.tree_map(recompute_coeffs_leaf, gamma, tay_coeffs)
        outer = jtu.tree_structure(gamma)
        assert inner is not sentinel, "inner tree structure not set"
        coeffs_with_tree = jtu.tree_transpose(outer, inner, tree_with_coeffs)  # type: ignore
        return coeffs_with_tree

    def init(
        self,
        terms: MultiTerm[tuple[AbstractTerm[Any, RealScalarLike], AbstractTerm]],
        t0: RealScalarLike,
        t1: RealScalarLike,
        y0: LangevinTuple,
        args: PyTree,
    ) -> SolverState:
        """Precompute _SolverState which carries the Taylor coefficients and the
        SRK coefficients (which can be computed from h and the Taylor coefficients).
        Some solvers of this type are FSAL, so _SolverState also carries the previous
        evaluation of grad_f.
        """
        drift, diffusion = terms.terms
        gamma, u, grad_f = _get_args_from_terms(terms)

        h = drift.contr(t0, t1)
        x0, v0 = y0

        gamma = _broadcast_pytree(gamma, x0)
        u = _broadcast_pytree(u, x0)
        grad_f_shape = jax.eval_shape(grad_f, x0)

        def _shape_check_fun(_x, _g, _u, _fx):
            return _x.shape == _g.shape == _u.shape == _fx.shape

        assert jtu.tree_all(
            jtu.tree_map(_shape_check_fun, x0, gamma, u, grad_f_shape)
        ), "The shapes of gamma, u, and grad_f(x0) must be the same as x0."

        tay_coeffs = jtu.tree_map(self._tay_coeffs_single, gamma)
        # tay_coeffs have the same tree structure as gamma, with each leaf being a
        # _Coeffs object and the arrays inside have an extra trailing dimension of 6
        # (or in the case of QUICSORT either (3, 6) or (1, 6))

        coeffs = self._recompute_coeffs(h, gamma, tay_coeffs)
        rho = jtu.tree_map(lambda c, _u: jnp.sqrt(2 * c * _u), gamma, u)

        state_out = SolverState(
            gamma=gamma,
            u=u,
            h=h,
            taylor_coeffs=tay_coeffs,
            coeffs=coeffs,
            rho=rho,
            prev_f=grad_f(x0),
        )

        return state_out

    @abc.abstractmethod
    def _compute_step(
        self,
        h: RealScalarLike,
        levy,
        x0: LangevinX,
        v0: LangevinX,
        langevin_args: _LangevinArgs,
        coeffs: _Coeffs,
        rho: LangevinX,
        prev_f: LangevinX,
    ) -> tuple[LangevinX, LangevinX, LangevinX, _ErrorEstimate]:
        r"""This method specifies how to compute a single step of the Langevin SRK
        method. This holds just the computation that differs between the different
        SRK methods. The common bits are handled by the `step` method."""
        raise NotImplementedError

    def step(
        self,
        terms: MultiTerm[tuple[AbstractTerm[Any, RealScalarLike], AbstractTerm]],
        t0: RealScalarLike,
        t1: RealScalarLike,
        y0: LangevinTuple,
        args: PyTree,
        solver_state: SolverState,
        made_jump: BoolScalarLike,
    ) -> tuple[LangevinTuple, _ErrorEstimate, DenseInfo, SolverState, RESULTS]:
        del args
        st = solver_state
        drift, diffusion = terms.terms

        h = drift.contr(t0, t1)
        h_prev = st.h
        tay: PyTree[_Coeffs] = st.taylor_coeffs
        old_coeffs: _Coeffs = st.coeffs

        gamma, u, rho = st.gamma, st.u, st.rho
        _, _, grad_f = _get_args_from_terms(terms)

        # If h changed, recompute coefficients
        # Even when using constant step sizes, h can fluctuate by small amounts,
        # so we use `jnp.isclose` for comparison
        cond = jnp.isclose(h_prev, h, rtol=1e-10, atol=1e-12)
        coeffs = lax.cond(
            eqxi.unvmap_all(cond),
            lambda: old_coeffs,
            lambda: self._recompute_coeffs(h, gamma, tay),
        )

        # compute the Brownian increment and space-time(-time) Levy area
        levy = diffusion.contr(t0, t1, use_levy=True)
        assert isinstance(levy, self.minimal_levy_area), (
            f"The Brownian motion must have"
            f" `levy_area={self.minimal_levy_area.__name__}`"
        )

        x0, v0 = y0
        prev_f = lax.cond(made_jump, lambda: grad_f(x0), lambda: st.prev_f)

        # The actual step computation, handled by the subclass
        x_out, v_out, f_fsal, error = self._compute_step(
            h, levy, x0, v0, (gamma, u, grad_f), coeffs, rho, prev_f
        )

        def check_shapes_dtypes(_x, _v, _f, _x0):
            assert _x.dtype == _v.dtype == _f.dtype == _x0.dtype, (
                f"dtypes don't match. x0: {x0.dtype},"
                f" v_out: {_v.dtype}, x_out: {_x.dtype}, f_fsal: {_f.dtype}"
            )
            assert _x.shape == _v.shape == _f.shape == _x0.shape, (
                f"Shapes don't match. x0: {x0.shape},"
                f" v_out: {_v.shape}, x_out: {_x.shape}, f_fsal: {_f.shape}"
            )

        jtu.tree_map(check_shapes_dtypes, x_out, v_out, f_fsal, x0)

        y1 = (x_out, v_out)

        dense_info = dict(y0=y0, y1=y1)
        st = SolverState(
            gamma=gamma,
            u=u,
            h=h,
            taylor_coeffs=tay,
            coeffs=coeffs,
            rho=st.rho,
            prev_f=f_fsal,
        )
        return y1, error, dense_info, st, RESULTS.successful

    def func(
        self,
        terms: MultiTerm[tuple[AbstractTerm[Any, RealScalarLike], AbstractTerm]],
        t0: RealScalarLike,
        y0: LangevinTuple,
        args: PyTree,
    ):
        return terms.vf(t0, y0, args)
