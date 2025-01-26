import abc
from collections.abc import Callable
from typing import Any, Generic, Optional, TypeVar

import equinox as eqx
import equinox.internal as eqxi
import jax
import jax.lax as lax
import jax.numpy as jnp
import jax.tree_util as jtu
from equinox import AbstractVar
from jaxtyping import PyTree

from .._custom_types import (
    AbstractBrownianIncrement,
    Args,
    BoolScalarLike,
    DenseInfo,
    RealScalarLike,
)
from .._local_interpolation import LocalLinearInterpolation
from .._solution import RESULTS
from .._term import (
    AbstractTerm,
    broadcast_underdamped_langevin_arg,
    MultiTerm,
    UnderdampedLangevinDiffusionTerm,
    UnderdampedLangevinDriftTerm,
    UnderdampedLangevinLeaf,
    UnderdampedLangevinTuple,
    UnderdampedLangevinX,
    WrapTerm,
)
from .base import AbstractStratonovichSolver


_ErrorEstimate = TypeVar("_ErrorEstimate", None, UnderdampedLangevinTuple)
UnderdampedLangevinArgs = tuple[
    UnderdampedLangevinX,
    UnderdampedLangevinX,
    Callable[[UnderdampedLangevinX, Args], UnderdampedLangevinX],
]


def _get_args_from_terms(
    terms: MultiTerm[tuple[AbstractTerm[Any, RealScalarLike], AbstractTerm]],
) -> tuple[
    PyTree,
    PyTree,
    PyTree,
    PyTree,
    Callable[[UnderdampedLangevinX, Args], UnderdampedLangevinX],
]:
    drift, diffusion = terms.terms
    if isinstance(drift, WrapTerm):
        assert isinstance(diffusion, WrapTerm)
        drift = drift.term
        diffusion = diffusion.term

    assert isinstance(drift, UnderdampedLangevinDriftTerm)
    assert isinstance(diffusion, UnderdampedLangevinDiffusionTerm)
    gamma_drift = drift.gamma
    u_drift = drift.u
    f = drift.grad_f
    gamma_diffusion = diffusion.gamma
    u_diffusion = diffusion.u
    return gamma_drift, u_drift, gamma_diffusion, u_diffusion, f


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
    gamma: UnderdampedLangevinX
    u: UnderdampedLangevinX
    h: RealScalarLike
    taylor_coeffs: PyTree[_Coeffs, "UnderdampedLangevinX"]
    coeffs: _Coeffs
    rho: UnderdampedLangevinX
    prev_f: Optional[UnderdampedLangevinX]


class AbstractFosterLangevinSRK(
    AbstractStratonovichSolver[SolverState],
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
    the friction and the damping of the system.

    Solvers which inherit from this class include [`diffrax.ALIGN`][],
    [`diffrax.ShOULD`][], and [`diffrax.QUICSORT`][].
    """

    term_structure = MultiTerm[
        tuple[UnderdampedLangevinDriftTerm, UnderdampedLangevinDiffusionTerm]
    ]
    interpolation_cls = LocalLinearInterpolation
    minimal_levy_area: eqx.AbstractClassVar[type[AbstractBrownianIncrement]]
    taylor_threshold: AbstractVar[RealScalarLike]
    _is_fsal: eqx.AbstractClassVar[bool]

    @abc.abstractmethod
    def _directly_compute_coeffs_leaf(
        self, h: RealScalarLike, c: UnderdampedLangevinLeaf
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
    def _tay_coeffs_single(self, c: UnderdampedLangevinLeaf) -> _Coeffs:
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
        self,
        h: RealScalarLike,
        gamma: UnderdampedLangevinX,
        tay_coeffs: PyTree[_Coeffs],
    ) -> _Coeffs:
        r"""When h changes, the SRK coefficients (which depend on h) are recomputed
        using this function."""
        # Inner will record the tree structure of the coefficients
        inner = sentinel = object()

        def recompute_coeffs_leaf(c: UnderdampedLangevinLeaf, _tay_coeffs: _Coeffs):
            # c is a leaf of gamma
            # Depending on the size of h*gamma choose whether the Taylor expansion or
            # direct computation is more accurate.
            cond = h * c < self.taylor_threshold
            tay_out = self._eval_taylor(h, _tay_coeffs)
            if cond.ndim < jtu.tree_leaves(tay_out)[0].ndim:
                # This happens when c is a scalar
                cond = jnp.expand_dims(cond, axis=-1)

            def select_tay_or_direct():
                direct_out = self._directly_compute_coeffs_leaf(h, c)

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
        coeffs_with_tree = jtu.tree_transpose(outer, inner, tree_with_coeffs)  # pyright: ignore
        return coeffs_with_tree

    def init(
        self,
        terms: MultiTerm[tuple[AbstractTerm[Any, RealScalarLike], AbstractTerm]],
        t0: RealScalarLike,
        t1: RealScalarLike,
        y0: UnderdampedLangevinTuple,
        args: PyTree,
    ) -> SolverState:
        """Precompute _SolverState which carries the Taylor coefficients and the
        SRK coefficients (which can be computed from h and the Taylor coefficients).
        Some solvers of this type are FSAL, so _SolverState also carries the previous
        evaluation of grad_f.
        """
        drift, diffusion = terms.terms
        del diffusion
        (
            gamma_drift,
            u_drift,
            gamma_diffusion,
            u_diffusion,
            grad_f,
        ) = _get_args_from_terms(terms)

        h = drift.contr(t0, t1)
        x0, v0 = y0
        del v0

        gamma = broadcast_underdamped_langevin_arg(gamma_drift, x0, "gamma")
        u = broadcast_underdamped_langevin_arg(u_drift, x0, "u")

        # Check that drift and diffusion have the same arguments
        gamma_diffusion = broadcast_underdamped_langevin_arg(
            gamma_diffusion, x0, "gamma"
        )
        u_diffusion = broadcast_underdamped_langevin_arg(u_diffusion, x0, "u")

        def compare_args_fun(arg1, arg2):
            arg = eqx.error_if(
                arg1,
                jnp.any(arg1 != arg2),
                "The arguments of the drift and diffusion terms must match.",
            )
            return arg

        gamma = jtu.tree_map(compare_args_fun, gamma, gamma_diffusion)
        u = jtu.tree_map(compare_args_fun, u, u_diffusion)

        try:
            grad_f_shape = jax.eval_shape(grad_f, x0, args)
        except ValueError:
            raise RuntimeError(
                "The function `grad_f` in the Underdamped Langevin term must be"
                " a callable, whose input and output have the same PyTree structure"
                " and shapes as the position `x`."
            )

        def shape_check_fun(_x, _g, _u, _fx):
            return _x.shape == _g.shape == _u.shape == _fx.shape

        if not jtu.tree_all(jtu.tree_map(shape_check_fun, x0, gamma, u, grad_f_shape)):
            raise RuntimeError(
                "The shapes and PyTree structures of x0, gamma, u, and grad_f(x0, args)"
                " must match."
            )

        tay_coeffs = jtu.tree_map(self._tay_coeffs_single, gamma)
        # tay_coeffs have the same tree structure as gamma, with each leaf being a
        # _Coeffs object and the arrays inside have an extra trailing dimension of 6
        # (or in the case of QUICSORT either (3, 6) or (1, 6))

        coeffs = self._recompute_coeffs(h, gamma, tay_coeffs)
        rho = jtu.tree_map(lambda c, _u: jnp.sqrt(2 * c * _u), gamma, u)
        prev_f = grad_f(x0, args) if self._is_fsal else None

        state_out = SolverState(
            gamma=gamma,
            u=u,
            h=h,
            taylor_coeffs=tay_coeffs,
            coeffs=coeffs,
            rho=rho,
            prev_f=prev_f,
        )

        return state_out

    @abc.abstractmethod
    def _compute_step(
        self,
        h: RealScalarLike,
        levy,
        x0: UnderdampedLangevinX,
        v0: UnderdampedLangevinX,
        underdamped_langevin_args: UnderdampedLangevinArgs,
        coeffs: _Coeffs,
        rho: UnderdampedLangevinX,
        prev_f: Optional[UnderdampedLangevinX],
        args: Args,
    ) -> tuple[
        UnderdampedLangevinX,
        UnderdampedLangevinX,
        Optional[UnderdampedLangevinX],
        _ErrorEstimate,
    ]:
        r"""This method specifies how to compute a single step of the Underdamped
        Langevin SRK method.
        This holds just the computation that differs between the different
        SRK methods. The common bits are handled by the `step` method.

        **Returns:**

        (x_out, v_out, f_fsal, error), where:

        - `x_out` and `v_out` are the new position and velocity.
        - `f_fsal` is the new evaluation of the gradient of the potential function.
        - `error` is the error estimate.
        """
        raise NotImplementedError

    def step(
        self,
        terms: MultiTerm[tuple[AbstractTerm[Any, RealScalarLike], AbstractTerm]],
        t0: RealScalarLike,
        t1: RealScalarLike,
        y0: UnderdampedLangevinTuple,
        args: PyTree,
        solver_state: SolverState,
        made_jump: BoolScalarLike,
    ) -> tuple[
        UnderdampedLangevinTuple, _ErrorEstimate, DenseInfo, SolverState, RESULTS
    ]:
        st = solver_state
        drift, diffusion = terms.terms

        h = drift.contr(t0, t1)
        h_prev = st.h
        tay: PyTree[_Coeffs] = st.taylor_coeffs
        old_coeffs: _Coeffs = st.coeffs

        gamma, u, rho = st.gamma, st.u, st.rho
        _, _, _, _, grad_f = _get_args_from_terms(terms)

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
        if not isinstance(levy, self.minimal_levy_area):
            raise ValueError(
                f"The Brownian motion must have"
                f" `levy_area={self.minimal_levy_area.__name__}`"
            )

        x0, v0 = y0
        if made_jump is False:
            prev_f = st.prev_f
        else:
            prev_f = lax.cond(
                eqxi.unvmap_any(made_jump), lambda: grad_f(x0, args), lambda: st.prev_f
            )

        # The actual step computation, handled by the subclass
        x_out, v_out, f_fsal, error = self._compute_step(
            h, levy, x0, v0, (gamma, u, grad_f), coeffs, rho, prev_f, args
        )

        def check_shapes_dtypes(arg, *args):
            for x in args:
                assert x.shape == arg.shape
                assert x.dtype == arg.dtype

        # Some children classes may not use f_fsal, so we allow it to be None
        if self._is_fsal:
            jtu.tree_map(check_shapes_dtypes, x_out, v_out, f_fsal, x0)
        else:
            assert f_fsal is None
            jtu.tree_map(check_shapes_dtypes, x_out, v_out, x0)

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
        y0: UnderdampedLangevinTuple,
        args: PyTree,
    ):
        return terms.vf(t0, y0, args)
