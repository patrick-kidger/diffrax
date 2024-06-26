import math
from typing import TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
from equinox.internal import ω
from jaxtyping import Array, PyTree

from .._custom_types import (
    AbstractSpaceTimeTimeLevyArea,
    RealScalarLike,
)
from .._local_interpolation import LocalLinearInterpolation
from .._term import _LangevinArgs, LangevinTerm, LangevinX
from .langevin_srk import _AbstractCoeffs, _SolverState, AbstractLangevinSRK


# UBU evaluates at l = (3 -sqrt(3))/6, at r = (3 + sqrt(3))/6 and at 1,
# so we need 3 versions of each coefficient


class _UBU3Coeffs(_AbstractCoeffs):
    @property
    def dtype(self):
        return jtu.tree_leaves(self.beta_lr1)[0].dtype

    beta_lr1: PyTree[Array]  # (gamma, 3, *taylor)
    a_lr1: PyTree[Array]  # (gamma, 3, *taylor)
    b_lr1: PyTree[Array]  # (gamma, 3, *taylor)
    a_third: PyTree[Array]  # (gamma, 1, *taylor)
    a_div_h: PyTree[Array]  # (gamma, 1, *taylor)


_ErrorEstimate: TypeAlias = None


class UBU3(AbstractLangevinSRK[_UBU3Coeffs, _ErrorEstimate]):
    r"""The third order version of the UBU method by Daire O'Kane and James Foster.
    Works for underdamped Langevin SDEs of the form

    $$d x_t = v_t dt$$

    $$d v_t = - gamma v_t dt - u ∇f(x_t) dt + (2gammau)^(1/2) dW_t$$

    where $v$ is the velocity, $f$ is the potential, $gamma$ is the friction, and
    $W$ is a Brownian motion.
    """

    term_structure = LangevinTerm
    interpolation_cls = LocalLinearInterpolation
    taylor_threshold: RealScalarLike = eqx.field(static=True)
    _coeffs_structure = jtu.tree_structure(
        _UBU3Coeffs(
            beta_lr1=jnp.array(0.0),
            a_lr1=jnp.array(0.0),
            b_lr1=jnp.array(0.0),
            a_third=jnp.array(0.0),
            a_div_h=jnp.array(0.0),
        )
    )

    @property
    def minimal_levy_area(self):
        return AbstractSpaceTimeTimeLevyArea

    def __init__(self, taylor_threshold: RealScalarLike = 0.0):
        r"""**Arguments:**

        - `taylor_threshold`: If the product `h*gamma` is less than this, then
        the Taylor expansion will be used to compute the coefficients.
        Otherwise they will be computed directly. When using float32, the
        empirically optimal value is 0.1, and for float64 about 0.01.
        """
        self.taylor_threshold = taylor_threshold

    def order(self, terms):
        return 3

    def strong_order(self, terms):
        return 3.0

    @staticmethod
    def _directly_compute_coeffs_leaf(h, c) -> _UBU3Coeffs:
        # compute the coefficients directly (as opposed to via Taylor expansion)
        dtype = jnp.dtype(c)
        original_shape = c.shape
        c = jnp.expand_dims(c, axis=c.ndim)
        alpha = c * h
        l = 0.5 - math.sqrt(3) / 6
        r = 0.5 + math.sqrt(3) / 6
        l_r_1 = jnp.array([l, r, 1.0], dtype=jnp.dtype(c))
        alpha_lr1 = alpha * l_r_1
        assert alpha_lr1.shape == original_shape + (
            3,
        ), f"expected {original_shape + (3,)}, got {alpha_lr1.shape}"
        beta_lr1 = jnp.exp(-alpha_lr1)
        a_lr1 = (1.0 - beta_lr1) / c
        b_lr1 = (beta_lr1 + alpha_lr1 - 1.0) / (c**2 * h)
        a_third = (1.0 - jnp.exp(-alpha / 3)) / c
        a_div_h = (1.0 - jnp.exp(-alpha)) / (c * h)

        assert a_third.shape == a_div_h.shape == original_shape + (1,)

        out = _UBU3Coeffs(
            beta_lr1=beta_lr1,
            a_lr1=a_lr1,
            b_lr1=b_lr1,
            a_third=a_third,
            a_div_h=a_div_h,
        )
        return jtu.tree_map(lambda x: jnp.array(x, dtype=dtype), out)

    @staticmethod
    def _tay_cfs_single(c: Array) -> _UBU3Coeffs:
        # c is a leaf of gamma
        dtype = jnp.dtype(c)
        zero = jnp.zeros_like(c)
        one = jnp.ones_like(c)
        c2 = jnp.square(c)
        c3 = c2 * c
        c4 = c3 * c
        c5 = c4 * c

        l = 0.5 - math.sqrt(3) / 6
        r = 0.5 + math.sqrt(3) / 6
        lr1 = jnp.expand_dims(jnp.array([l, r, 1.0], dtype=dtype), axis=-1)
        exponents = jnp.expand_dims(jnp.arange(0, 6, dtype=dtype), axis=0)
        lr1_pows = jnp.power(lr1, exponents)
        assert lr1_pows.shape == (3, 6)

        beta = jnp.stack([one, -c, c2 / 2, -c3 / 6, c4 / 24, -c5 / 120], axis=-1)
        a = jnp.stack([zero, one, -c / 2, c2 / 6, -c3 / 24, c4 / 120], axis=-1)
        b = jnp.stack([zero, one / 2, -c / 6, c2 / 24, -c3 / 120, c4 / 720], axis=-1)

        with jax.numpy_rank_promotion("allow"):
            beta_lr1 = lr1_pows * jnp.expand_dims(beta, axis=c.ndim)
            a_lr1 = lr1_pows * jnp.expand_dims(a, axis=c.ndim)
            # b needs an extra power of l and r
            b_lr1 = lr1_pows * lr1 * jnp.expand_dims(b, axis=c.ndim)
        assert beta_lr1.shape == a_lr1.shape == b_lr1.shape == c.shape + (3, 6)

        # a_third = (1 - exp(-1/3 * gamma * h))/gamma
        a_third = jnp.stack(
            [zero, one / 3, -c / 18, c2 / 162, -c3 / 1944, c4 / 29160], axis=-1
        )
        a_third = jnp.expand_dims(a_third, axis=c.ndim)
        a_div_h = jnp.stack(
            [one, -c / 2, c2 / 6, -c3 / 24, c4 / 120, -c5 / 720], axis=-1
        )
        a_div_h = jnp.expand_dims(a_div_h, axis=c.ndim)
        assert a_third.shape == a_div_h.shape == c.shape + (1, 6)

        out = _UBU3Coeffs(
            beta_lr1=beta_lr1,
            a_lr1=a_lr1,
            b_lr1=b_lr1,
            a_third=a_third,
            a_div_h=a_div_h,
        )
        return jtu.tree_map(lambda x: jnp.array(x, dtype=dtype), out)

    @staticmethod
    def _compute_step(
        h: RealScalarLike,
        levy: AbstractSpaceTimeTimeLevyArea,
        x0: LangevinX,
        v0: LangevinX,
        langevin_args: _LangevinArgs,
        cfs: _UBU3Coeffs,
        st: _SolverState,
    ) -> tuple[LangevinX, LangevinX, LangevinX, _ErrorEstimate]:
        w: LangevinX = levy.W
        hh: LangevinX = levy.H
        kk: LangevinX = levy.K

        gamma, u, f = langevin_args

        def _l(coeff):
            return jtu.tree_map(lambda arr: arr[..., 0], coeff)

        def _r(coeff):
            return jtu.tree_map(lambda arr: arr[..., 1], coeff)

        def _one(coeff):
            return jtu.tree_map(lambda arr: arr[..., 2], coeff)

        beta_l = _l(cfs.beta_lr1)
        beta_r = _r(cfs.beta_lr1)
        beta_1 = _one(cfs.beta_lr1)
        a_l = _l(cfs.a_lr1)
        a_r = _r(cfs.a_lr1)
        a_1 = _one(cfs.a_lr1)
        b_l = _l(cfs.b_lr1)
        b_r = _r(cfs.b_lr1)
        b_1 = _one(cfs.b_lr1)
        a_third = _l(cfs.a_third)
        a_div_h = _l(cfs.a_div_h)

        rho_w_k = (st.rho**ω * (w**ω - 12 * kk**ω)).ω
        uh = (u**ω * h).ω
        v_tilde = (v0**ω + st.rho**ω * (hh**ω + 6 * kk**ω)).ω

        x1 = (x0**ω + a_l**ω * v_tilde**ω + b_l**ω * rho_w_k**ω).ω
        f1uh = (f(x1) ** ω * uh**ω).ω

        x2 = (
            x0**ω + a_r**ω * v_tilde**ω + b_r**ω * rho_w_k**ω - a_third**ω * f1uh**ω
        ).ω
        f2uh = (f(x2) ** ω * uh**ω).ω

        x_out = (
            x0**ω
            + a_1**ω * v_tilde**ω
            + b_1**ω * rho_w_k**ω
            - 0.5 * (a_r**ω * f1uh**ω + a_l**ω * f2uh**ω)
        ).ω

        v_out_tilde = (
            beta_1**ω * v_tilde**ω
            - 0.5 * (beta_r**ω * f1uh**ω + beta_l**ω * f2uh**ω)
            + a_div_h**ω * rho_w_k**ω
        ).ω
        v_out = (v_out_tilde**ω - st.rho**ω * (hh**ω - 6 * kk**ω)).ω

        f_fsal = (
            st.prev_f
        )  # this method is not FSAL, but this is for compatibility with the base class

        # TODO: compute error estimate
        return x_out, v_out, f_fsal, None
