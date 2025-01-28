import math
from typing import Optional

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
from equinox.internal import scan_trick, ω
from jaxtyping import ArrayLike, PyTree

from .._custom_types import (
    AbstractSpaceTimeTimeLevyArea,
    Args,
    RealScalarLike,
)
from .._local_interpolation import LocalLinearInterpolation
from .._term import UnderdampedLangevinLeaf, UnderdampedLangevinX
from .foster_langevin_srk import (
    AbstractCoeffs,
    AbstractFosterLangevinSRK,
    UnderdampedLangevinArgs,
)


# For an explanation of the coefficients, see foster_langevin_srk.py
# UBU evaluates at l = (3 -sqrt(3))/6, at r = (3 + sqrt(3))/6 and at 1,
# so we need 3 versions of each coefficient
class _QUICSORTCoeffs(AbstractCoeffs):
    beta_lr1: PyTree[ArrayLike]  # gamma.shape + (3, *taylor)
    a_lr1: PyTree[ArrayLike]  # gamma.shape + (3, *taylor)
    b_lr1: PyTree[ArrayLike]  # gamma.shape + (3, *taylor)
    a_third: PyTree[ArrayLike]  # gamma.shape + (1, *taylor)
    a_div_h: PyTree[ArrayLike]  # gamma.shape + (1, *taylor)
    dtype: jnp.dtype = eqx.field(static=True)

    def __init__(self, beta_lr1, a_lr1, b_lr1, a_third, a_div_h):
        self.beta_lr1 = beta_lr1
        self.a_lr1 = a_lr1
        self.b_lr1 = b_lr1
        self.a_third = a_third
        self.a_div_h = a_div_h
        all_leaves = jtu.tree_leaves(
            [self.beta_lr1, self.a_lr1, self.b_lr1, self.a_third, self.a_div_h]
        )
        self.dtype = jnp.result_type(*all_leaves)


class QUICSORT(AbstractFosterLangevinSRK[_QUICSORTCoeffs, None]):
    r"""The QUadrature Inspired and Contractive Shifted ODE with Runge-Kutta Three
    method by James Foster and Daire O'Kane. This is a third order solver for the
    Underdamped Langevin Diffusion, and accepts terms of the form
    `MultiTerm(UnderdampedLangevinDriftTerm, UnderdampedLangevinDiffusionTerm)`.
    Uses two evaluations of the vector field per step.

    ??? cite "Reference"

        This is a variant of the SORT method from Definition 1.2 of

        ```bibtex
        @misc{foster2021shiftedode,
            title={The shifted ODE method for underdamped Langevin MCMC},
            author={James Foster and Terry Lyons and Harald Oberhauser},
            year={2021},
            eprint={2101.03446},
            archivePrefix={arXiv},
            primaryClass={math.NA},
            url={https://arxiv.org/abs/2101.03446},
        }
        ```

        with the modifications inspired by the UBU method from

        ```bibtex
        @misc{chada2024unbiased,
            title={Unbiased Kinetic Langevin Monte Carlo with Inexact Gradients},
            author={Neil K. Chada and Benedict Leimkuhler and Daniel Paulin
                and Peter A. Whalley},
            year={2024},
            eprint={2311.05025},
            archivePrefix={arXiv},
            primaryClass={stat.CO},
            url={https://arxiv.org/abs/2311.05025},
        }
        ```
    """

    interpolation_cls = LocalLinearInterpolation
    minimal_levy_area = AbstractSpaceTimeTimeLevyArea
    taylor_threshold: float = eqx.field(static=True)
    _is_fsal = False

    def __init__(self, taylor_threshold: float = 0.1):
        r"""**Arguments:**

        - `taylor_threshold`: If the product `h*gamma` is less than this, then
        the Taylor expansion will be used to compute the coefficients.
        Otherwise they will be computed directly. When using float32, the
        empirically optimal value is 0.1, and for float64 about 0.01.
        """
        self.taylor_threshold = taylor_threshold

    def order(self, terms):
        del terms
        return 3

    def strong_order(self, terms):
        del terms
        return 3.0

    def _directly_compute_coeffs_leaf(
        self, h: RealScalarLike, c: UnderdampedLangevinLeaf
    ) -> _QUICSORTCoeffs:
        del self
        # compute the coefficients directly (as opposed to via Taylor expansion)
        original_shape = jnp.shape(c)
        c = jnp.expand_dims(c, axis=c.ndim)
        alpha = c * h
        l = 0.5 - math.sqrt(3) / 6
        r = 0.5 + math.sqrt(3) / 6
        l_r_1 = jnp.array([l, r, 1.0], dtype=jnp.result_type(c))
        l_r_1 = jnp.broadcast_to(l_r_1, original_shape + (3,))
        alpha_lr1 = alpha * l_r_1
        assert alpha_lr1.shape == original_shape + (3,)
        beta_lr1 = jnp.exp(-alpha_lr1)
        a_lr1 = (1.0 - beta_lr1) / c
        b_lr1 = (beta_lr1 + alpha_lr1 - 1.0) / (c**2 * h)
        a_third = (1.0 - jnp.exp(-alpha / 3)) / c
        a_div_h = (1.0 - jnp.exp(-alpha)) / (c * h)

        assert a_third.shape == a_div_h.shape == original_shape + (1,)

        return _QUICSORTCoeffs(
            beta_lr1=beta_lr1,
            a_lr1=a_lr1,
            b_lr1=b_lr1,
            a_third=a_third,
            a_div_h=a_div_h,
        )

    def _tay_coeffs_single(self, c: UnderdampedLangevinLeaf) -> _QUICSORTCoeffs:
        del self
        # c is a leaf of gamma
        dtype = jnp.result_type(c)
        zero = jnp.zeros_like(c)
        one = jnp.ones_like(c)
        c2 = jnp.square(c)
        c3 = c2 * c
        c4 = c3 * c
        c5 = c4 * c

        # Compute the coefficients of the Taylor expansion, starting from 5th power
        # to 0th power. The descending power order is because of jnp.polyval

        # Some coefficients must be computed at h*l, h*r and h*1, so we
        # pre-multiply the coefficients with powers of l, r and 1.
        l = 0.5 - math.sqrt(3) / 6
        r = 0.5 + math.sqrt(3) / 6
        lr1 = jnp.expand_dims(jnp.array([l, r, 1.0], dtype=dtype), axis=-1)
        exponents = jnp.expand_dims(jnp.arange(5, -1, step=-1, dtype=dtype), axis=0)
        lr1_pows = jnp.power(lr1, exponents)
        assert lr1_pows.shape == (3, 6)

        beta = jnp.stack([-c5 / 120, c4 / 24, -c3 / 6, c2 / 2, -c, one], axis=-1)
        a = jnp.stack([c4 / 120, -c3 / 24, c2 / 6, -c / 2, one, zero], axis=-1)
        b = jnp.stack([c4 / 720, -c3 / 120, c2 / 24, -c / 6, one / 2, zero], axis=-1)

        with jax.numpy_rank_promotion("allow"):
            beta_lr1 = lr1_pows * jnp.expand_dims(beta, axis=c.ndim)
            a_lr1 = lr1_pows * jnp.expand_dims(a, axis=c.ndim)
            # b needs an extra power of lr1 (just work out the expansion to see why)
            b_lr1 = lr1_pows * lr1 * jnp.expand_dims(b, axis=c.ndim)
        assert beta_lr1.shape == a_lr1.shape == b_lr1.shape == jnp.shape(c) + (3, 6)

        # a_third = (1 - exp(-1/3 * gamma * h))/gamma
        a_third = jnp.stack(
            [c4 / 29160, -c3 / 1944, c2 / 162, -c / 18, one / 3, zero], axis=-1
        )
        a_third = jnp.expand_dims(a_third, axis=c.ndim)
        a_div_h = jnp.stack(
            [-c5 / 720, c4 / 120, -c3 / 24, c2 / 6, -c / 2, one], axis=-1
        )
        a_div_h = jnp.expand_dims(a_div_h, axis=c.ndim)
        assert a_third.shape == a_div_h.shape == jnp.shape(c) + (1, 6)

        return _QUICSORTCoeffs(
            beta_lr1=beta_lr1,
            a_lr1=a_lr1,
            b_lr1=b_lr1,
            a_third=a_third,
            a_div_h=a_div_h,
        )

    def _compute_step(
        self,
        h: RealScalarLike,
        levy: AbstractSpaceTimeTimeLevyArea,
        x0: UnderdampedLangevinX,
        v0: UnderdampedLangevinX,
        underdamped_langevin_args: UnderdampedLangevinArgs,
        coeffs: _QUICSORTCoeffs,
        rho: UnderdampedLangevinX,
        prev_f: Optional[UnderdampedLangevinX],
        args: Args,
    ) -> tuple[UnderdampedLangevinX, UnderdampedLangevinX, None, None]:
        del prev_f
        dtypes = jtu.tree_map(jnp.result_type, x0)
        w: UnderdampedLangevinX = jtu.tree_map(jnp.asarray, levy.W, dtypes)
        hh: UnderdampedLangevinX = jtu.tree_map(jnp.asarray, levy.H, dtypes)
        kk: UnderdampedLangevinX = jtu.tree_map(jnp.asarray, levy.K, dtypes)

        gamma, u, f = underdamped_langevin_args

        def _extract_coeffs(coeff, index):
            return jtu.tree_map(lambda arr: arr[..., index], coeff)

        beta_l = _extract_coeffs(coeffs.beta_lr1, 0)
        beta_r = _extract_coeffs(coeffs.beta_lr1, 1)
        beta_1 = _extract_coeffs(coeffs.beta_lr1, 2)
        a_l = _extract_coeffs(coeffs.a_lr1, 0)
        a_r = _extract_coeffs(coeffs.a_lr1, 1)
        a_1 = _extract_coeffs(coeffs.a_lr1, 2)
        b_l = _extract_coeffs(coeffs.b_lr1, 0)
        b_r = _extract_coeffs(coeffs.b_lr1, 1)
        b_1 = _extract_coeffs(coeffs.b_lr1, 2)
        a_third = _extract_coeffs(coeffs.a_third, 0)
        a_div_h = _extract_coeffs(coeffs.a_div_h, 0)

        rho_w_k = (rho**ω * (w**ω - 12 * kk**ω)).ω
        uh = (u**ω * h).ω
        v_tilde = (v0**ω + rho**ω * (hh**ω + 6 * kk**ω)).ω

        x1 = (x0**ω + a_l**ω * v_tilde**ω + b_l**ω * rho_w_k**ω).ω

        # Use eqinox.internal.scan_trick to compute f1, x2 and f2 in one go
        # carry = x, f1, f2. We use x0 as the initial value for f1 and f2
        init = x1, x0, x0

        def fn(carry):
            x, _f, _ = carry
            fx_uh = (f(x, args) ** ω * uh**ω).ω
            return x, _f, fx_uh

        def compute_x2(carry):
            _, _, f1 = carry
            x = (
                x0**ω + a_r**ω * v_tilde**ω + b_r**ω * rho_w_k**ω - a_third**ω * f1**ω
            ).ω
            return x, f1, f1

        x2, f1uh, f2uh = scan_trick(fn, [compute_x2], init)

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
        v_out = (v_out_tilde**ω - rho**ω * (hh**ω - 6 * kk**ω)).ω

        # TODO: compute error estimate
        return x_out, v_out, None, None
