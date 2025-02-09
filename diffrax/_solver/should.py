import equinox as eqx
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
class _ShOULDCoeffs(AbstractCoeffs):
    beta_half: PyTree[ArrayLike]
    a_half: PyTree[ArrayLike]
    b_half: PyTree[ArrayLike]
    beta1: PyTree[ArrayLike]
    a1: PyTree[ArrayLike]
    b1: PyTree[ArrayLike]
    aa: PyTree[ArrayLike]
    chh: PyTree[ArrayLike]
    ckk: PyTree[ArrayLike]
    dtype: jnp.dtype = eqx.field(static=True)

    def __init__(self, beta_half, a_half, b_half, beta1, a1, b1, aa, chh, ckk):
        self.beta_half = beta_half
        self.a_half = a_half
        self.b_half = b_half
        self.beta1 = beta1
        self.a1 = a1
        self.b1 = b1
        self.aa = aa
        self.chh = chh
        self.ckk = ckk
        all_leaves = jtu.tree_leaves(
            [
                self.beta_half,
                self.a_half,
                self.b_half,
                self.beta1,
                self.a1,
                self.b1,
                self.aa,
                self.chh,
                self.ckk,
            ]
        )
        self.dtype = jnp.result_type(*all_leaves)


class ShOULD(AbstractFosterLangevinSRK[_ShOULDCoeffs, None]):
    r"""The Shifted-ODE Runge-Kutta Three method
    designed by James Foster. This is a third order solver for the
    Underdamped Langevin Diffusion, the terms of the form
    `MultiTerm(UnderdampedLangevinDriftTerm, UnderdampedLangevinDiffusionTerm)`.
    Uses three evaluations of the vector
    field per step, but is FSAL, so in practice it only requires two.

    ??? cite "Reference"

        This solver is based on Definition 7.1 from

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
    """

    interpolation_cls = LocalLinearInterpolation
    minimal_levy_area = AbstractSpaceTimeTimeLevyArea
    taylor_threshold: RealScalarLike = eqx.field(static=True)
    _is_fsal = True

    def __init__(self, taylor_threshold: RealScalarLike = 0.1):
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

    def _directly_compute_coeffs_leaf(
        self, h: RealScalarLike, c: UnderdampedLangevinLeaf
    ) -> _ShOULDCoeffs:
        del self
        # c is a leaf of gamma
        # compute the coefficients directly (as opposed to via Taylor expansion)
        al = c * h
        beta_half = jnp.exp(-al / 2)
        beta1 = jnp.exp(-al)
        a_half = (1 - beta_half) / c
        a1 = (1 - beta1) / c
        b_half = (beta_half + al / 2 - 1) / (al * c)
        b1 = (beta1 + al - 1) / (al * c)
        aa = a1 / h

        al2 = al**2
        al3 = al2 * al
        chh = 6 * (beta1 * (al + 2) + al - 2) / (al2 * c)
        ckk = 60 * (beta1 * (al * (al + 6) + 12) - al * (al - 6) - 12) / (al3 * c)

        return _ShOULDCoeffs(
            beta_half=beta_half,
            a_half=a_half,
            b_half=b_half,
            beta1=beta1,
            a1=a1,
            b1=b1,
            aa=aa,
            chh=chh,
            ckk=ckk,
        )

    def _tay_coeffs_single(self, c: UnderdampedLangevinLeaf) -> _ShOULDCoeffs:
        del self
        # c is a leaf of gamma
        zero = jnp.zeros_like(c)
        one = jnp.ones_like(c)
        c2 = jnp.square(c)
        c3 = c2 * c
        c4 = c3 * c
        c5 = c4 * c

        # Coefficients of the Taylor expansion, starting from 5th power
        # to 0th power. The descending power order is because of jnp.polyval
        beta_half = jnp.stack(
            [-c5 / 3840, c4 / 384, -c3 / 48, c2 / 8, -c / 2, one], axis=-1
        )
        beta1 = jnp.stack([-c5 / 120, c4 / 24, -c3 / 6, c2 / 2, -c, one], axis=-1)
        a_half = jnp.stack(
            [c4 / 3840, -c3 / 384, c2 / 48, -c / 8, one / 2, zero], axis=-1
        )
        a1 = jnp.stack([c4 / 120, -c3 / 24, c2 / 6, -c / 2, one, zero], axis=-1)
        aa = jnp.stack([-c5 / 720, c4 / 120, -c3 / 24, c2 / 6, -c / 2, one], axis=-1)
        b_half = jnp.stack(
            [c4 / 46080, -c3 / 3840, c2 / 384, -c / 48, one / 8, zero], axis=-1
        )
        b1 = jnp.stack([c4 / 720, -c3 / 120, c2 / 24, -c / 6, one / 2, zero], axis=-1)
        chh = jnp.stack([c4 / 168, -c3 / 30, 3 * c2 / 20, -c / 2, one, zero], axis=-1)
        ckk = jnp.stack([5 * c4 / 168, -c3 / 7, c2 / 2, -c, zero, zero], axis=-1)

        correct_shape = jnp.shape(c) + (6,)
        assert (
            beta_half.shape
            == a_half.shape
            == b_half.shape
            == beta1.shape
            == a1.shape
            == b1.shape
            == aa.shape
            == chh.shape
            == ckk.shape
            == correct_shape
        )

        return _ShOULDCoeffs(
            beta_half=beta_half,
            a_half=a_half,
            b_half=b_half,
            beta1=beta1,
            a1=a1,
            b1=b1,
            aa=aa,
            chh=chh,
            ckk=ckk,
        )

    def _compute_step(
        self,
        h: RealScalarLike,
        levy: AbstractSpaceTimeTimeLevyArea,
        x0: UnderdampedLangevinX,
        v0: UnderdampedLangevinX,
        underdamped_langevin_args: UnderdampedLangevinArgs,
        coeffs: _ShOULDCoeffs,
        rho: UnderdampedLangevinX,
        prev_f: UnderdampedLangevinX,
        args: Args,
    ) -> tuple[UnderdampedLangevinX, UnderdampedLangevinX, UnderdampedLangevinX, None]:
        dtypes = jtu.tree_map(jnp.result_type, x0)
        w: UnderdampedLangevinX = jtu.tree_map(jnp.asarray, levy.W, dtypes)
        hh: UnderdampedLangevinX = jtu.tree_map(jnp.asarray, levy.H, dtypes)
        kk: UnderdampedLangevinX = jtu.tree_map(jnp.asarray, levy.K, dtypes)

        chh_hh_plus_ckk_kk = (coeffs.chh**ω * hh**ω + coeffs.ckk**ω * kk**ω).ω

        gamma, u, f = underdamped_langevin_args

        rho_w_k = (rho**ω * (w**ω - 12 * kk**ω)).ω
        uh = (u**ω * h).ω

        f0 = prev_f
        v1 = (v0**ω + rho**ω * (hh**ω + 6 * kk**ω)).ω
        x1 = (
            x0**ω
            + coeffs.a_half**ω * v1**ω
            + coeffs.b_half**ω * (-(uh**ω) * f0**ω + rho_w_k**ω)
        ).ω

        # Use equinox.internal.scan_trick to compute f1, x_out and f_out in one go
        # carry = x, f1, f2. We use x0 as the initial value for f1 and f2
        init = x1, x0, x0

        def fn(carry):
            x, _f, _ = carry
            fx = f(x, args)
            return x, _f, fx

        def compute_x2(carry):
            _, _, _f1 = carry
            x = (
                x0**ω
                + coeffs.a1**ω * v0**ω
                - uh**ω * coeffs.b1**ω * (1 / 3 * f0**ω + 2 / 3 * _f1**ω)
                + rho**ω * (coeffs.b1**ω * w**ω + chh_hh_plus_ckk_kk**ω)
            ).ω
            return x, _f1, _f1

        x_out, f1, f_out = scan_trick(fn, [compute_x2], init)

        v_out = (
            coeffs.beta1**ω * v0**ω
            - uh**ω
            * (
                coeffs.beta1**ω / 6 * f0**ω
                + 2 / 3 * coeffs.beta_half**ω * f1**ω
                + 1 / 6 * f_out**ω
            )
            + rho**ω * (coeffs.aa**ω * w**ω - gamma**ω * chh_hh_plus_ckk_kk**ω)
        ).ω

        return x_out, v_out, f_out, None
