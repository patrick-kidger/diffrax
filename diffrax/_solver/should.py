from typing import TypeAlias

import equinox as eqx
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


class _ShOULDCoeffs(_AbstractCoeffs):
    @property
    def dtype(self):
        return jtu.tree_leaves(self.beta1)[0].dtype

    beta_half: PyTree[Array]
    a_half: PyTree[Array]
    b_half: PyTree[Array]
    beta1: PyTree[Array]
    a1: PyTree[Array]
    b1: PyTree[Array]
    aa: PyTree[Array]
    chh: PyTree[Array]
    ckk: PyTree[Array]


_ErrorEstimate: TypeAlias = None


class ShOULD(AbstractLangevinSRK[_ShOULDCoeffs, _ErrorEstimate]):
    r"""The Shifted-ODE Runge-Kutta Three method
    designed by James Foster. Only works for Underdamped Langevin Diffusion
    of the form

    $$d x_t = v_t dt$$

    $$d v_t = - gamma v_t dt - u ∇f(x_t) dt + (2gammau)^(1/2) dW_t$$

    where $v$ is the velocity, $f$ is the potential, $gamma$ is the friction, and
    $W$ is a Brownian motion.
    """

    term_structure = LangevinTerm
    interpolation_cls = LocalLinearInterpolation
    taylor_threshold: RealScalarLike = eqx.field(static=True)
    _coeffs_structure = jtu.tree_structure(
        _ShOULDCoeffs(
            beta_half=jnp.array(0.0),
            a_half=jnp.array(0.0),
            b_half=jnp.array(0.0),
            beta1=jnp.array(0.0),
            a1=jnp.array(0.0),
            b1=jnp.array(0.0),
            aa=jnp.array(0.0),
            chh=jnp.array(0.0),
            ckk=jnp.array(0.0),
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
    def _directly_compute_coeffs_leaf(h, c) -> _ShOULDCoeffs:
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

        out = _ShOULDCoeffs(
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
        return jtu.tree_map(lambda x: jnp.array(x, dtype=jnp.dtype(c)), out)

    @staticmethod
    def _tay_cfs_single(c: Array) -> _ShOULDCoeffs:
        # c is a leaf of gamma
        zero = jnp.zeros_like(c)
        one = jnp.ones_like(c)
        dtype = jnp.dtype(c)
        c2 = jnp.square(c)
        c3 = c2 * c
        c4 = c3 * c
        c5 = c4 * c

        beta_half = jnp.stack(
            [one, -c / 2, c2 / 8, -c3 / 48, c4 / 384, -c5 / 3840], axis=-1
        )
        beta1 = jnp.stack([one, -c, c2 / 2, -c3 / 6, c4 / 24, -c5 / 120], axis=-1)

        a_half = jnp.stack(
            [zero, one / 2, -c / 8, c2 / 48, -c3 / 384, c4 / 3840], axis=-1
        )
        a1 = jnp.stack([zero, one, -c / 2, c2 / 6, -c3 / 24, c4 / 120], axis=-1)
        # aa = a1/h
        aa = jnp.stack([one, -c / 2, c2 / 6, -c3 / 24, c4 / 120, -c5 / 720], axis=-1)

        # b_half is not exactly b(1/2 h), but 1/2 * b(1/2 h)
        b_half = jnp.stack(
            [zero, one / 8, -c / 48, c2 / 384, -c3 / 3840, c4 / 46080], axis=-1
        )
        b1 = jnp.stack([zero, one / 2, -c / 6, c2 / 24, -c3 / 120, c4 / 720], axis=-1)

        chh = jnp.stack([zero, one, -c / 2, 3 * c2 / 20, -c3 / 30, c4 / 168], axis=-1)
        ckk = jnp.stack([zero, zero, -c, c2 / 2, -c3 / 7, 5 * c4 / 168], axis=-1)

        out = _ShOULDCoeffs(
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
        return jtu.tree_map(lambda x: jnp.array(x, dtype=dtype), out)

    @staticmethod
    def _compute_step(
        h: RealScalarLike,
        levy: AbstractSpaceTimeTimeLevyArea,
        x0: LangevinX,
        v0: LangevinX,
        langevin_args: _LangevinArgs,
        cfs: _ShOULDCoeffs,
        st: _SolverState,
    ) -> tuple[LangevinX, LangevinX, LangevinX, _ErrorEstimate]:
        w: LangevinX = levy.W
        hh: LangevinX = levy.H
        kk: LangevinX = levy.K

        gamma, u, f = langevin_args

        rho_w_k = (st.rho**ω * (w**ω - 12 * kk**ω)).ω
        uh = (u**ω * h).ω

        f0 = st.prev_f
        v1 = (v0**ω + st.rho**ω * (hh**ω + 6 * kk**ω)).ω
        x1 = (
            x0**ω
            + cfs.a_half**ω * v1**ω
            + cfs.b_half**ω * (-(uh**ω) * f0**ω + rho_w_k**ω)
        ).ω
        f1 = f(x1)

        chh_hh_plus_ckk_kk = (cfs.chh**ω * hh**ω + cfs.ckk**ω * kk**ω).ω

        x_out = (
            x0**ω
            + cfs.a1**ω * v0**ω
            - uh**ω * cfs.b1**ω * (1 / 3 * f0**ω + 2 / 3 * f1**ω)
            + st.rho**ω * (cfs.b1**ω * w**ω + chh_hh_plus_ckk_kk**ω)
        ).ω
        f_out = f(x_out)
        v_out = (
            cfs.beta1**ω * v0**ω
            - uh**ω
            * (
                cfs.beta1**ω / 6 * f0**ω
                + 2 / 3 * cfs.beta_half**ω * f1**ω
                + 1 / 6 * f_out**ω
            )
            + st.rho**ω * (cfs.aa**ω * w**ω - gamma**ω * chh_hh_plus_ckk_kk**ω)
        ).ω

        return x_out, v_out, f_out, None
