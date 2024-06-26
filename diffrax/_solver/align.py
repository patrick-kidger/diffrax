from typing import TypeAlias

import equinox as eqx
import jax.numpy as jnp
import jax.tree_util as jtu
from equinox.internal import ω
from jaxtyping import Array, PyTree

from .._custom_types import (
    AbstractSpaceTimeLevyArea,
    RealScalarLike,
)
from .._local_interpolation import LocalLinearInterpolation
from .._term import _LangevinArgs, LangevinTerm, LangevinTuple, LangevinX
from .langevin_srk import _AbstractCoeffs, _SolverState, AbstractLangevinSRK


# UBU evaluates at l = (3 -sqrt(3))/6, at r = (3 + sqrt(3))/6 and at 1,
# so we need 3 versions of each coefficient


class _ALIGNCoeffs(_AbstractCoeffs):
    @property
    def dtype(self):
        return jtu.tree_leaves(self.beta)[0].dtype

    beta: PyTree[Array]
    a1: PyTree[Array]
    b1: PyTree[Array]
    aa: PyTree[Array]
    chh: PyTree[Array]


_ErrorEstimate: TypeAlias = LangevinTuple


class ALIGN(AbstractLangevinSRK[_ALIGNCoeffs, _ErrorEstimate]):
    r"""The Adaptive Langevin via Interpolated Gradients and Noise method
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
        _ALIGNCoeffs(
            beta=jnp.array(0.0),
            a1=jnp.array(0.0),
            b1=jnp.array(0.0),
            aa=jnp.array(0.0),
            chh=jnp.array(0.0),
        )
    )

    @property
    def minimal_levy_area(self):
        return AbstractSpaceTimeLevyArea

    def __init__(self, taylor_threshold: RealScalarLike = 0.0):
        r"""**Arguments:**

        - `taylor_threshold`: If the product `h*gamma` is less than this, then
        the Taylor expansion will be used to compute the coefficients.
        Otherwise they will be computed directly. When using float32, the
        empirically optimal value is 0.1, and for float64 about 0.01.
        """
        self.taylor_threshold = taylor_threshold

    def order(self, terms):
        return 2

    def strong_order(self, terms):
        return 2.0

    @staticmethod
    def _directly_compute_coeffs_leaf(h, c) -> _ALIGNCoeffs:
        # c is a leaf of gamma
        # compute the coefficients directly (as opposed to via Taylor expansion)
        al = c * h
        beta = jnp.exp(-al)
        a1 = (1 - beta) / c
        b1 = (beta + al - 1) / (c * al)
        aa = a1 / h

        al2 = al**2
        chh = 6 * (beta * (al + 2) + al - 2) / (al2 * c)

        out = _ALIGNCoeffs(
            beta=beta,
            a1=a1,
            b1=b1,
            aa=aa,
            chh=chh,
        )
        return jtu.tree_map(lambda x: jnp.array(x, dtype=jnp.dtype(c)), out)

    @staticmethod
    def _tay_cfs_single(c: Array) -> _ALIGNCoeffs:
        # c is a leaf of gamma
        dtype = jnp.dtype(c)
        zero = jnp.zeros_like(c)
        one = jnp.ones_like(c)
        c2 = jnp.square(c)
        c3 = c2 * c
        c4 = c3 * c
        c5 = c4 * c

        beta = jnp.stack([one, -c, c2 / 2, -c3 / 6, c4 / 24, -c5 / 120], axis=-1)
        a1 = jnp.stack([zero, one, -c / 2, c2 / 6, -c3 / 24, c4 / 120], axis=-1)
        b1 = jnp.stack([zero, one / 2, -c / 6, c2 / 24, -c3 / 120, c4 / 720], axis=-1)
        aa = jnp.stack([one, -c / 2, c2 / 6, -c3 / 24, c4 / 120, -c5 / 720], axis=-1)
        chh = jnp.stack([zero, one, -c / 2, 3 * c2 / 20, -c3 / 30, c4 / 168], axis=-1)

        out = _ALIGNCoeffs(
            beta=beta,
            a1=a1,
            b1=b1,
            aa=aa,
            chh=chh,
        )
        return jtu.tree_map(lambda x: jnp.array(x, dtype=dtype), out)

    @staticmethod
    def _compute_step(
        h: RealScalarLike,
        levy: AbstractSpaceTimeLevyArea,
        x0: LangevinX,
        v0: LangevinX,
        langevin_args: _LangevinArgs,
        cfs: _ALIGNCoeffs,
        st: _SolverState,
    ) -> tuple[LangevinX, LangevinX, LangevinX, _ErrorEstimate]:
        w: LangevinX = levy.W
        hh: LangevinX = levy.H

        gamma, u, f = langevin_args

        uh = (u**ω * h).ω
        f0 = st.prev_f
        x1 = (
            x0**ω
            + cfs.a1**ω * v0**ω
            - cfs.b1**ω * uh**ω * f0**ω
            + st.rho**ω * (cfs.b1**ω * w**ω + cfs.chh**ω * hh**ω)
        ).ω
        f1 = f(x1)
        v1 = (
            cfs.beta**ω * v0**ω
            - u**ω * ((cfs.a1**ω - cfs.b1**ω) * f0**ω + cfs.b1**ω * f1**ω)
            + st.rho**ω * (cfs.aa**ω * w**ω - gamma**ω * cfs.chh**ω * hh**ω)
        ).ω

        error_estimate = (
            jtu.tree_map(lambda leaf: jnp.zeros_like(leaf), x0),
            (-(u**ω) * cfs.b1**ω * (f1**ω - f0**ω)).ω,
        )

        return x1, v1, f1, error_estimate
