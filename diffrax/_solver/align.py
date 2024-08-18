import equinox as eqx
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np
from equinox.internal import ω
from jaxtyping import Array, ArrayLike, PyTree

from .._custom_types import (
    AbstractSpaceTimeLevyArea,
    RealScalarLike,
)
from .._local_interpolation import LocalLinearInterpolation
from .._term import LangevinTuple, LangevinX
from .langevin_srk import (
    _LangevinArgs,
    AbstractCoeffs,
    AbstractLangevinSRK,
    SolverState,
)


# For an explanation of the coefficients, see langevin_srk.py
class _ALIGNCoeffs(AbstractCoeffs):
    beta: PyTree[ArrayLike]
    a1: PyTree[ArrayLike]
    b1: PyTree[ArrayLike]
    aa: PyTree[ArrayLike]
    chh: PyTree[ArrayLike]
    dtype: jnp.dtype = eqx.field(static=True)

    def __init__(self, beta, a1, b1, aa, chh):
        self.beta = beta
        self.a1 = a1
        self.b1 = b1
        self.aa = aa
        self.chh = chh
        all_leaves = jtu.tree_leaves([self.beta, self.a1, self.b1, self.aa, self.chh])
        self.dtype = jnp.result_type(*all_leaves)


_ErrorEstimate = LangevinTuple


class ALIGN(AbstractLangevinSRK[_ALIGNCoeffs, _ErrorEstimate]):
    r"""The Adaptive Langevin via Interpolated Gradients and Noise method
    designed by James Foster.
    Accepts only terms given by [`diffrax.make_langevin_term`][].
    """

    interpolation_cls = LocalLinearInterpolation
    taylor_threshold: RealScalarLike = eqx.field(static=True)
    _coeffs_structure = jtu.tree_structure(
        _ALIGNCoeffs(
            beta=np.array(0.0),
            a1=np.array(0.0),
            b1=np.array(0.0),
            aa=np.array(0.0),
            chh=np.array(0.0),
        )
    )
    minimal_levy_area = AbstractSpaceTimeLevyArea

    def __init__(self, taylor_threshold: RealScalarLike = 0.1):
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

    def _directly_compute_coeffs_leaf(self, h, c) -> _ALIGNCoeffs:
        del self
        # c is a leaf of gamma
        # compute the coefficients directly (as opposed to via Taylor expansion)
        al = c * h
        beta = jnp.exp(-al)
        a1 = (1 - beta) / c
        b1 = (beta + al - 1) / (c * al)
        aa = a1 / h

        al2 = al**2
        chh = 6 * (beta * (al + 2) + al - 2) / (al2 * c)

        return _ALIGNCoeffs(
            beta=beta,
            a1=a1,
            b1=b1,
            aa=aa,
            chh=chh,
        )

    def _tay_coeffs_single(self, c: Array) -> _ALIGNCoeffs:
        del self
        # c is a leaf of gamma
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

        return _ALIGNCoeffs(
            beta=beta,
            a1=a1,
            b1=b1,
            aa=aa,
            chh=chh,
        )

    @staticmethod
    def _compute_step(
        h: RealScalarLike,
        levy: AbstractSpaceTimeLevyArea,
        x0: LangevinX,
        v0: LangevinX,
        langevin_args: _LangevinArgs,
        coeffs: _ALIGNCoeffs,
        st: SolverState,
    ) -> tuple[LangevinX, LangevinX, LangevinX, LangevinTuple]:
        dtypes = jtu.tree_map(jnp.dtype, x0)
        w: LangevinX = jtu.tree_map(jnp.asarray, levy.W, dtypes)
        hh: LangevinX = jtu.tree_map(jnp.asarray, levy.H, dtypes)

        gamma, u, f = langevin_args

        uh = (u**ω * h).ω
        f0 = st.prev_f
        x1 = (
            x0**ω
            + coeffs.a1**ω * v0**ω
            - coeffs.b1**ω * uh**ω * f0**ω
            + st.rho**ω * (coeffs.b1**ω * w**ω + coeffs.chh**ω * hh**ω)
        ).ω
        f1 = f(x1)
        v1 = (
            coeffs.beta**ω * v0**ω
            - u**ω * ((coeffs.a1**ω - coeffs.b1**ω) * f0**ω + coeffs.b1**ω * f1**ω)
            + st.rho**ω * (coeffs.aa**ω * w**ω - gamma**ω * coeffs.chh**ω * hh**ω)
        ).ω

        error_estimate = (
            jtu.tree_map(lambda leaf: jnp.zeros_like(leaf), x0),
            (-(u**ω) * coeffs.b1**ω * (f1**ω - f0**ω)).ω,
        )

        return x1, v1, f1, error_estimate
