import equinox as eqx
import jax.numpy as jnp
import jax.tree_util as jtu
from equinox.internal import ω
from jaxtyping import ArrayLike, PyTree

from .._custom_types import (
    AbstractSpaceTimeLevyArea,
    Args,
    RealScalarLike,
)
from .._local_interpolation import LocalLinearInterpolation
from .._term import (
    UnderdampedLangevinLeaf,
    UnderdampedLangevinTuple,
    UnderdampedLangevinX,
)
from .foster_langevin_srk import (
    AbstractCoeffs,
    AbstractFosterLangevinSRK,
    UnderdampedLangevinArgs,
)


# For an explanation of the coefficients, see foster_langevin_srk.py
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


_ErrorEstimate = UnderdampedLangevinTuple


class ALIGN(AbstractFosterLangevinSRK[_ALIGNCoeffs, _ErrorEstimate]):
    r"""The Adaptive Langevin via Interpolated Gradients and Noise method
    designed by James Foster. This is a second order solver for the
    Underdamped Langevin Diffusion, and accepts terms of the form
    `MultiTerm(UnderdampedLangevinDriftTerm, UnderdampedLangevinDiffusionTerm)`.
    Uses two evaluations of the vector
    field per step, but is FSAL, so in practice it only requires one.

    ??? cite "Reference"

        This is a modification of the Strang-Splitting method from Definition 4.2 of

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
    minimal_levy_area = AbstractSpaceTimeLevyArea
    taylor_threshold: float = eqx.field(static=True)
    _is_fsal = True

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
        return 2

    def strong_order(self, terms):
        del terms
        return 2.0

    def _directly_compute_coeffs_leaf(
        self, h: RealScalarLike, c: UnderdampedLangevinLeaf
    ) -> _ALIGNCoeffs:
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

    def _tay_coeffs_single(self, c: UnderdampedLangevinLeaf) -> _ALIGNCoeffs:
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
        beta = jnp.stack([-c5 / 120, c4 / 24, -c3 / 6, c2 / 2, -c, one], axis=-1)
        a1 = jnp.stack([c4 / 120, -c3 / 24, c2 / 6, -c / 2, one, zero], axis=-1)
        b1 = jnp.stack([c4 / 720, -c3 / 120, c2 / 24, -c / 6, one / 2, zero], axis=-1)
        aa = jnp.stack([-c5 / 720, c4 / 120, -c3 / 24, c2 / 6, -c / 2, one], axis=-1)
        chh = jnp.stack([c4 / 168, -c3 / 30, 3 * c2 / 20, -c / 2, one, zero], axis=-1)

        correct_shape = jnp.shape(c) + (6,)
        assert (
            beta.shape == a1.shape == b1.shape == aa.shape == chh.shape == correct_shape
        )

        return _ALIGNCoeffs(
            beta=beta,
            a1=a1,
            b1=b1,
            aa=aa,
            chh=chh,
        )

    def _compute_step(
        self,
        h: RealScalarLike,
        levy: AbstractSpaceTimeLevyArea,
        x0: UnderdampedLangevinX,
        v0: UnderdampedLangevinX,
        underdamped_langevin_args: UnderdampedLangevinArgs,
        coeffs: _ALIGNCoeffs,
        rho: UnderdampedLangevinX,
        prev_f: UnderdampedLangevinX,
        args: Args,
    ) -> tuple[
        UnderdampedLangevinX,
        UnderdampedLangevinX,
        UnderdampedLangevinX,
        UnderdampedLangevinTuple,
    ]:
        dtypes = jtu.tree_map(jnp.result_type, x0)
        w: UnderdampedLangevinX = jtu.tree_map(jnp.asarray, levy.W, dtypes)
        hh: UnderdampedLangevinX = jtu.tree_map(jnp.asarray, levy.H, dtypes)

        gamma, u, f = underdamped_langevin_args

        uh = (u**ω * h).ω
        f0 = prev_f
        x1 = (
            x0**ω
            + coeffs.a1**ω * v0**ω
            - coeffs.b1**ω * uh**ω * f0**ω
            + rho**ω * (coeffs.b1**ω * w**ω + coeffs.chh**ω * hh**ω)
        ).ω
        f1 = f(x1, args)
        v1 = (
            coeffs.beta**ω * v0**ω
            - u**ω * ((coeffs.a1**ω - coeffs.b1**ω) * f0**ω + coeffs.b1**ω * f1**ω)
            + rho**ω * (coeffs.aa**ω * w**ω - gamma**ω * coeffs.chh**ω * hh**ω)
        ).ω

        error_estimate = (
            jtu.tree_map(jnp.zeros_like, x0),
            (-(u**ω) * coeffs.b1**ω * (f1**ω - f0**ω)).ω,
        )

        return x1, v1, f1, error_estimate
