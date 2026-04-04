"""Runge--Kutta--Chebyshev (RKC) stabilised explicit solver."""

from collections.abc import Callable
from typing import ClassVar, TypeAlias

import equinox as eqx
import numpy as np
from equinox.internal import ω

from .._custom_types import Args, BoolScalarLike, DenseInfo, RealScalarLike, VF, Y
from .._local_interpolation import LocalLinearInterpolation
from .._solution import RESULTS
from .._term import AbstractTerm
from .base import AbstractSolver


_ErrorEstimate: TypeAlias = None
_SolverState: TypeAlias = None


def _rkc_coefficients(
    num_stages: int, damping: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Precompute RKC1 coefficients via the Chebyshev recurrence.

    Returns ``(mu, nu, mu_tilde, gamma_tilde, c)`` arrays of length
    ``num_stages + 1``.  Indices 0 and 1 of ``mu``, ``nu``, ``gamma_tilde``
    are unused; the recurrence starts at index 2.
    """

    s = num_stages
    w0 = 1.0 + damping / s**2

    # T_j(w0), T_j'(w0), T_j''(w0) via the three-term recurrence.
    t_val = np.zeros(s + 1)
    dt_val = np.zeros(s + 1)
    d2t_val = np.zeros(s + 1)

    t_val[0], t_val[1] = 1.0, w0
    dt_val[0], dt_val[1] = 0.0, 1.0
    # d2t_val[0] and d2t_val[1] are already 0.

    for j in range(2, s + 1):
        t_val[j] = 2.0 * w0 * t_val[j - 1] - t_val[j - 2]
        dt_val[j] = 2.0 * t_val[j - 1] + 2.0 * w0 * dt_val[j - 1] - dt_val[j - 2]
        d2t_val[j] = (
            4.0 * dt_val[j - 1] + 2.0 * w0 * d2t_val[j - 1] - d2t_val[j - 2]
        )

    # b_j weights.
    b = np.zeros(s + 1)
    b[1] = 1.0 / w0
    for j in range(2, s + 1):
        b[j] = d2t_val[j] / dt_val[j] ** 2
    b[0] = b[2]

    w1 = dt_val[s] / d2t_val[s]

    mu = np.zeros(s + 1)
    nu = np.zeros(s + 1)
    mu_tilde = np.zeros(s + 1)
    gamma_tilde = np.zeros(s + 1)
    c = np.zeros(s + 1)

    mu_tilde[1] = b[1] * w1
    c[1] = mu_tilde[1]

    for j in range(2, s + 1):
        mu[j] = 2.0 * b[j] * w0 / b[j - 1]
        nu[j] = -b[j] / b[j - 2]
        mu_tilde[j] = 2.0 * b[j] * w1 / b[j - 1]
        gamma_tilde[j] = -(1.0 - b[j - 1] * t_val[j - 1]) * mu_tilde[j]
        c[j] = w1 * d2t_val[j] / dt_val[j]

    return mu, nu, mu_tilde, gamma_tilde, c


class RKC(AbstractSolver):
    r"""Runge--Kutta--Chebyshev (RKC) stabilised explicit method.

    A first-order stabilised explicit Runge--Kutta method whose stability
    region along the negative real axis grows quadratically with the number of
    stages.  This makes it well-suited for mildly stiff problems —
    particularly semi-discrete parabolic PDEs — at much lower cost per step
    than implicit methods.

    The stability boundary satisfies
    $|\lambda|\,\Delta t \lesssim 0.65\,s^{2}$ where $s$ is `num_stages`.

    Does not provide an error estimate; use
    [`diffrax.HalfSolver`][] to obtain one if adaptive stepping is desired.

    Uses 1st order local linear interpolation for dense/ts output.

    **Arguments:**

    - `num_stages`: the number of stages $s \ge 2$.
    - `damping`: damping parameter $\varepsilon$.  The default value
      $2/13 \approx 0.154$ is the one recommended by the original paper.

    !!! example

        ```python
        import diffrax
        import jax.numpy as jnp

        def vector_field(t, y, args):
            return -50 * y  # stiff scalar ODE

        sol = diffrax.diffeqsolve(
            diffrax.ODETerm(vector_field),
            diffrax.RKC(num_stages=5),
            t0=0.0,
            t1=1.0,
            dt0=0.01,
            y0=jnp.array(1.0),
        )
        ```

    !!! cite

        ```bibtex
        @article{sommeijer1998rkc,
            author={Sommeijer, B. P. and Shampine, L. F. and Verwer, J. G.},
            title={{RKC}: An explicit solver for parabolic {PDEs}},
            journal={Journal of Computational and Applied Mathematics},
            volume={88},
            pages={315--326},
            year={1998},
        }
        ```
    """

    term_structure: ClassVar = AbstractTerm
    interpolation_cls: ClassVar[Callable[..., LocalLinearInterpolation]] = (
        LocalLinearInterpolation
    )

    num_stages: int = eqx.field(static=True)
    damping: float = eqx.field(static=True, default=2.0 / 13.0)

    def __check_init__(self):
        if self.num_stages < 2:
            raise ValueError(
                f"`num_stages` must be at least 2, got {self.num_stages}."
            )

    def order(self, terms):
        return 1

    def strong_order(self, terms):
        return 0.5

    def init(
        self,
        terms: AbstractTerm,
        t0: RealScalarLike,
        t1: RealScalarLike,
        y0: Y,
        args: Args,
    ) -> _SolverState:
        return None

    def step(
        self,
        terms: AbstractTerm,
        t0: RealScalarLike,
        t1: RealScalarLike,
        y0: Y,
        args: Args,
        solver_state: _SolverState,
        made_jump: BoolScalarLike,
    ) -> tuple[Y, _ErrorEstimate, DenseInfo, _SolverState, RESULTS]:
        del solver_state, made_jump

        s = self.num_stages
        mu, nu, mu_tilde, gamma_tilde, c = _rkc_coefficients(s, self.damping)

        control = terms.contr(t0, t1)
        f0 = terms.vf(t0, y0, args)
        k0 = terms.prod(f0, control)  # f(t0, y0) * dt

        # Stage 1: Y_1 = y0 + mu_tilde_1 * dt * f0
        y_prev2 = y0
        y_prev1 = (y0**ω + mu_tilde[1] * k0**ω).ω

        # Stages 2, ..., s via the three-term Chebyshev recurrence.
        for j in range(2, s + 1):
            tj = t0 + c[j - 1] * (t1 - t0)
            fj = terms.vf(tj, y_prev1, args)
            kj = terms.prod(fj, control)
            y_new = (
                (1.0 - mu[j] - nu[j]) * y0**ω
                + mu[j] * y_prev1**ω
                + nu[j] * y_prev2**ω
                + mu_tilde[j] * kj**ω
                + gamma_tilde[j] * k0**ω
            ).ω
            y_prev2 = y_prev1
            y_prev1 = y_new

        y1 = y_prev1
        dense_info = dict(y0=y0, y1=y1)
        return y1, None, dense_info, None, RESULTS.successful

    def func(
        self,
        terms: AbstractTerm,
        t0: RealScalarLike,
        y0: Y,
        args: Args,
    ) -> VF:
        return terms.vf(t0, y0, args)
