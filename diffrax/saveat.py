from typing import Optional, Sequence, Union

import equinox as eqx
import jax
import jax.numpy as jnp

from .custom_types import Array, Scalar


class SaveAt(eqx.Module):
    """Determines what to save as output from the differential equation solve.

    Instances of this class should be passed as the `saveat` argument of
    [`diffrax.diffeqsolve`][].
    """

    t0: bool = False
    t1: bool = False
    ts: Optional[Union[Sequence[Scalar], Array["times"]]] = None  # noqa: F821
    steps: bool = False
    dense: bool = False
    solver_state: bool = False
    controller_state: bool = False
    made_jump: bool = False

    def __post_init__(self):
        with jax.ensure_compile_time_eval():
            ts = None if self.ts is None else jnp.asarray(self.ts)
        object.__setattr__(self, "ts", ts)
        if (
            not self.t0
            and not self.t1
            and self.ts is None
            and not self.steps
            and not self.dense
        ):
            raise ValueError("Empty saveat -- nothing will be saved.")


SaveAt.__init__.__doc__ = """**Main Arguments:**

- `t0`: If `True`, save the initial input `y0`.
- `t1`: If `True`, save the output at `t1`.
- `ts`: Some array of times at which to save the output.
- `steps`: If `True`, save the output at every step of the numerical solver.
- `dense`: If `True`, save dense output, that can later be evaluated at any part of
    the interval $[t_0, t_1]$ via `sol = diffeqsolve(...); sol.evaluate(...)`.

**Other Arguments:**

It is less likely you will need to use these options.

- `solver_state`: If `True`, save the internal state of the numerical solver at
    `t1`.
- `controller_state`: If `True`, save the internal state of the step size
    controller at `t1`.
- `made_jump`: If `True`, save the internal state of the jump tracker at `t1`.
"""
