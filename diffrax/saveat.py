from typing import Optional

import equinox as eqx
import jax.numpy as jnp

from .custom_types import Array


class SaveAt(eqx.Module):
    """Determines what to save as output from the differential equation solve.

    Instances of this class should be passed as the `saveat` argument of
    [`diffrax.diffeqsolve`][].
    """

    t0: bool
    t1: bool
    ts: Optional[Array["times"]]  # noqa: F821
    steps: bool
    dense: bool
    solver_state: bool
    controller_state: bool
    made_jump: bool

    # Explicit __init__ so we can do jnp.asarray(ts)
    # No super().__init__ call in mimicry of dataclasses' (and thus Equinox's) lack of
    # doing so.
    def __init__(
        self,
        *,
        t0=False,
        t1=False,
        ts=None,
        steps=False,
        dense=False,
        solver_state=False,
        controller_state=False,
        made_jump=False,
    ):
        self.t0 = t0
        self.t1 = t1
        self.ts = None if ts is None else jnp.asarray(ts)
        self.steps = steps
        self.dense = dense
        self.solver_state = solver_state
        self.controller_state = controller_state
        self.made_jump = made_jump

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
    the interval $[t_0, t_1]$.

**Other Arguments:**

It is unlikely you will need to use options.

- `solver_state`: If `True`, save the internal state of the numerical solver at
    `t1`.
- `controller_state`: If `True`, save the internal state of the step size
    controller at `t1`.
- `made_jump`: If `True`, save the internal state of the jump tracker at `t1`.
"""
