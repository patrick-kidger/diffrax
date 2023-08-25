# Interactively step through a solve

Sometimes you might want to do perform a differential equation solve just one step at a time (or a few steps at a time), and perhaps do some other computations in between.  A common example is when solving a differential equation in real time, and wanting to continually produce some output.

One option is to repeatedly call `diffrax.diffeqsolve`. However if that seems inelegant/inefficient to you, then it is possible to use the solvers (and step size controllers, etc.) yourself directly.

In the following example, we solve an ODE using [`diffrax.Tsit5`][], and print out the result as we go.

!!! note

    See the [Abstract solvers](../api/solvers/abstract_solvers.md) page for a reference on the solver methods (`init`, `step`) used here.

```python
import jax.numpy as jnp
from diffrax import ODETerm, Tsit5

vector_field = lambda t, y, args: -y
term = ODETerm(vector_field)
solver = Tsit5()

t0 = 0
dt0 = 0.05
t1 = 1
y0 = jnp.array(1.0)
args = None

tprev = t0
tnext = t0 + dt0
y = y0
state = solver.init(term, tprev, tnext, y0, args)

while tprev < t1:
    y, _, _, state, _ = solver.step(term, tprev, tnext, y, args, state, made_jump=False)
    print(f"At time {tnext} obtained value {y}")
    tprev = tnext
    tnext = min(tprev + dt0, t1)
```
