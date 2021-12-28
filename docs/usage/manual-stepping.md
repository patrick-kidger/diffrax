# Interactively step through a solve

!!! warning

    This API should now be relatively stable, but in principle may still be subject to change.

Sometimes you might want to do perform a differential equation solve just one step at a time (or a few steps at a time), and perhaps do some other computations in between.  A common example is when solving a differential equation in real time, and wanting to continually produce some output.

One option is to repeatedly call `diffrax.diffeqsolve`. However if that seems inelegant/inefficient to you, then it is possible to use the solvers (and step size controllers, etc.) yourself directly.

In the following example, we solve an ODE using [`diffrax.tsit5`][], and print out the result as we go.

!!! note

    See the [Solvers](../api/solver.md) page for a reference on the solver methods (`wrap`, `init`, `step`) used here.

```python
from diffrax import tsit5
from diffrax.misc import ravel_pytree

vector_field = lambda t, y, args: -y
solver = tsit5(vector_field)

t0 = 0
dt0 = 0.05
t1 = 1
y0 = 1
args = None
direction = t0 < t1

tprev = t0
tnext = t0 + dt0
y, unravel_y = ravel_pytree(y0)

solver = solver.wrap(tprev, y, args, direction)
state = solver.init(tprev, tnext, y, args)

while tprev < t1:
    y, _, _, state, _ = solver.step(tprev, tnext, y, args, state, made_jump=False)
    print(f"At time {tnext} obtained value {unravel_y(y)}")
    tprev = tnext
    tnext = min(tprev + dt0, t1)
```

## JIT'ing the above example

We can modify the above example to happen even faster by JIT'ing each step.

One option is to just JIT the vector field: replace
```python
vector_field = lambda t, y, args: -y
```
with
```python
vector_field = jax.jit(lambda t, y, args: -y)
```

However it is possible to be subtantially more efficient than this, by JIT'ing the solver and the vector field together. This is done by modifying our original example like so.

!!! tip

    This modified example uses the [Equinox](https://github.com/patrick-kidger/equinox) libary to automatically sort out which arguments should be JIT-traced and which should be JIT-static'd.

```python
from diffrax import tsit5
from diffrax.misc import ravel_pytree
from equinox import filter_jit

vector_field = lambda t, y, args: -y
solver = tsit5(vector_field)

t0 = 0
dt0 = 0.05
t1 = 1
y0 = 1
args = None
direction = t0 < t1

tprev = t0
tnext = t0 + dt0
y, unravel_y = ravel_pytree(y0)

solver = solver.wrap(tprev, y, args, direction)
state = solver.init(tprev, tnext, y, args)

@filter_jit
def make_step(solver, tprev, tnext, y, args, state):
    y, _, _, state, _ = solver.step(tprev, tnext, y, args, state, made_jump=False)
    return y, state

while tprev < t1:
    y, state = make_step(solver, tprev, tnext, y, args, state)
    print(f"At time {tnext} obtained value {unravel_y(y)}")
    tprev = tnext
    tnext = min(tprev + dt0, t1)
```
