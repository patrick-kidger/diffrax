<h1 align='center'>Diffrax</h1>
<h2 align='center'>Autodifferentiable CPU+GPU-capable differential equation solvers in JAX</h2>

Diffrax is a [JAX](https://github.com/google/jax)-based library providing numerical differential equation solvers.

Features include:
- ODE/SDE/CDE (ordinary/stochastic/controlled) solvers
- lots of different solvers (including `tsit5`, `dopri8`, symplectic solvers);
- vmappable _everything_ (including simultaneous solves over different regions of integration `[t0, t1]`);
- several modes of backpropagation (including discrete-then-optimise, optimise-then-discretise, and reversible solvers);
- using a PyTree as the state;
- dense solutions;
- support for neural differential equations.

_From a technical point of view, the internal structure of the library is pretty cool -- all kinds of equations (ODEs, SDEs, CDEs) are solved in a unified way (rather than being treated separately), producing a small tightly-written library._

---

## Installation

```
TODO
```
Requires Python 3.8+ and JAX 0.2.18+

## Examples

- [`neural_ode.py`](./examples/neural_ode.py) trains a neural ODE to match a spiral.
- [`neural_cde.py`](./examples/neural_cde.py) trains a neural CDE to classify clockwise vs anticlockwise spirals.
- [`latent_ode.py`](./examples/latent_ode.py) trains a latent ODE -- a generative model for time series -- on a dataset of decaying oscillators.
- [`continuous_normalising_flow.py`](./examples/continuous_normalising_flow.py) trains a CNF -- a generative model for e.g. images -- to reproduce whatever input image you give it!
- [`stochastic_gradient_descent.py`](./examples/stochastic_gradient_descent.py) trains a simple neural network, using the fact that SGD is just Euler's method for solving an ODE.

Quick example:
```python
from diffrax import diffeqsolve, dopri5
import jax.numpy as jnp

def f(t, y, args):
    return -y

solver = dopri5(f)
solution = diffeqsolve(solver, t0=0, t1=1, y0=jnp.array([2., 3.]), dt0=0.1)
```

## Documentation

See TODO.
