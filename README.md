<h1 align='center'>Diffrax</h1>
<h2 align='center'>Autodifferentiable CPU+GPU-capable differential equation solvers in JAX</h2>

Diffrax is a [JAX](https://github.com/google/jax)-based library providing numerical differential equation solvers.

Features include:
- ODE/SDE/CDE (ordinary/stochastic/controlled) solvers;
- lots of different solvers (including `tsit5`, `dopri8`, symplectic solvers);
- vmappable _everything_;
- using a PyTree as the state;
- dense solutions;
- support for neural differential equations.

_From a technical point of view, the internal structure of the library is pretty cool -- all kinds of equations (ODEs, SDEs, CDEs) are solved in a unified way (rather than being treated separately), producing a small tightly-written library._

---

## Installation

```
TODO
```
Requires Python 3.8+ and JAX 0.2.20+

## Examples

- [`neural_ode.py`](./examples/neural_ode.py) trains a [neural ODE](https://arxiv.org/abs/1806.07366) to match a spiral.
- [`neural_cde.py`](./examples/neural_cde.py) trains a [neural CDE](https://arxiv.org/abs/2005.08926) to classify clockwise vs anticlockwise spirals.
- [`latent_ode.py`](./examples/latent_ode.py) trains a [latent ODE](https://arxiv.org/abs/1907.03907) -- a generative model for time series -- on a dataset of decaying oscillators.
- [`continuous_normalising_flow.py`](./examples/continuous_normalising_flow.py) trains a [CNF](https://arxiv.org/abs/1810.01367) -- a generative model for e.g. images -- to reproduce whatever input image you give it!
- [`stochastic_gradient_descent.py`](./examples/stochastic_gradient_descent.py) trains a simple neural network via SGD, using an ODE solver. (SGD is just Euler's method for solving an ODE.)
- [`symbolic_regression.py`](./examples/symbolic_regression.py) extends the neural ODE example, by additionally performing [regularised evolution](https://arxiv.org/abs/1802.01548) to discover the exact symbolic form of the governing equations. (An improvement on [SINDy](https://www.pnas.org/content/113/15/3932), basically.)

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
