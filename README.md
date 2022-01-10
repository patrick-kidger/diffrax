<h1 align='center'>Diffrax</h1>
<h2 align='center'>Autodifferentiable CPU+GPU-capable differential equation solvers in JAX</h2>

Diffrax is a [JAX](https://github.com/google/jax)-based library providing numerical differential equation solvers.

Features include:
- ODE/SDE/CDE (ordinary/stochastic/controlled) solvers;
- lots of different solvers (including `Tsit5`, `Dopri8`, symplectic solvers, implicit solvers);
- vmappable _everything_ (including the region of integration);
- using a PyTree as the state;
- dense solutions;
- multiple adjoint methods for backpropagation;
- support for neural differential equations.

_From a technical point of view, the internal structure of the library is pretty cool -- all kinds of equations (ODEs, SDEs, CDEs) are solved in a unified way (rather than being treated separately), producing a small tightly-written library._

---

## Installation

```
TODO
```
Requires Python 3.8+ and JAX 0.2.20+

## Examples

- [`neural_ode.ipynb`](./examples/neural_ode.ipynb) trains a [neural ODE](https://arxiv.org/abs/1806.07366) to match a spiral.
- [`neural_cde.ipynb`](./examples/neural_cde.ipynb) trains a [neural CDE](https://arxiv.org/abs/2005.08926) to classify clockwise vs anticlockwise spirals.
- [`latent_ode.ipynb`](./examples/latent_ode.ipynb) trains a [latent ODE](https://arxiv.org/abs/1907.03907) -- a generative model for time series -- on a dataset of decaying oscillators.
- [`continuous_normalising_flow.ipynb`](./examples/continuous_normalising_flow.ipynb) trains a [CNF](https://arxiv.org/abs/1810.01367) -- a generative model for e.g. images -- to reproduce whatever input image you give it!
- [`symbolic_regression.ipynb`](./examples/symbolic_regression.ipynb) extends the neural ODE example, by additionally performing [regularised evolution](https://arxiv.org/abs/1802.01548) to discover the exact symbolic form of the governing equations. (An improvement on [SINDy](https://www.pnas.org/content/113/15/3932), basically.)
- [`stiff_ode.ipynb`](./examples/stiff_ode.ipynb) demonstrates the use of implicit solvers to solve a stiff ODE, namely the Robertson problem.
- [`stochastic_gradient_descent.ipynb`](./examples/stochastic_gradient_descent.ipynb) trains a simple neural network via SGD, using an ODE solver. (SGD is just Euler's method for solving an ODE.)

Quick example:

```python
from diffrax import diffeqsolve, ODETerm, Dopri5
import jax.numpy as jnp

def f(t, y, args):
    return -y

term = ODETerm(f)
t0, t1 = 0, 1
y0 = jnp.array([2., 3.])
dt0 = 0.1
solver = Dopri5()
solution = diffeqsolve(term, t0, t1, y0, dt0, solver)
```

## Documentation

See TODO.

## Citation

If you found this library useful in academic research, please consider citing:

```bibtex
@phdthesis{kidger2021on,
    title={{O}n {N}eural {D}ifferential {E}quations},
    author={Patrick Kidger},
    year={2021},
    school={University of Oxford},
}
```
