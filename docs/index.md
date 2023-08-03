# Diffrax in a nutshell

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

## Installation

```
pip install diffrax
```

Requires Python 3.9+, JAX 0.4.13+, and [Equinox](https://github.com/patrick-kidger/equinox) 0.10.11+.

## Quick example

```python
from diffrax import diffeqsolve, ODETerm, Dopri5
import jax.numpy as jnp

def f(t, y, args):
    return -y

term = ODETerm(f)
solver = Dopri5()
y0 = jnp.array([2., 3.])
solution = diffeqsolve(term, solver, t0=0, t1=1, dt0=0.1, y0=y0)
```

Here, `Dopri5` refers to the Dormand--Prince 5(4) numerical differential equation solver, which is a standard choice for many problems.

## Citation

--8<-- "further_details/.citation.md"

## Next steps

Have a look at the [Getting Started](./usage/getting-started.md) page.

## See also: other libraries in the JAX ecosystem

[Equinox](https://github.com/patrick-kidger/equinox): neural networks.

[Optax](https://github.com/deepmind/optax): first-order gradient (SGD, Adam, ...) optimisers.

[Lineax](https://github.com/google/lineax): linear solvers and linear least squares.

[jaxtyping](https://github.com/google/jaxtyping): type annotations for shape/dtype of arrays.

[Eqxvision](https://github.com/paganpasta/eqxvision): computer vision models.

[sympy2jax](https://github.com/google/sympy2jax): SymPy<->JAX conversion; train symbolic expressions via gradient descent.

[Levanter](https://github.com/stanford-crfm/levanter): scalable+reliable training of foundation models (e.g. LLMs).
