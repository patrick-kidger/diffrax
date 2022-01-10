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

TODO

Requires Python >=3.8 and JAX >=0.2.20.

## Quick example

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

Here, `Dopri5` refers to the Dormand--Prince 5(4) numerical differential equation solver, which is a standard choice for many problems.

## Citation

--8<-- "further_details/.citation.md"

## Getting started

If this page has caught your interest, then have a look at the [Getting Started](./usage/getting-started.md) page.
