# Diffrax in a nutshell

Diffrax is a [JAX](https://github.com/google/jax)-based library providing numerical differential equation solvers.

Features include:

- ODE/SDE/CDE (ordinary/stochastic/controlled) solvers;
- lots of different solvers (including `tsit5`, `dopri8`, symplectic solvers, implicit solvers);
- vmappable _everything_ (including the region of integration);
- using a PyTree as the state;
- dense solutions;
- support for neural differential equations.

_From a technical point of view, the internal structure of the library is pretty cool -- all kinds of equations (ODEs, SDEs, CDEs) are solved in a unified way (rather than being treated separately), producing a small tightly-written library._

## Installation

TODO

Requires Python >=3.8 and JAX >=0.2.20.

## Quick example

```python
from diffrax import diffeqsolve, dopri5
import jax.numpy as jnp

def f(t, y, args):
    return -y

solver = dopri5(f)
solution = diffeqsolve(solver, t0=0, t1=1, y0=jnp.array([2., 3.]), dt0=0.1)
```

(Here, `dopri5` refers to the Dormand--Prince 5(4) numerical differential equation solver, which is a standard choice for many problems.)

## Citation

--8<-- "further_details/.citation.md"

## Getting started

If this page has caught your interest, then have a look at the [Getting Started](./usage/overview.md) page.
