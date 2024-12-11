<h1 align='center'>DiffraxU</h1>
<h2 align='center'>Unit-aware numerical differential equation solvers in JAX.</h2>

Diffrax is a [JAX](https://github.com/google/jax)-based library providing [unit-aware](https://github.com/chaobrain/brainunit) numerical differential equation solvers.

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
pip install git+https://github.com/chaoming0625/diffrax.git
```

Requires Python 3.9+, JAX 0.4.13+, and [Equinox](https://github.com/patrick-kidger/equinox) 0.10.11+.

## Quick example

```python
import brainunit as u
from diffrax import diffeqsolve, ODETerm, Dopri5

def f(t, y, args):
    return -y / u.ms

term = ODETerm(f)
solver = Dopri5()
y0 = u.math.array([2., 3.]) * u.mV
solution = diffeqsolve(term, solver, t0=0 * u.ms, t1=1 * u.ms, dt0=0.1 * u.ms, y0=y0)
```

Here, `Dopri5` refers to the Dormand--Prince 5(4) numerical differential equation solver, which is a standard choice for many problems.

## Documentation

Available at [https://docs.kidger.site/diffrax](https://docs.kidger.site/diffrax).

## Citation

If you found this library useful in academic research, please cite: [(arXiv link)](https://arxiv.org/abs/2202.02435)

```bibtex



@phdthesis{kidger2021on,
    title={{O}n {N}eural {D}ifferential {E}quations},
    author={Patrick Kidger},
    year={2021},
    school={University of Oxford},
}
```

