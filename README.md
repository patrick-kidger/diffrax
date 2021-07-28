<h1 align='center'>Diffrax</h1>
<h2 align='center'>Autodifferentiable GPU-capable differential equation solvers in JAX</h2>

Diffrax is a [JAX](https://github.com/google/jax)-based library providing numerical solvers for:
- ordinary differential equations
- stochastic differential equations
- controlled differential equations

Features include:
- fixed step size solvers
- adaptive step size solvers
- reversible solvers
- discretise-then-optimise backpropagation
- continuous adjoint backpropagation
- interpolated adjoint backpropagation
- reversible adjoint backpropagation
- efficient Brownian Interval-based sampling
- using a PyTree as the state

From a technical point of view, its unique selling point is the way all kinds of equations (ODEs, SDEs, CDEs) are solved in a unified way (rather than being treated separately), producing a small tightly-written library.

---

Example
