import diffrax
import jax
import jax.numpy as jnp
import lineax as lx
import pytest


@pytest.fixture(autouse=True)
def _standard_dtype_promotion():
    # These SDEs intentionally combine real Wiener increments with complex states.
    with jax.numpy_dtype_promotion("standard"):
        yield


@pytest.mark.parametrize("dtype", (jnp.float64, jnp.complex128))
@pytest.mark.parametrize("use_operator", (False, True))
def test_ito_milstein_real_noise_realification(dtype, use_operator):
    # Two independent scalar linear SDEs. Their complex and real-coordinate
    # representations must give the same Milstein step.
    y0 = jnp.array([0.7, -1.1], dtype=dtype)
    drift_coeff = jnp.array([-0.1, 0.4], dtype=dtype)
    diffusion_coeff = jnp.array([0.3, -0.2], dtype=dtype)
    if jnp.issubdtype(dtype, jnp.complexfloating):
        y0 = y0 + 1j * jnp.array([1.2, -0.5])
        drift_coeff = drift_coeff + 1j * jnp.array([0.2, -0.1])
        diffusion_coeff = diffusion_coeff + 1j * jnp.array([0.8, 0.5])
    dt = 0.125
    dw = jnp.array([0.2, -0.1])

    def diffusion(t, y, args):
        diagonal = diffusion_coeff * y
        if use_operator:
            return lx.DiagonalLinearOperator(diagonal)
        else:
            return jnp.diag(diagonal)

    drift = lambda t, y, args: drift_coeff * y
    control = lambda t0, t1: dw
    terms = diffrax.MultiTerm(
        diffrax.ODETerm(drift), diffrax.ControlTerm(diffusion, control)
    )
    solver = diffrax.ItoMilstein()
    actual, _, _, _, result = solver.step(terms, 0.0, dt, y0, None, None, False)

    # Respect the existing conjugation convention for array-valued ControlTerm;
    # the operator branch is an ordinary, non-conjugating matrix-vector product.
    if use_operator:
        g0 = diffusion_coeff * y0
        directional_derivative = diffusion_coeff**2 * y0
    else:
        g0 = jnp.conj(diffusion_coeff * y0)
        directional_derivative = jnp.abs(diffusion_coeff) ** 2 * y0
    expected = (
        y0
        + drift_coeff * y0 * dt
        + g0 * dw
        + 0.5 * directional_derivative * (dw**2 - dt)
    )
    assert result == diffrax.RESULTS.successful
    assert jnp.allclose(actual, expected, rtol=1e-12, atol=1e-12)

    def encode(y):
        return jnp.concatenate([jnp.real(y), jnp.imag(y)])

    def decode(y):
        real, imag = jnp.split(y, 2)
        return real + 1j * imag

    real_drift = lambda t, y, args: encode(drift(t, decode(y), args))

    def real_diffusion(t, y, args):
        g = diffusion_coeff * decode(y)
        if not use_operator:
            g = jnp.conj(g)
        return jnp.concatenate([jnp.diag(jnp.real(g)), jnp.diag(jnp.imag(g))])

    real_terms = diffrax.MultiTerm(
        diffrax.ODETerm(real_drift), diffrax.ControlTerm(real_diffusion, control)
    )
    real_actual, _, _, _, _ = solver.step(
        real_terms, 0.0, dt, encode(y0), None, None, False
    )
    assert jnp.allclose(encode(actual), real_actual, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("use_operator", (False, True))
def test_ito_milstein_complex_parameter_gradient(use_operator):
    y0 = jnp.array([0.7 + 1.2j])
    dt = 0.125
    dw = jnp.array([0.2])

    def solve(rate):
        def diffusion(t, y, args):
            diagonal = rate * (0.3 + 0.8j) * y
            if use_operator:
                return lx.DiagonalLinearOperator(diagonal)
            else:
                return jnp.diag(diagonal)

        terms = diffrax.MultiTerm(
            diffrax.ODETerm(lambda t, y, args: jnp.zeros_like(y)),
            diffrax.ControlTerm(
                diffusion,
                lambda t0, t1: dw,
            ),
        )
        y1, _, _, _, _ = diffrax.ItoMilstein().step(
            terms, 0.0, dt, y0, None, None, False
        )
        return jnp.sum(jnp.abs(y1) ** 2)

    def reference(rate):
        b = rate * (0.3 + 0.8j)
        if use_operator:
            g0 = b * y0
            directional_derivative = b**2 * y0
        else:
            g0 = jnp.conj(b * y0)
            directional_derivative = jnp.abs(b) ** 2 * y0
        y1 = y0 + g0 * dw + 0.5 * directional_derivative * (dw**2 - dt)
        return jnp.sum(jnp.abs(y1) ** 2)

    assert jnp.allclose(jax.grad(solve)(1.0), jax.grad(reference)(1.0), atol=1e-12)
