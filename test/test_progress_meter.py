import re

import diffrax
import jax
import jax.numpy as jnp
import pytest


def test_tqdm_progress_meter(capfd):
    def solve(t0):
        term = diffrax.ODETerm(lambda t, y, args: -0.2 * y)
        solver = diffrax.Dopri5()
        t1 = 5
        dt0 = 0.01
        y0 = 1.0
        saveat = diffrax.SaveAt(steps=True)
        sol = diffrax.diffeqsolve(
            term,
            solver,
            t0,
            t1,
            dt0,
            y0,
            saveat=saveat,
            progress_meter=diffrax.TqdmProgressMeter(refresh_steps=5),
        )
        return sol

    solve_fns = [
        (62, lambda: solve(2.0)),
        (62, lambda: jax.jit(solve)(2.0)),
        (102, lambda: jax.vmap(solve)(jnp.arange(3.0))),
        (102, lambda: jax.jit(jax.vmap(solve))(jnp.arange(3.0))),
    ]
    for num_lines, solve_fn in solve_fns:
        capfd.readouterr()
        solve_fn()
        captured = capfd.readouterr()
        err = captured.err.strip()
        assert re.match("0.00%|[ ]+|", err.split("\r", 1)[0])
        assert re.match("100.00%|█+|", err.rsplit("\r", 1)[1])
        assert captured.err.count("\r") == num_lines
        assert captured.err.count("\n") == 1


def test_text_progress_meter(capfd):
    def solve(t0):
        return diffrax.diffeqsolve(
            terms=diffrax.ODETerm(lambda t, y, args: -0.2 * y),
            solver=diffrax.Dopri5(),
            t0=t0,
            t1=5,
            dt0=0.01,
            y0=1.0,
            progress_meter=diffrax.TextProgressMeter(minimum_increase=0.1),
        )

    solve(2.0)
    captured = capfd.readouterr()
    expected = "0.00%\n10.33%\n20.67%\n31.00%\n41.33%\n51.67%\n62.00%\n72.33%\n82.67%\n93.00%\n100.00%\n"  # noqa: E501
    assert captured.out == expected

    jax.vmap(solve)(jnp.arange(3.0))
    captured = capfd.readouterr()
    expected = "0.00%\n10.00%\n20.00%\n30.00%\n40.00%\n50.20%\n60.40%\n70.60%\n80.80%\n91.00%\n100.00%\n"  # noqa: E501
    assert captured.out == expected

    jax.jit(solve)(2.0)
    captured = capfd.readouterr()
    expected = "0.00%\n10.33%\n20.67%\n31.00%\n41.33%\n51.67%\n62.00%\n72.33%\n82.67%\n93.00%\n100.00%\n"  # noqa: E501
    assert captured.out == expected

    jax.jit(jax.vmap(solve))(jnp.arange(3.0))
    captured = capfd.readouterr()
    expected = "0.00%\n10.00%\n20.00%\n30.00%\n40.00%\n50.20%\n60.40%\n70.60%\n80.80%\n91.00%\n100.00%\n"  # noqa: E501
    assert captured.out == expected


@pytest.mark.parametrize(
    "progress_meter", [diffrax.TqdmProgressMeter(), diffrax.TextProgressMeter()]
)
def test_grad_progress_meter(progress_meter, capfd):
    def solve(p):
        term = diffrax.ODETerm(lambda t, y, args: -y)
        solver = diffrax.Dopri5()
        y0 = p * jnp.array([2.0, 3.0])
        sol = diffrax.diffeqsolve(
            term,
            solver,
            t0=0,
            t1=1,
            dt0=0.1,
            y0=y0,
            progress_meter=progress_meter,
        )
        return sol.ys[-1, 0]  # pyright: ignore

    capfd.readouterr()
    jax.grad(solve)(jnp.array(1.0))
    captured = capfd.readouterr()

    if isinstance(progress_meter, diffrax.TextProgressMeter):
        true_out = (
            "0.00%\n10.00%\n20.00%\n30.00%\n40.00%\n50.00%\n60.00%"
            "\n70.00%\n80.00%\n90.00%\n100.00%\n"
        )
        assert captured.out == true_out

    jax.jit(jax.grad(solve))(jnp.array(1.0))
