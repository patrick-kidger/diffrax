import diffrax
import jax
import jax.numpy as jnp


def test_tqdm_progress_meter():
    # TODO: use a mock to check if tqdm is called with the correct arguments

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
            progress_meter=diffrax.TqdmProgressMeter(),
        )
        return sol

    solve(2.0)
    jax.vmap(solve)(jnp.arange(3.0))


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
    # assert captured.out == expected

    jax.vmap(solve)(jnp.arange(3.0))
    captured = capfd.readouterr()
    expected = "0.00%\n10.00%\n20.00%\n30.00%\n40.00%\n50.20%\n60.40%\n70.60%\n80.80%\n91.00%\n100.00%\n"  # noqa: E501
    assert captured.out == expected
