import diffrax
import numpy as np


def test_tqdm_progress_bar():
    # todo: use a mock to check if tqdm is called with the correct arguments
    term = diffrax.ODETerm(lambda t, y, args: -0.2 * y)
    solver = diffrax.Dopri5()
    t0 = 0
    t1 = 5
    dt0 = 0.01
    y0 = 1.0

    saveat = diffrax.SaveAt(steps=True)
    diffrax.diffeqsolve(
        term,
        solver,
        t0,
        t1,
        dt0,
        y0,
        saveat=saveat,
        progress_bar=diffrax.TqdmProgressBar(),
    )


def test_text_progress_bar(capfd):
    term = diffrax.ODETerm(lambda t, y, args: -0.2 * y)
    solver = diffrax.Dopri5()
    t0 = 0
    t1 = 5
    dt0 = 0.01
    y0 = 1.0
    saveat = diffrax.SaveAt(steps=True)

    diffrax.diffeqsolve(
        term,
        solver,
        t0,
        t1,
        dt0,
        y0,
        saveat=saveat,
        progress_bar=diffrax.TextProgressBar(minimum_increase=10.0),
    )
    captured = capfd.readouterr()

    values = captured.out.split("\n")
    assert len(values) > 10  # assert at least a few steps were printed
    # skip the last update (100%) because it could be smaller than `minimum_increase`
    # and also skip the last empty line
    values = values[:-2]
    values = list(map(lambda x: float(x[:-1]), values))
    assert values == sorted(values)
    assert np.all(np.diff(values) >= 10.0)
