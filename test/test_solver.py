import diffrax


def test_half_solver():
    term = diffrax.ODETerm(lambda t, y, args: -y)
    t0 = 0
    t1 = 1
    y0 = 1.0
    dt0 = None
    solver = diffrax.HalfSolver(diffrax.Euler())
    stepsize_controller = diffrax.PIDController(rtol=1e-3, atol=1e-6)
    diffrax.diffeqsolve(
        term, solver, t0, t1, dt0, y0, stepsize_controller=stepsize_controller
    )


def test_instance_check():
    assert isinstance(diffrax.HalfSolver(diffrax.Euler()), diffrax.Euler)
    assert not isinstance(diffrax.HalfSolver(diffrax.Euler()), diffrax.Heun)
