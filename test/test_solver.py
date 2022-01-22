import diffrax


def test_half_solver():
    term = diffrax.ODETerm(lambda t, y, args: -y)
    t0 = 0
    t1 = 1
    y0 = 1.0
    dt0 = None
    solver = diffrax.HalfSolver(diffrax.Euler())
    stepsize_controller = diffrax.IController()
    diffrax.diffeqsolve(
        term, t0, t1, y0, dt0, solver, stepsize_controller=stepsize_controller
    )


def test_instance_check():
    assert isinstance(diffrax.HalfSolver(diffrax.Euler()), diffrax.Euler)
    assert not isinstance(diffrax.HalfSolver(diffrax.Euler()), diffrax.Heun)
