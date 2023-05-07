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


def test_implicit_euler_adaptive():
    term = diffrax.ODETerm(lambda t, y, args: -10 * y**3)
    solver1 = diffrax.ImplicitEuler(
        nonlinear_solver=diffrax.NewtonNonlinearSolver(rtol=1e-5, atol=1e-5)
    )
    solver2 = diffrax.ImplicitEuler()
    t0 = 0
    t1 = 1
    dt0 = 1
    y0 = 1.0
    stepsize_controller = diffrax.PIDController(rtol=1e-5, atol=1e-5)
    out1 = diffrax.diffeqsolve(term, solver1, t0, t1, dt0, y0, throw=False)
    out2 = diffrax.diffeqsolve(
        term,
        solver2,
        t0,
        t1,
        dt0,
        y0,
        stepsize_controller=stepsize_controller,
        throw=False,
    )
    assert out1.result == diffrax.RESULTS.implicit_nonconvergence
    assert out2.result == diffrax.RESULTS.successful
