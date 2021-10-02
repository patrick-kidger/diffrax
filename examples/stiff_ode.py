###########
#
# This example demonstrates the use of implicit integrators to handle stiff dynamical
# systems. In this case we study the Robertson problem.
#
###########
import diffrax
import equinox as eqx
import jax
import jax.numpy as jnp


###########
# Will often want 64-bit precision when handling this kind of problem. In particular,
# one the channels of the Robertson problem will start at 0, increase to about 4e-5,
# and then decay back down to 0 again. We don't want that behaviour to get lost due to
# floating-point inaccuracies.
###########
jax.config.update("jax_enable_x64", True)


class Robertson(eqx.Module):
    k1: float
    k2: float
    k3: float

    def __call__(self, t, y, args):
        f0 = -self.k1 * y[0] + self.k3 * y[1] * y[2]
        f1 = self.k1 * y[0] - self.k2 * y[1] ** 2 - self.k3 * y[1] * y[2]
        f2 = self.k2 * y[1] ** 2
        return jnp.stack([f0, f1, f2])


robertson = Robertson(0.04, 3e7, 1e4)
solver = diffrax.kvaerno5(
    robertson,
    nonlinear_solver=diffrax.NewtonNonlinearSolver(rtol=1e-8, atol=1e-8, tol=1e-9),
)
t0 = 0.0
t1 = 100.0
y0 = jnp.array([1.0, 0.0, 0.0])
dt0 = 0.0002
saveat = diffrax.SaveAt(ts=jnp.array([0.0, 1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2]))
stepsize_controller = diffrax.IController(rtol=1e-8, atol=1e-8)
sol = diffrax.diffeqsolve(
    solver,
    t0=t0,
    t1=t1,
    y0=y0,
    dt0=dt0,
    saveat=saveat,
    stepsize_controller=stepsize_controller,
)
print("Results:")
for ti, yi in zip(sol.ts, sol.ys):
    print(f"t={ti.item()}, y={yi.tolist()}")

###########
# One should typically use adaptive step sizes when using implicit integrators. Every
# step of an implicit integrator involves solving a nonlinear problem via an iterative
# procedure -- but this iterative procedure can sometimes fail. When this happens we
# want to be able to use reduce to a smaller step size.
#
# (If you use a constant step size and the iterative procedure fails, then Diffrax
# will at least raise a helpful error message.)
###########

###########
# FAQ:
#
# Q: The integration is taking a long time.
# A: Implicit integrators take a really long time compared to explicit integrators.
#    This is normal. It's also the reason we don't use them unless we have to.
#
# Q: How do I pick tolerances?
# A: Pick tolerances for the `stepsize_controller` just like an explicit problem --
#    it's up to you what tolerance you want. Generally speaking you should use the same
#    `rtol`, `atol` for the `nonlinear_solver`. Finally, it's recommended to take `tol`
#    in the `nonlinear_solver` as being about 10-1000 times smaller than `atol`.
#    That said, these are rules-of-thumb. You can play around with these a bit.
#
# Q: What other options should I know about?
# A: The main other option worth knowing is the number of steps in the
#    `nonlinear_solver`, which can be set via `NewtonNonlinearSolver(max_steps=...)`.
#    (The default is 6.) Generally speaking this is a sensible default though.
###########
