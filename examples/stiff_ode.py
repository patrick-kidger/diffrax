###########
#
# This example demonstrates the use of implicit integrators to handle stiff dynamical
# systems. In this case we consider the Robertson problem.
#
###########
import diffrax
import equinox as eqx
import jax
import jax.numpy as jnp


###########
# We often need 64-bit precision when handling this kind of problem. In particular,
# one of the channels of the Robertson problem starts at 0, increases to about 3.6e-5,
# and then decays back down to 0 again. We don't want that behaviour to get lost due to
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


###########
# One should typically use adaptive step sizes when using implicit integrators, so that
# the step size can be reduced if its nonlinear solve fails. It's generally worth
# setting the `rtol`, `atol` of the nonlinear solver to the same as is used in the
# stepsize controller.
###########
robertson = Robertson(0.04, 3e7, 1e4)
solver = diffrax.kvaerno5(
    robertson,
    nonlinear_solver=diffrax.NewtonNonlinearSolver(rtol=1e-8, atol=1e-8),
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
