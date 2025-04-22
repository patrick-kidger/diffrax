from warnings import simplefilter


simplefilter(action="ignore", category=FutureWarning)

import timeit
from functools import partial

import diffrax
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from old_pid_controller import OldPIDController


t0 = 0
t1 = 5
dt0 = 0.5
y0 = 1.0
drift = diffrax.ODETerm(lambda t, y, args: -0.2 * y)


def diffusion_vf(t, y, args):
    return jnp.ones((), dtype=y.dtype)


def get_terms(key):
    bm = diffrax.VirtualBrownianTree(t0, t1, 2**-5, (), key)
    diffusion = diffrax.ControlTerm(diffusion_vf, bm)
    return diffrax.MultiTerm(drift, diffusion)


solver = diffrax.Heun()
step_ts = jnp.linspace(t0, t1, 129, endpoint=True)
pid_controller = diffrax.PIDController(
    rtol=0, atol=1e-3, dtmin=2**-9, dtmax=1.0, pcoeff=0.3, icoeff=0.7
)
new_controller = diffrax.ClipStepSizeController(
    pid_controller,
    step_ts=step_ts,
    store_rejected_steps=None,
)
old_controller = OldPIDController(
    rtol=0, atol=1e-3, dtmin=2**-9, dtmax=1.0, pcoeff=0.3, icoeff=0.7, step_ts=step_ts
)


@eqx.filter_jit
@partial(jax.vmap, in_axes=(0, None))
def solve(key, controller):
    term = get_terms(key)
    return diffrax.diffeqsolve(
        term,
        solver,
        t0,
        t1,
        dt0,
        y0,
        stepsize_controller=controller,
        saveat=diffrax.SaveAt(ts=step_ts),
    )


num_samples = 100
keys = jr.split(jr.PRNGKey(0), num_samples)


def do_timing(controller):
    @jax.jit
    @eqx.debug.assert_max_traces(max_traces=1)
    def time_controller_fun():
        sols = solve(keys, controller)
        assert sols.ys is not None
        assert sols.ys.shape == (num_samples, len(step_ts))
        return sols.ys

    def time_controller():
        jax.block_until_ready(time_controller_fun())

    return min(timeit.repeat(time_controller, number=3, repeat=20))


time_new = do_timing(new_controller)

time_old = do_timing(old_controller)

print(f"New controller: {time_new:.5} s, Old controller: {time_old:.5} s")

# How expensive is revisiting rejected steps?
revisiting_controller_short = diffrax.ClipStepSizeController(
    pid_controller,
    step_ts=step_ts,
    store_rejected_steps=10,
)

revisiting_controller_long = diffrax.ClipStepSizeController(
    pid_controller,
    step_ts=step_ts,
    store_rejected_steps=4096,
)

time_revisiting_short = do_timing(revisiting_controller_short)
time_revisiting_long = do_timing(revisiting_controller_long)

print(
    f"Revisiting controller\n"
    f"with buffer len 10:   {time_revisiting_short:.5} s\n"
    f"with buffer len 4096: {time_revisiting_long:.5} s"
)

# ======= RESULTS =======
# New controller: 0.23506 s, Old controller: 0.30735 s
# Revisiting controller
# with buffer len 10:   0.23636 s
# with buffer len 4096: 0.23965 s
