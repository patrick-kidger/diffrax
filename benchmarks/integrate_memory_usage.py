"""does the t0 == t1 identity cond in diffrax's loop() cost a
copy of the save buffer?

Toy Workload: B trajectories of an n-dim linear ODE y' = A y, Tsit5 + PID.
Reports XLA temp memory against the size of one batched save buffer
(B * n_ts * n * 8 bytes for --saveat ts, B * max_steps * n * 8 bytes for --saveat steps)

How t0 reaches the cond (--t0):
    const    vmap over y0, t0 a Python constant: XLA can fold t0 == t1 and drop the cond
    shared   vmap over y0, t0 a traced scalar shared by all trajectories: the inner cond
             predicate is unbatched, so it stays a real cond
    batched  vmap over (y0, t0): the inner cond predicate is batched, so vmap turns it
             into a select over the whole save state
    novmap   no vmap, the batch is solved as one (B, n) ODE with a traced scalar t0

Compare temp memory between cases with the same --saveat and --consume flags, not
against an absolute ratio: other loop state also lands in temp.

    python repro_t0t1.py [--t0 {const,shared,batched,novmap}] [--saveat {ts,steps}]
                         [--consume] [--B B] [--n N] [--n-ts N_TS]

"""

import argparse
import json

import jax
import jax.numpy as jnp


jax.config.update("jax_enable_x64", True)

import diffrax


parser = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
)
parser.add_argument(
    "--consume", action="store_true", help="use sol.ys downstream of the solve"
)
parser.add_argument(
    "--t0",
    choices=["const", "shared", "batched", "novmap"],
    default="const",
    help="how t0 is passed in (see above)",
)
parser.add_argument(
    "--saveat", choices=["ts", "steps"], default="ts", help="SaveAt(ts=...) or steps"
)
parser.add_argument("--B", type=int, default=1000, help="number of trajectories")
parser.add_argument("--n", type=int, default=100, help="state dimension")
parser.add_argument("--n-ts", type=int, default=1000, help="number of save points")
cli = parser.parse_args()

consume = cli.consume
B, n, n_ts = cli.B, cli.n, cli.n_ts
max_steps = 4096

A = -jnp.eye(n) + 0.1 * jax.random.normal(jax.random.PRNGKey(0), (n, n)) / jnp.sqrt(n)
ts = jnp.linspace(0.0, 10.0, n_ts)
if cli.saveat == "ts":
    saveat = diffrax.SaveAt(ts=ts)
else:
    saveat = diffrax.SaveAt(steps=True)


def solve(y0, t0):
    sol = diffrax.diffeqsolve(
        diffrax.ODETerm(lambda t, y, args: y @ A.T),  # y is (n,) or, for novmap, (B, n)
        diffrax.Tsit5(),
        t0=t0,
        t1=10.0,
        dt0=0.01,
        y0=y0,
        saveat=saveat,
        stepsize_controller=diffrax.PIDController(rtol=1e-8, atol=1e-8),
        max_steps=max_steps,
    )
    ys = sol.ys
    assert ys is not None
    if consume:  # use the saved ys downstream to prevent XLA optimizing it away
        first_half = jnp.arange(len(ys)) < len(ys) // 2
        mask = first_half.reshape((-1,) + (1,) * (ys.ndim - 1))
        ys = jnp.where(mask, ys, 2.0 * ys)
    return ys


y0 = jax.random.normal(jax.random.PRNGKey(1), (B, n))
t0 = jnp.array(0.0)
if cli.t0 == "const":
    f, args = jax.jit(jax.vmap(lambda y: solve(y, 0.0))), (y0,)
elif cli.t0 == "shared":
    f, args = jax.jit(jax.vmap(solve, in_axes=(0, None))), (y0, t0)
elif cli.t0 == "batched":
    f, args = jax.jit(jax.vmap(solve)), (y0, jnp.zeros(B))
else:
    f, args = jax.jit(solve), (y0, t0)
m = f.lower(*args).compile().memory_analysis()
assert m is not None
buf = B * (n_ts if cli.saveat == "ts" else max_steps) * n * 8
ys = f(*args)  # also run it. Answer shouldnt change
print(
    json.dumps(
        dict(
            consume=consume,
            t0=cli.t0,
            saveat=cli.saveat,
            B=B,
            n=n,
            n_ts=n_ts,
            device=str(jax.devices()[0]),
            save_buffer_GB=round(buf / 1e9, 3),
            temp_GB=round(m.temp_size_in_bytes / 1e9, 3),
            out_GB=round(m.output_size_in_bytes / 1e9, 3),
            temp_in_save_buffers=round(m.temp_size_in_bytes / buf, 2),
            # dirty check to make sure output doesnt degrade; steps pads unused rows
            # with inf, so only sum the finite entries
            checksum=float(jnp.sum(jnp.where(jnp.isfinite(ys), ys, 0.0))),
        )
    ),
    flush=True,
)
