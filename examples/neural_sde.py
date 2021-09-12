import functools as ft

import diffrax
import equinox as eqx
import fire
import jax
import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jrandom
import matplotlib.pyplot as plt
import optax
import tqdm


def lipswish(x):
    return 0.909 * jnn.silu(x)


class VectorField(eqx.Module):
    mlp: eqx.nn.MLP

    def __init__(self, hidden_size, width_size, depth, *, key):
        super().__init__()
        self.mlp = eqx.nn.MLP(
            in_size=hidden_size + 1,
            out_size=hidden_size,
            width_size=width_size,
            depth=depth,
            activation=lipswish,
            final_activation=jnn.tanh,
            key=key,
        )

    @ft.partial(eqx.filter_jit, filter_spec=eqx.is_array)
    def __call__(self, t, y, args):
        del args
        return self.mlp(jnp.concatenate([t[None], y]))


class ControlledVectorField(eqx.Module):
    mlp: eqx.nn.MLP
    control_size: int
    hidden_size: int

    def __init__(self, control_size, hidden_size, width_size, depth, *, key):
        super().__init__()
        self.control_size = control_size
        self.hidden_size = hidden_size
        self.mlp = eqx.nn.MLP(
            in_size=hidden_size + 1,
            out_size=hidden_size * control_size,
            width_size=width_size,
            depth=depth,
            activation=lipswish,
            final_activation=jnn.tanh,
            key=key,
        )

    @ft.partial(eqx.filter_jit, filter_spec=eqx.is_array)
    def __call__(self, t, y, args):
        del args
        return self.mlp(jnp.concatenate([t[None], y])).reshape(
            self.hidden_size, self.control_size
        )


class DifferentialEquation(eqx.Module):
    initial: eqx.nn.MLP
    vf: VectorField  # drift
    cvf: ControlledVectorField  # diffusion
    readout: eqx.nn.Linear

    def __init__(
        self,
        initial_size,
        control_size,
        hidden_size,
        width_size,
        depth,
        readout_size,
        *,
        key,
    ):
        super().__init__()
        initial_key, vf_key, cvf_key, readout_key = jrandom.split(key, 4)

        self.initial = eqx.nn.MLP(
            initial_size, hidden_size, width_size, depth, key=initial_key
        )
        self.vf = VectorField(hidden_size, width_size, depth, key=vf_key)
        self.cvf = ControlledVectorField(
            control_size, hidden_size, width_size, depth, key=cvf_key
        )
        self.readout = eqx.nn.Linear(hidden_size, readout_size, key=readout_key)

    @ft.partial(eqx.filter_jit, filter_spec=eqx.is_array)
    def _initial(self, ts, init):
        t0 = ts[0]
        t1 = ts[-1]
        y0 = self.initial(init)
        dt0 = 1.0
        return t0, t1, y0, dt0

    @ft.partial(eqx.filter_jit, filter_spec=eqx.is_array)
    def _readout(self, ys):
        return jax.vmap(self.readout)(ys)

    def __call__(self, ts, init, control, saveat):
        vf = diffrax.ODETerm(self.vf)
        cvf = diffrax.ControlTerm(self.cvf, control)
        term = diffrax.MultiTerm((vf, cvf))
        solver = diffrax.ReversibleHeun(term)

        t0, t1, y0, dt0 = self._initial(ts, init)
        sol = diffrax.diffeqsolve(solver, t0, t1, y0, dt0, saveat=saveat)
        return self._readout(sol.ys)


class NeuralSDE(eqx.Module):
    diffeq: DifferentialEquation
    initial_noise_size: int
    noise_size: int

    def __init__(
        self,
        data_size,
        initial_noise_size,
        noise_size,
        hidden_size,
        width_size,
        depth,
        *,
        key,
    ):
        super().__init__()
        self.diffeq = DifferentialEquation(
            initial_size=initial_noise_size,
            control_size=noise_size,
            hidden_size=hidden_size,
            width_size=width_size,
            depth=depth,
            readout_size=data_size,
            key=key,
        )
        self.initial_noise_size = initial_noise_size
        self.noise_size = noise_size

    def __call__(self, ts, *, key):
        init_key, bm_key = jrandom.split(key, 2)
        init = jrandom.normal(init_key, (self.initial_noise_size,))
        control = diffrax.UnsafeBrownianPath(bm_key, (self.noise_size,))
        saveat = diffrax.SaveAt(ts=ts)
        return self.diffeq(ts, init, control, saveat)


class NeuralCDE(eqx.Module):
    diffeq: DifferentialEquation

    def __init__(self, data_size, hidden_size, width_size, depth, *, key):
        super().__init__()
        self.diffeq = DifferentialEquation(
            initial_size=data_size + 1,
            control_size=data_size,
            hidden_size=hidden_size,
            width_size=width_size,
            depth=depth,
            readout_size=1,
            key=key,
        )

    def __call__(self, ts, ys):
        ys = diffrax.linear_interpolation(
            ts, ys, replace_nans_at_start=0.0, fill_forward_nans_at_end=True
        )
        init = jnp.concatenate([ts[0, None], ys[0]])
        control = diffrax.LinearInterpolation(ts, ys)
        saveat = diffrax.SaveAt(t1=True)
        return self.diffeq(ts, init, control, saveat)


def get_data(key):
    bm_key, y0_key, drop_key = jrandom.split(key, 3)

    mu = 0.02
    theta = 0.1
    sigma = 0.4

    t0 = 0
    t1 = 63
    t_size = 64

    def drift(t, y, args):
        return mu * t - theta * y

    def diffusion(t, y, args):
        return 2 * sigma * t / t1

    bm = diffrax.UnsafeBrownianPath(bm_key, ())
    solver = diffrax.euler_maruyama(drift, diffusion, bm)
    y0 = jrandom.uniform(y0_key, (1,), minval=-1, maxval=1)
    dt0 = 0.1
    ts = jnp.linspace(t0, t1, t_size)
    saveat = diffrax.SaveAt(ts=ts)
    sol = diffrax.diffeqsolve(solver, t0, t1, y0, dt0, saveat=saveat)
    ys = sol.ys

    to_drop = jrandom.bernoulli(drop_key, 0.3, (t_size, 1))
    ys = jnp.where(to_drop, jnp.nan, ys)

    return ts, ys


def make_dataloader(arrays, batch_size, loop, *, key):
    dataset_size = arrays[0].shape[0]
    assert all(array.shape[0] == dataset_size for array in arrays)
    indices = jnp.arange(dataset_size)
    while True:
        perm = jrandom.permutation(key, indices)
        (key,) = jrandom.split(key, 1)
        start = 0
        end = batch_size
        while end < dataset_size:
            batch_perm = perm[start:end]
            yield tuple(array[batch_perm] for array in arrays)
            start = end
            end = start + batch_size
        if not loop:
            break


def loss(generator, discriminator, ts_i, ys_i, key, step=0):
    batch_size, _ = ts_i.shape
    key = jrandom.fold_in(key, step)
    key = jrandom.split(key, batch_size)
    fake_ys_i = jax.vmap(generator)(ts_i, key=key)
    real_score = jax.vmap(discriminator)(ts_i, ys_i)
    fake_score = jax.vmap(discriminator)(ts_i, fake_ys_i)
    return jnp.mean(real_score - fake_score)


@ft.partial(eqx.filter_value_and_grad, filter_spec=eqx.is_inexact_array)
def loss_and_grads(g_d, ts_i, ys_i, key, step):
    generator, discriminator = g_d  # We differentiate just the first argument
    return loss(generator, discriminator, ts_i, ys_i, key, step)


def main(
    initial_noise_size=5,
    noise_size=3,
    hidden_size=16,
    width_size=16,
    depth=1,
    generator_lr=2e-4,
    discriminator_lr=1e-3,
    batch_size=1024,
    steps=10000,
    steps_per_print=10,
    dataset_size=8192,
    seed=5678,
):

    key = jrandom.PRNGKey(seed)
    (
        data_key,
        generator_key,
        discriminator_key,
        dataloader_key,
        train_key,
        evaluate_key,
        sample_key,
    ) = jrandom.split(key, 7)
    data_key = jrandom.split(data_key, dataset_size)

    ts, ys = jax.vmap(get_data)(data_key)
    _, _, data_size = ys.shape

    generator = NeuralSDE(
        data_size,
        initial_noise_size,
        noise_size,
        hidden_size,
        width_size,
        depth,
        key=generator_key,
    )
    discriminator = NeuralCDE(
        data_size, hidden_size, width_size, depth, key=discriminator_key
    )

    g_optim = optax.adam(generator_lr, b1=0)
    d_optim = optax.adam(-discriminator_lr, b1=0)
    g_opt_state = g_optim.init(eqx.filter(generator, eqx.is_array))
    d_opt_state = d_optim.init(eqx.filter(discriminator, eqx.is_array))

    trange = tqdm.tqdm(range(steps))
    infinite_dataloader = make_dataloader(
        (ts, ys), batch_size, loop=True, key=dataloader_key
    )
    for step, (ts_i, ys_i) in zip(trange, infinite_dataloader):
        score, (g_grad, d_grad) = loss_and_grads(
            (generator, discriminator), ts_i, ys_i, train_key, step
        )
        g_updates, g_opt_state = g_optim.update(g_grad, g_opt_state)
        d_updates, d_opt_state = d_optim.update(d_grad, d_opt_state)
        generator = eqx.apply_updates(generator, g_updates)
        discriminator = eqx.apply_updates(discriminator, d_updates)

        if (step % steps_per_print) == 0 or step == steps - 1:
            total_score = 0
            num_batches = 0
            for ts_i, ys_i in make_dataloader(
                (ts, ys), batch_size, loop=False, key=evaluate_key
            ):
                score = loss(generator, discriminator, ts_i, ys_i, sample_key)
                total_score += score.item()
                num_batches += 1
            trange.write(f"Step: {step}, Loss: {total_score / num_batches}")

    # Plot samples
    fig, ax = plt.subplots()
    num_samples = min(50, dataset_size)
    ts_to_plot = ts[:num_samples]
    ys_to_plot = ys[:num_samples]

    def _interp(ti, yi):
        return diffrax.linear_interpolation(
            ti, yi, replace_nans_at_start=0.0, fill_forward_nans_at_end=True
        )

    ys_to_plot = jax.vmap(_interp)(ts_to_plot, ys_to_plot)[..., 0]
    ys_sampled = jax.vmap(generator)(
        ts_to_plot, key=jrandom.split(sample_key, num_samples)
    )[..., 0]
    kwargs = dict(label="Real")
    for ti, yi in zip(ts_to_plot, ys_to_plot):
        ax.plot(ti, yi, c="dodgerblue", linewidth=0.5, alpha=0.7, **kwargs)
        kwargs = {}
    kwargs = dict(label="Generated")
    for ti, yi in zip(ts_to_plot, ys_sampled):
        ax.plot(ti, yi, c="crimson", linewidth=0.5, alpha=0.7, **kwargs)
        kwargs = {}
    ax.set_title(f"{num_samples} samples from both real and generated distributions.")
    fig.legend()
    fig.tight_layout()
    fig.savefig("neural_sde_samples.png")


if __name__ == "__main__":
    fire.Fire(main)
