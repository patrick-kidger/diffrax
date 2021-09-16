import functools as ft
import pathlib

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


_here = pathlib.Path(__file__).resolve().parent


def lipswish(x):
    return 0.909 * jnn.silu(x)


class VectorField(eqx.Module):
    mlp: eqx.nn.MLP

    def __init__(self, *, extra_size, hidden_size, width_size, depth, key):
        super().__init__()
        self.mlp = eqx.nn.MLP(
            in_size=hidden_size + extra_size + 1,
            out_size=hidden_size,
            width_size=width_size,
            depth=depth,
            activation=lipswish,
            final_activation=jnn.tanh,
            key=key,
        )

    @eqx.filter_jit
    def __call__(self, t, y, args):
        return self.mlp(jnp.concatenate([t[None], y]))


class ControlledVectorField(eqx.Module):
    mlp: eqx.nn.MLP
    control_size: int
    hidden_size: int

    def __init__(self, *, control_size, hidden_size, width_size, depth, key):
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

    @eqx.filter_jit
    def __call__(self, t, y, args):
        return self.mlp(jnp.concatenate([t[None], y])).reshape(
            self.hidden_size, self.control_size
        )


class NeuralSDE(eqx.Module):
    initial: eqx.nn.MLP
    drift: VectorField
    diffusion: ControlledVectorField
    readout: eqx.nn.Linear
    auxiliary_drift: VectorField

    initial_noise_size: int
    noise_size: int

    def __init__(
        self,
        *,
        data_size,
        initial_noise_size,
        noise_size,
        hidden_size,
        width_size,
        depth,
        key,
    ):
        super().__init__()
        (
            initial_key,
            drift_key,
            diffusion_key,
            readout_key,
            aux_drift_key,
        ) = jrandom.split(key, 5)

        self.initial = eqx.nn.MLP(
            initial_noise_size, hidden_size, width_size, depth, key=initial_key
        )
        self.drift = VectorField(
            extra_size=0,
            hidden_size=hidden_size,
            width_size=width_size,
            depth=depth,
            key=drift_key,
        )
        self.diffusion = ControlledVectorField(
            control_size=noise_size,
            hidden_size=hidden_size,
            width_size=width_size,
            depth=depth,
            key=diffusion_key,
        )
        self.readout = eqx.nn.Linear(
            in_features=hidden_size, out_features=data_size, key=readout_key
        )
        self.auxiliary_drift = VectorField(
            extra_size=data_size,
            hidden_size=hidden_size,
            width_size=width_size,
            depth=depth,
            key=aux_drift_key,
        )

        self.initial_noise_size = initial_noise_size
        self.noise_size = noise_size

    @eqx.filter_jit
    def _initial(self, ts, ys, key):
        bm_key, init_key = jrandom.split(key, 2)
        bm = diffrax.UnsafeBrownianPath(bm_key, (self.noise_size,))
        init = jrandom.normal(init_key, (self.initial_noise_size,))
        y0 = self.initial(init)

        if ys is None:
            drift = self.drift
            diffusion = self.diffusion
        else:
            interp_ys = diffrax.linear_interpolation(
                ts, ys, replace_nans_at_start=0.0, fill_forward_nans_at_end=True
            )
            context = diffrax.LinearInterpolation(ts, interp_ys).evaluate
            drift, diffusion, y0, bm = diffrax.sde_kl_divergence(
                drift1=self.auxiliary_drift,
                drift2=self.drift,
                diffusion=self.diffusion,
                context=context,
                y0=y0,
                bm=bm,
            )
        drift = diffrax.ODETerm(drift)
        diffusion = diffrax.ControlTerm(diffusion, bm)
        term = diffrax.MultiTerm((drift, diffusion))
        solver = diffrax.ReversibleHeun(term)

        t0 = ts[0]
        t1 = ts[-1]
        dt0 = 1.0
        saveat = diffrax.SaveAt(ts=ts)
        return solver, t0, t1, y0, dt0, saveat

    @eqx.filter_jit
    def _sample(self, sol):
        return jax.vmap(self.readout)(sol.ys)

    def sample(self, ts, *, key):
        solver, t0, t1, y0, dt0, saveat = self._initial(ts, None, key)
        sol = diffrax.diffeqsolve(solver, t0, t1, y0, dt0, saveat=saveat)
        return self._sample(sol)

    def loss_gan(self, ts, ys, discriminator, *, key):
        sample_ys = self.sample(ts, key=key)
        real_score = discriminator(ts, ys)
        fake_score = discriminator(ts, sample_ys)
        return jnp.mean(real_score - fake_score)

    @eqx.filter_jit
    def _loss_vae(self, sol_with_kl, ys):
        hidden, kl_divergence = sol_with_kl.ys
        sample_ys = jax.vmap(self.readout)(hidden)
        kl_divergence = kl_divergence[-1]
        isnan = jnp.isnan(ys)
        sample_ys = jnp.where(isnan, 0, sample_ys)
        ys = jnp.where(isnan, 0, ys)
        return jnp.sum((sample_ys - ys) ** 2) + kl_divergence

    def loss_vae(self, ts, ys, *, key):
        solver, t0, t1, y0, dt0, saveat = self._initial(ts, ys, key)
        sol_with_kl = diffrax.diffeqsolve(solver, t0, t1, y0, dt0, saveat=saveat)
        return self._loss_vae(sol_with_kl, ys)


class NeuralCDE(eqx.Module):
    initial: eqx.nn.MLP
    cvf: ControlledVectorField
    readout: eqx.nn.Linear

    def __init__(self, *, data_size, hidden_size, width_size, depth, key):
        super().__init__()
        initial_key, cvf_key, readout_key = jrandom.split(key, 3)

        self.initial = eqx.nn.MLP(
            in_size=data_size + 1,
            out_size=hidden_size,
            width_size=width_size,
            depth=depth,
            key=initial_key,
        )
        self.cvf = ControlledVectorField(
            control_size=data_size + 1,
            hidden_size=hidden_size,
            width_size=width_size,
            depth=depth,
            key=cvf_key,
        )
        self.readout = eqx.nn.Linear(
            in_features=hidden_size, out_features=1, key=readout_key
        )

    @eqx.filter_jit
    def _initial(self, ts, ys):
        ys = jnp.concatenate([ts[:, None], ys], axis=1)
        ys = diffrax.linear_interpolation(
            ts, ys, replace_nans_at_start=0.0, fill_forward_nans_at_end=True
        )
        control = diffrax.LinearInterpolation(ts, ys)
        y0 = self.initial(ys[0])

        term = diffrax.ControlTerm(self.cvf, control)
        solver = diffrax.ReversibleHeun(term)

        t0 = ts[0]
        t1 = ts[-1]
        dt0 = 1.0
        return solver, t0, t1, y0, dt0

    @eqx.filter_jit
    def _readout(self, sol):
        return self.readout(sol.ys[-1])

    def __call__(self, ts, ys):
        solver, t0, t1, y0, dt0 = self._initial(ts, ys)
        sol = diffrax.diffeqsolve(solver, t0, t1, y0, dt0)
        return self._readout(sol)

    @eqx.filter_jit
    def clip_weights(self):
        leaves, treedef = jax.tree_flatten(
            self, is_leaf=lambda x: isinstance(x, eqx.nn.Linear)
        )
        new_leaves = []
        for leaf in leaves:
            if isinstance(leaf, eqx.nn.Linear):
                lim = 1 / leaf.out_features
                leaf = eqx.tree_at(
                    lambda x: x.weight, leaf, leaf.weight.clip(-lim, lim)
                )
            new_leaves.append(leaf)
        return jax.tree_unflatten(treedef, new_leaves)


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


@eqx.filter_grad
def loss(g_d, ts_i, ys_i, *, gan, vae, key, step):
    generator, discriminator = g_d  # Just the first argument is differentiated
    batch_size, _ = ts_i.shape
    key = jrandom.fold_in(key, step)
    key = jrandom.split(key, batch_size)
    if vae:
        vae_loss = jax.vmap(generator.loss_vae)(ts_i, ys_i, key=key)
        vae_loss = jnp.mean(vae_loss)
    else:
        vae_loss = 0
    if gan:
        gan_loss = jax.vmap(
            ft.partial(generator.loss_gan, discriminator=discriminator)
        )(ts_i, ys_i, key=key)
        gan_loss = jnp.mean(gan_loss)
    else:
        gan_loss = 0
    return vae_loss + gan_loss


@eqx.filter_jit
def update(
    generator, discriminator, g_opt_state, d_opt_state, g_optim, d_optim, g_grad, d_grad
):
    g_updates, g_opt_state = g_optim.update(g_grad, g_opt_state)
    d_updates, d_opt_state = d_optim.update(d_grad, d_opt_state)
    generator = eqx.apply_updates(generator, g_updates)
    discriminator = eqx.apply_updates(discriminator, d_updates)
    if discriminator is not None:
        discriminator = discriminator.clip_weights()
    return generator, discriminator, g_opt_state, d_opt_state


def main(
    initial_noise_size=5,
    noise_size=3,
    hidden_size=16,
    width_size=16,
    depth=1,
    generator_lr=2e-5,
    discriminator_lr=1e-4,
    batch_size=1024,
    steps=3000,
    steps_per_print=10,
    dataset_size=8192,
    out_path=_here / "neural_sde_samples.png",
    seed=5678,
    gan=True,
    vae=True,
):
    if not gan and not vae:
        raise ValueError(
            "Must train with respect to either or both of VAE and GAN losses."
        )

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
    evaluate_key = jrandom.split(evaluate_key, batch_size)

    ts, ys = jax.vmap(get_data)(data_key)
    _, _, data_size = ys.shape

    generator = NeuralSDE(
        data_size=data_size,
        initial_noise_size=initial_noise_size,
        noise_size=noise_size,
        hidden_size=hidden_size,
        width_size=width_size,
        depth=depth,
        key=generator_key,
    )
    if gan:
        discriminator = NeuralCDE(
            data_size=data_size,
            hidden_size=hidden_size,
            width_size=width_size,
            depth=depth,
            key=discriminator_key,
        )
    else:
        discriminator = None

    g_optim = optax.rmsprop(generator_lr)
    d_optim = optax.rmsprop(-discriminator_lr)
    g_opt_state = g_optim.init(eqx.filter(generator, eqx.is_array))
    d_opt_state = d_optim.init(eqx.filter(discriminator, eqx.is_array))

    trange = tqdm.tqdm(range(steps))
    infinite_dataloader = make_dataloader(
        (ts, ys), batch_size, loop=True, key=dataloader_key
    )

    for step, (ts_i, ys_i) in zip(trange, infinite_dataloader):
        g_grad, d_grad = loss(
            (generator, discriminator),
            ts_i,
            ys_i,
            gan=gan,
            vae=vae,
            key=train_key,
            step=step,
        )
        generator, discriminator, g_opt_state, d_opt_state = update(
            generator,
            discriminator,
            g_opt_state,
            d_opt_state,
            g_optim,
            d_optim,
            g_grad,
            d_grad,
        )

        if (step % steps_per_print) == 0 or step == steps - 1:
            total_vae_loss = 0
            total_gan_loss = 0
            num_batches = 0
            for ts_i, ys_i in make_dataloader(
                (ts, ys), batch_size, loop=False, key=dataloader_key
            ):
                if vae:
                    vae_loss = jax.vmap(generator.loss_vae)(
                        ts_i, ys_i, key=evaluate_key
                    )
                    total_vae_loss += jnp.mean(vae_loss).item()
                if gan:
                    gan_loss = jax.vmap(
                        ft.partial(generator.loss_gan, discriminator=discriminator)
                    )(ts_i, ys_i, key=evaluate_key)
                    total_gan_loss += jnp.mean(gan_loss).item()
                num_batches += 1
            total_vae_loss /= num_batches
            total_gan_loss /= num_batches
            msg = [f"Step: {step}"]
            if vae:
                msg.append(f"VAE loss: {total_vae_loss}")
            if gan:
                msg.append(f"GAN loss: {total_gan_loss}")
            trange.write(", ".join(msg))

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
    ys_sampled = jax.vmap(generator.sample)(
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
    fig.savefig(out_path)
    return locals()  # TODO


if __name__ == "__main__":
    fire.Fire(main)
