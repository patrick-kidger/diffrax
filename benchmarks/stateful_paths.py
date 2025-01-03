import math
from typing import cast, Optional, Union

import diffrax
import equinox as eqx
import equinox.internal as eqxi
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
import lineax.internal as lxi
from jaxtyping import PRNGKeyArray, PyTree
from lineax.internal import complex_to_real_dtype


class OldBrownianPath(diffrax.AbstractBrownianPath):
    shape: PyTree[jax.ShapeDtypeStruct] = eqx.field(static=True)
    levy_area: type[
        Union[
            diffrax.BrownianIncrement,
            diffrax.SpaceTimeLevyArea,
            diffrax.SpaceTimeTimeLevyArea,
        ]
    ] = eqx.field(static=True)
    key: PRNGKeyArray
    precompute: Optional[int] = eqx.field(static=True)

    def __init__(
        self,
        shape,
        key,
        levy_area=diffrax.BrownianIncrement,
        precompute=None,
    ):
        self.shape = (
            jax.ShapeDtypeStruct(shape, lxi.default_floating_dtype())
            if diffrax._misc.is_tuple_of_ints(shape)
            else shape
        )
        self.key = key
        self.levy_area = levy_area
        self.precompute = precompute

        if any(
            not jnp.issubdtype(x.dtype, jnp.inexact)
            for x in jtu.tree_leaves(self.shape)
        ):
            raise ValueError("OldBrownianPath dtypes all have to be floating-point.")

    @property
    def t0(self):
        return -jnp.inf

    @property
    def t1(self):
        return jnp.inf

    def init(
        self,
        t0,
        t1,
        y0,
        args,
    ):
        return None

    def __call__(
        self,
        t0,
        brownian_state,
        t1=None,
        left=True,
        use_levy=False,
    ):
        return self.evaluate(t0, t1, left, use_levy), brownian_state

    @eqx.filter_jit
    def evaluate(
        self,
        t0,
        t1=None,
        left=True,
        use_levy=False,
    ):
        del left
        if t1 is None:
            dtype = jnp.result_type(t0)
            t1 = t0
            t0 = jnp.array(0, dtype)
        else:
            with jax.numpy_dtype_promotion("standard"):
                dtype = jnp.result_type(t0, t1)
            t0 = jnp.astype(t0, dtype)
            t1 = jnp.astype(t1, dtype)
        t0 = eqxi.nondifferentiable(t0, name="t0")
        t1 = eqxi.nondifferentiable(t1, name="t1")
        t1 = cast(diffrax._custom_types.RealScalarLike, t1)
        t0_ = diffrax._misc.force_bitcast_convert_type(t0, jnp.int32)
        t1_ = diffrax._misc.force_bitcast_convert_type(t1, jnp.int32)
        key = jr.fold_in(self.key, t0_)
        key = jr.fold_in(key, t1_)
        key = diffrax._misc.split_by_tree(key, self.shape)
        out = jtu.tree_map(
            lambda key, shape: self._evaluate_leaf(
                t0, t1, key, shape, self.levy_area, use_levy
            ),
            key,
            self.shape,
        )
        if use_levy:
            out = diffrax._custom_types.levy_tree_transpose(self.shape, out)
            assert isinstance(out, self.levy_area)
        return out

    @staticmethod
    def _evaluate_leaf(
        t0,
        t1,
        key,
        shape,
        levy_area,
        use_levy,
    ):
        w_std = jnp.sqrt(t1 - t0).astype(shape.dtype)
        dt = jnp.asarray(t1 - t0, dtype=complex_to_real_dtype(shape.dtype))

        if levy_area is diffrax.SpaceTimeTimeLevyArea:
            key_w, key_hh, key_kk = jr.split(key, 3)
            w = jr.normal(key_w, shape.shape, shape.dtype) * w_std
            hh_std = w_std / math.sqrt(12)
            hh = jr.normal(key_hh, shape.shape, shape.dtype) * hh_std
            kk_std = w_std / math.sqrt(720)
            kk = jr.normal(key_kk, shape.shape, shape.dtype) * kk_std
            levy_val = diffrax.SpaceTimeTimeLevyArea(dt=dt, W=w, H=hh, K=kk)

        elif levy_area is diffrax.SpaceTimeLevyArea:
            key_w, key_hh = jr.split(key, 2)
            w = jr.normal(key_w, shape.shape, shape.dtype) * w_std
            hh_std = w_std / math.sqrt(12)
            hh = jr.normal(key_hh, shape.shape, shape.dtype) * hh_std
            levy_val = diffrax.SpaceTimeLevyArea(dt=dt, W=w, H=hh)
        elif levy_area is diffrax.BrownianIncrement:
            w = jr.normal(key, shape.shape, shape.dtype) * w_std
            levy_val = diffrax.BrownianIncrement(dt=dt, W=w)
        else:
            assert False

        if use_levy:
            return levy_val
        return w


# https://github.com/patrick-kidger/diffrax/issues/517
key = jax.random.key(42)
t0 = 0
t1 = 100
y0 = 1.0
ndt = 4000
dt = (t1 - t0) / (ndt - 1)
drift = lambda t, y, args: -y
diffusion = lambda t, y, args: 0.2

brownian_motion = diffrax.VirtualBrownianTree(t0, t1, tol=1e-3, shape=(), key=key)
ubp = OldBrownianPath(shape=(), key=key)
new_ubp = diffrax.UnsafeBrownianPath(shape=(), key=key)
new_ubp_pre = diffrax.UnsafeBrownianPath(shape=(), key=key, precompute=True)
solver = diffrax.Euler()
terms = diffrax.MultiTerm(
    diffrax.ODETerm(drift), diffrax.ControlTerm(diffusion, brownian_motion)
)
terms_old = diffrax.MultiTerm(
    diffrax.ODETerm(drift), diffrax.ControlTerm(diffusion, ubp)
)
terms_new = diffrax.MultiTerm(
    diffrax.ODETerm(drift), diffrax.ControlTerm(diffusion, new_ubp)
)
terms_new_precompute = diffrax.MultiTerm(
    diffrax.ODETerm(drift), diffrax.ControlTerm(diffusion, new_ubp_pre)
)
saveat = diffrax.SaveAt(ts=jnp.linspace(t0, t1, ndt))


@jax.jit
def diffrax_vbt():
    return diffrax.diffeqsolve(terms, solver, t0, t1, dt0=dt, y0=y0, saveat=saveat).ys


@jax.jit
def diffrax_old():
    return diffrax.diffeqsolve(
        terms_old, solver, t0, t1, dt0=dt, y0=y0, saveat=saveat
    ).ys


@jax.jit
def diffrax_new():
    return diffrax.diffeqsolve(
        terms_new, solver, t0, t1, dt0=dt, y0=y0, saveat=saveat
    ).ys


@jax.jit
def diffrax_new_pre():
    return diffrax.diffeqsolve(
        terms_new_precompute, solver, t0, t1, dt0=dt, y0=y0, saveat=saveat
    ).ys


_ = diffrax_vbt().block_until_ready()
_ = diffrax_old().block_until_ready()
_ = diffrax_new().block_until_ready()
_ = diffrax_new_pre().block_until_ready()

from timeit import Timer


num_runs = 10

timer = Timer(stmt="_ = diffrax_vbt().block_until_ready()", globals=globals())
total_time = timer.timeit(number=num_runs)
print(f"VBT: {total_time / num_runs:.6f}")

timer = Timer(stmt="_ = diffrax_old().block_until_ready()", globals=globals())
total_time = timer.timeit(number=num_runs)
print(f"Old UBP: {total_time / num_runs:.6f}")

timer = Timer(stmt="_ = diffrax_new().block_until_ready()", globals=globals())
total_time = timer.timeit(number=num_runs)
print(f"New UBP: {total_time / num_runs:.6f}")

timer = Timer(stmt="_ = diffrax_new_pre().block_until_ready()", globals=globals())
total_time = timer.timeit(number=num_runs)
print(f"New UBP + Precompute: {total_time / num_runs:.6f}")

"""
Results on Mac M1 CPU:
VBT: 0.282765
Old UBP: 0.015823
New UBP: 0.013105
New UBP + Precompute: 0.002506

Results on A100 GPU:
VBT: 3.881952
Old UBP: 0.337173
New UBP: 0.364158
New UBP + Precompute: 0.325521

GPU being much slower isn't unsurprising and is a common trend for
small-medium sized SDEs with VFs that are relatively cheap to evaluate
(i.e. not neural networks).
"""
