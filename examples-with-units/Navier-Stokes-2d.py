# Copyright 2024 BDP Ecosystem Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================


import brainunit as u
import jax
import jax.numpy as jnp

import diffrax


# 定义离散化操作
def laplacian(phi, dx, dy):
    return (
        (jnp.roll(phi, -1, axis=0) - 2 * phi + jnp.roll(phi, 1, axis=0)) / dx ** 2 +
        (jnp.roll(phi, -1, axis=1) - 2 * phi + jnp.roll(phi, 1, axis=1)) / dy ** 2
    )


def solve_poisson(omega, dx, dy):
    Nx, Ny = omega.shape
    kx = jnp.fft.fftfreq(Nx, d=dx) * 2 * jnp.pi
    ky = jnp.fft.fftfreq(Ny, d=dy) * 2 * jnp.pi
    Kx, Ky = jnp.meshgrid(kx, ky, indexing='ij')
    K_squared = Kx ** 2 + Ky ** 2
    K_squared = jnp.where(K_squared == 0, 1.0, K_squared)  # 避免除以零
    omega_hat = jnp.fft.fft2(omega)
    psi_hat = -omega_hat / K_squared
    psi = jnp.real(jnp.fft.ifft2(psi_hat))
    return psi


def navier_stokes_ode(t, state, args):
    omega, psi = state
    dx, dy, nu, f = args

    # 计算速度场
    u = (jnp.roll(psi, -1, axis=1) - jnp.roll(psi, 1, axis=1)) / (2 * dy)
    v = -(jnp.roll(psi, -1, axis=0) - jnp.roll(psi, 1, axis=0)) / (2 * dx)

    # 计算涡度的对流项
    domega_dx = (jnp.roll(omega, -1, axis=0) - jnp.roll(omega, 1, axis=0)) / (2 * dx)
    domega_dy = (jnp.roll(omega, -1, axis=1) - jnp.roll(omega, 1, axis=1)) / (2 * dy)
    advect = u * domega_dx + v * domega_dy

    # 计算涡度的扩散项
    diffusion = nu * laplacian(omega, dx, dy)

    # 计算外力的涡度贡献
    f_vorticity = ((jnp.roll(f[1], -1, axis=0) - jnp.roll(f[1], 1, axis=0)) / (2 * dx) -
                   (jnp.roll(f[0], -1, axis=1) - jnp.roll(f[0], 1, axis=1)) / (2 * dy))

    # d(omega)/dt
    domega_dt = -advect + diffusion + f_vorticity

    # 由于 psi 是通过泊松方程即时更新的，dpsi_dt 为零
    dpsi_dt = jnp.zeros_like(psi)

    return (domega_dt, dpsi_dt)


def initial_conditions(Nx, Ny, dx, dy):
    # 示例：初始化为一个中心涡旋
    x = jnp.linspace(0, 2 * jnp.pi, Nx, endpoint=False)
    y = jnp.linspace(0, 2 * jnp.pi, Ny, endpoint=False)
    X, Y = jnp.meshgrid(x, y, indexing='ij')
    omega0 = jnp.sin(X) * jnp.sin(Y)
    psi0 = solve_poisson(omega0, dx, dy)
    return omega0, psi0


# 设置网格和物理参数
Nx, Ny = 64, 64
Lx, Ly = 2.0 * jnp.pi, 2.0 * jnp.pi
dx, dy = Lx / Nx, Ly / Ny
nu = 1e-3
f = (jnp.zeros((Nx, Ny)), jnp.zeros((Nx, Ny)))  # 无外力

# 初始条件
state0 = initial_conditions(Nx, Ny, dx, dy)

# 时间范围
t0, t1 = 0.0, 10.0

# ODE 参数
args = (dx, dy, nu, f)

# 定义 ODE 系统
ode_system = lambda t, state, args: navier_stokes_ode(t, state, args)

# 创建求解器
solver = diffrax.Dopri5()
stepsize_controller = diffrax.PIDController(rtol=1e-6, atol=1e-9)
ts = jnp.linspace(t0, t1, 1000)


# 自定义求解器步骤，以在每个时间步后更新 psi
def coupled_ode_system(t, state, args):
    omega, psi = state
    domega_dt, dpsi_dt = ode_system(t, state, args)
    return (domega_dt, dpsi_dt)


# 求解 ODE
solution = diffrax.diffeqsolve(
    terms=diffrax.ODETerm(coupled_ode_system),
    t0=t0,
    t1=t1,
    dt0=1e-3,
    y0=state0,
    args=args,
    solver=solver,
    stepsize_controller=stepsize_controller,
    saveat=diffrax.SaveAt(ts=ts)
)

# 提取结果
omega_sol = solution.ys[0]
psi_sol = solution.ys[1]


# 更新流函数
def update_psi(omega_sol, dx, dy):
    return jax.vmap(lambda t: solve_poisson(t, dx, dy))(omega_sol)


psi_updated = update_psi(omega_sol, dx, dy)

import matplotlib.pyplot as plt

# 选择时间步
time_index = -1  # 最后一个时间步

# 提取涡度和流函数
omega_final = omega_sol[time_index]
psi_final = psi_updated[time_index]

# 绘制涡度场
plt.figure(figsize=(6, 5))
plt.contourf(omega_final, levels=50, cmap='viridis')
plt.colorbar(label='Vorticity')
plt.title('Final Vorticity Field\n(simulation without units)')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()
