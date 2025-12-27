"""
Quantum Tunneling Simulation
============================

This example demonstrates how to solve the Time-Dependent Schrödinger Equation
using Diffrax. It simulates a Gaussian wave packet tunneling through a 
potential barrier.

Key JAX features used:
- `jax.jit` for compilation
- `diffrax.Tsit5` solver
- Complex number arithmetic in ODEs

![Simulation Result](simulation_result.png)
"""

import dataclasses
import jax
import jax.numpy as jnp
import diffrax
import matplotlib.pyplot as plt


# configure sim we use @dataclass to not write the __init__ method
@dataclasses.dataclass(frozen=True)
class SimConfig:
    N: int = 500  # Grid points
    L: float = 40.0  # Box size
    t0: float = 0.0  # Start time
    t1: float = 4.0  # End time
    num_frames: int = 200  # Number of frames to save (Fixed integer is safer than dt)

    # dx function can be called without () because of @property
    @property
    def dx(self):
        return self.L / self.N

    @property
    def x_grid(self):
        return jnp.linspace(-self.L / 2, self.L / 2, self.N)


# Physics


def get_potential(x_grid):
    return jnp.where((x_grid > -0.5) & (x_grid < 0.5), 20.0, 0.0)


def build_hamiltonian(config: SimConfig):
    x = config.x_grid
    V_arr = get_potential(x)
    V_matrix = jnp.diag(V_arr)

    main_diag = jnp.full(config.N, -2.0)
    off_diag = jnp.ones(config.N - 1)

    M = jnp.diag(main_diag, k=0) + jnp.diag(off_diag, k=1) + jnp.diag(off_diag, k=-1)
    KE = (-1.0 / (2.0 * config.dx**2)) * M

    H = KE + V_matrix
    return H, V_arr


def get_initial_psi(config: SimConfig, x0=-10.0, k0=5.0, sigma=1.0):
    x = config.x_grid
    norm = 1.0 / (jnp.pi * sigma**2) ** 0.25
    psi = norm * jnp.exp(-((x - x0) ** 2) / (2 * sigma**2)) * jnp.exp(1j * k0 * x)
    return psi


def vector_field(t, psi, args):
    H = args
    return -1j * (H @ psi)


# compile jit
@jax.jit
def run_simulation(hamiltonian, psi0, t0, t1, save_times):

    term = diffrax.ODETerm(vector_field)
    solver = diffrax.Tsit5()

    # Use the passed-in array instead of creating one inside
    saveat = diffrax.SaveAt(ts=save_times)

    step_controller = diffrax.PIDController(rtol=1e-8, atol=1e-8)

    sol = diffrax.diffeqsolve(
        term,
        solver,
        t0,
        t1,
        dt0=0.01,
        y0=psi0,
        args=hamiltonian,
        saveat=saveat,
        stepsize_controller=step_controller,
    )
    return sol


# Visualization


def create_plot(sol, config, V_arr, psi0):
    x = config.x_grid
    psi_final = sol.ys[-1]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # Plot 1
    ax1.set_title(f"Quantum Tunneling (t={config.t1})", fontsize=14)
    ax1.fill_between(
        x, 0, V_arr * 0.01, color="gray", alpha=0.3, label="Potential Barrier (Scaled)"
    )
    ax1.plot(x, jnp.abs(psi0) ** 2, "b--", alpha=0.6, label="Initial |$\psi$|$^2$")
    ax1.plot(x, jnp.abs(psi_final) ** 2, "r-", linewidth=2, label=f"Final |$\psi$|$^2$")
    ax1.set_ylabel("Probability Density")
    ax1.legend(loc="upper right")
    ax1.grid(True, alpha=0.3)

    # Plot 2
    ax2.set_title("Wave Function Components", fontsize=12)
    ax2.plot(x, jnp.real(psi_final), "g-", alpha=0.7, label="Real Part")
    ax2.plot(x, jnp.imag(psi_final), "orange", alpha=0.7, label="Imaginary Part")
    ax2.set_xlabel("Position ($x$)")
    ax2.set_ylabel("Amplitude")
    ax2.legend(loc="upper right")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("simulation_result.png", dpi=150)
    print("Plot saved as simulation_result.png")
    plt.show()


# main
def main():
    print("Initializing Quantum Simulation...")
    config = SimConfig(N=500, L=40.0, t0=0.0, t1=4.0, num_frames=200)

    print("Building Hamiltonian...")
    H_matrix, V_array = build_hamiltonian(config)
    psi0 = get_initial_psi(config)

    # always initialize outside the helper functions because helper functions are for abstractions
    save_times = jnp.linspace(config.t0, config.t1, config.num_frames)

    print("Solving Schrödinger Equation (JIT Compiled)...")
    # Pass 'save_times' into the function
    sol = run_simulation(H_matrix, psi0, config.t0, config.t1, save_times)

    print("Generating Plots...")
    create_plot(sol, config, V_array, psi0)


if __name__ == "__main__":
    main()
