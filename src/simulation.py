import argparse
import time
from pathlib import Path

import numpy as np

from src.geometry import D, cylinder, nx, ny
from src.lbm import (
    collision, cylinder_force, equilibrium, macroscopic, open_outlet_right,
    streaming_with_bounce_back, velocity_inlet_left,
)

U_INF = 0.0625
FRAME_STRIDE = 100
PERTURBATION_RATIO = 1.0e-4


def run_simulation(reynolds, t_star_final, frame_stride=FRAME_STRIDE):
    """Run a new D2Q9 BGK cylinder-flow case from a uniform initial state."""
    if reynolds <= 0.0 or not np.isfinite(reynolds):
        raise ValueError("Reynolds number must be finite and positive")
    if t_star_final <= 0.0 or not np.isfinite(t_star_final):
        raise ValueError("Final t* must be finite and positive")
    if frame_stride <= 0:
        raise ValueError("Frame stride must be positive")
    if D != 20 or nx != 600 or ny != 200 or cylinder.shape != (ny, nx):
        raise ValueError(f"Unexpected geometry: {D=}, {nx=}, {ny=}")

    nu = U_INF * D / reynolds
    tau = 3.0 * nu + 0.5
    epsilon = U_INF * PERTURBATION_RATIO
    steps = int(round(t_star_final * D / U_INF))

    rho = np.ones((ny, nx), dtype=float)
    u = np.zeros((ny, nx, 2), dtype=float)
    u[:, :, 0] = U_INF
    y_coordinate = np.arange(ny, dtype=float)[:, None]
    u[:, :, 1] = epsilon * np.sin(2.0 * np.pi * y_coordinate / ny)
    u[cylinder] = 0.0
    f = equilibrium(rho, u)

    fluid = ~cylinder
    Fx_history, Fy_history = [], []
    omega_frames, omega_frame_steps = [], []
    rho_observed_min, rho_observed_max = np.inf, -np.inf
    speed_observed_max = -np.inf
    start_time = time.perf_counter()

    for step in range(1, steps + 1):
        rho, u = macroscopic(f)
        feq = equilibrium(rho, u)
        f_post = collision(f, feq, tau)
        Fx, Fy = cylinder_force(f_post, cylinder)
        Fx_history.append(Fx)
        Fy_history.append(Fy)

        f_next = streaming_with_bounce_back(f_post, cylinder)
        velocity_inlet_left(f_next, ux=U_INF, uy=0.0)
        open_outlet_right(f_next)
        f = f_next

        current_t_star = step * U_INF / D
        if not np.all(np.isfinite(f)):
            raise FloatingPointError(
                f"Non-finite f detected at step={step}, t*={current_t_star:.12g}"
            )
        rho_checked, u_checked = macroscopic(f)
        rho_fluid = rho_checked[fluid]
        speed_fluid = np.linalg.norm(u_checked[fluid], axis=1)
        if not (np.all(np.isfinite(rho_fluid))
                and np.all(np.isfinite(speed_fluid))):
            raise FloatingPointError(
                "Non-finite macroscopic field detected at "
                f"step={step}, t*={current_t_star:.12g}"
            )
        rho_observed_min = min(rho_observed_min, float(rho_fluid.min()))
        rho_observed_max = max(rho_observed_max, float(rho_fluid.max()))
        speed_observed_max = max(speed_observed_max, float(speed_fluid.max()))

        if step % frame_stride == 0:
            omega = (np.gradient(u_checked[:, :, 1], axis=1)
                     - np.gradient(u_checked[:, :, 0], axis=0))
            omega_frames.append(omega)
            omega_frame_steps.append(step)

    elapsed = time.perf_counter() - start_time
    Fx_history = np.asarray(Fx_history)
    Fy_history = np.asarray(Fy_history)
    force_scale = 0.5 * U_INF**2 * D
    omega_frames = (np.asarray(omega_frames) if omega_frames else
                    np.empty((0, ny, nx), dtype=float))
    omega_frame_steps = np.asarray(omega_frame_steps, dtype=int)

    return {
        "f": f,
        "Fx_history": Fx_history,
        "Fy_history": Fy_history,
        "Cd_history": Fx_history / force_scale,
        "Cl_history": Fy_history / force_scale,
        "omega_frames": omega_frames,
        "omega_frame_steps": omega_frame_steps,
        "omega_t_star": omega_frame_steps * U_INF / D,
        "D": D,
        "Re": float(reynolds),
        "U_inf": U_INF,
        "nu": nu,
        "tau": tau,
        "epsilon": epsilon,
        "steps": steps,
        "t_star_final": float(t_star_final),
        "elapsed_seconds": elapsed,
        "rho_fluid_min": rho_observed_min,
        "rho_fluid_max": rho_observed_max,
        "speed_fluid_max": speed_observed_max,
    }


def save_checkpoint(output, result):
    """Save a simulation result without overwriting an existing checkpoint."""
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing file: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **result)


def build_parser():
    parser = argparse.ArgumentParser(
        description="Run a new D2Q9 BGK circular-cylinder flow case."
    )
    parser.add_argument("--re", type=float, required=True,
                        help="Reynolds number (positive).")
    parser.add_argument("--t-star", type=float, required=True,
                        help="Final nondimensional time t* (positive).")
    parser.add_argument("--output", type=Path, required=True,
                        help="Destination .npz checkpoint path.")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    result = run_simulation(args.re, args.t_star)
    save_checkpoint(args.output, result)
    print(f"steps={result['steps']}")
    print(f"elapsed_seconds={result['elapsed_seconds']:.12g}")
    print(f"nu={result['nu']:.12g}")
    print(f"tau={result['tau']:.12g}")
    print(f"frames_total={len(result['omega_frames'])}")
    print(f"rho_fluid_min={result['rho_fluid_min']:.12g}")
    print(f"rho_fluid_max={result['rho_fluid_max']:.12g}")
    print(f"speed_fluid_max={result['speed_fluid_max']:.12g}")
    print(f"checkpoint={args.output.resolve()}")


if __name__ == "__main__":
    main()
