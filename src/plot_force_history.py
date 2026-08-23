from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


checkpoint_path = Path("data/re100_d20_tstar120_state.npz")
output_path = Path("data/re100_cl_history.png")

with np.load(checkpoint_path) as state:
    cl_history = state["Cl_history"].copy()
    diameter = float(state["D"])
    free_stream_velocity = float(state["U_inf"])
    t_star_final = float(state["t_star_final"])

t_star = np.arange(cl_history.size) * free_stream_velocity / diameter

fig, ax = plt.subplots(figsize=(11, 5.5), constrained_layout=True)
ax.plot(t_star, cl_history, color="tab:blue", linewidth=1.0)
ax.axhline(0.0, color="black", linewidth=0.8)
for marker in (40.0, 60.0, 80.0, 100.0, 120.0):
    ax.axvline(marker, color="0.5", linestyle="--", linewidth=0.8, alpha=0.75)

ax.set_xlim(0.0, t_star_final)
ax.set_xlabel("t*")
ax.set_ylabel("Cl")
ax.set_title("Lift coefficient history — Re = 100")
ax.grid(True, alpha=0.2)
fig.savefig(output_path, dpi=180)
plt.close(fig)

print(f"samples={len(cl_history)}")
print(f"output={output_path.resolve()}")
