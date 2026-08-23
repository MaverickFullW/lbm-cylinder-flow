from pathlib import Path
import tempfile

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.patches import Circle, Rectangle

CASES = [(25, Path("data/re25_d20_longdomain_tstar120_state.npz")),
         (75, Path("data/re75_d20_longdomain_tstar120_state.npz")),
         (100, Path("data/re100_d20_longdomain_tstar120_state.npz"))]
OUTPUT = Path("data/reynolds_comparison_re25_re75_re100_final.gif")
FPS, XLIM, YLIM, PERCENTILE = 10, (-2.0, 24.0), (-4.0, 4.0), 99.5
AQUA, NAVY, INK = "#A8D8CF", "#0B2A4A", "#20262B"

cases = []
for expected_re, checkpoint in CASES:
    with np.load(checkpoint) as state:
        field = state["omega_frames"].copy()
        times = state["omega_t_star"].copy()
        diameter, u_inf = int(state["D"]), float(state["U_inf"])
        reynolds = float(state["Re"])
    if not np.isclose(reynolds, expected_re):
        raise ValueError(f"Unexpected Re={reynolds} in {checkpoint}")
    if not np.all(np.isfinite(field)) or not np.all(np.isfinite(times)):
        raise ValueError(f"Non-finite stored data in {checkpoint}")
    field *= diameter / u_inf
    cases.append({"Re": expected_re, "omega_star": field, "times": times,
                  "D": diameter, "U_inf": u_inf})

# Match stored frames only; vorticity is never interpolated.
common_times, common_indices = [], [[] for _ in cases]
for reference_index, reference_time in enumerate(cases[0]["times"]):
    matched, compatible = [reference_index], True
    for case in cases[1:]:
        nearest = int(np.argmin(np.abs(case["times"] - reference_time)))
        tolerance = 1e-10 * max(1.0, abs(float(reference_time)))
        if abs(float(case["times"][nearest] - reference_time)) > tolerance:
            compatible = False
            break
        matched.append(nearest)
    if compatible:
        common_times.append(float(reference_time))
        for indices, index in zip(common_indices, matched):
            indices.append(index)
common_times = np.asarray(common_times)
common_indices = [np.asarray(x, dtype=int) for x in common_indices]
if common_times.size == 0 or not np.isclose(common_times[-1], 120.0):
    raise ValueError("No compatible common frame range through t*=120")

# Preserve the existing exact common 99.5th-percentile scale.
total_values = sum(case["omega_star"].size for case in cases)
temporary = tempfile.NamedTemporaryFile(prefix="omega_star_", suffix=".dat", delete=False)
temporary_path = Path(temporary.name)
temporary.close()
try:
    combined = np.memmap(temporary_path, dtype=np.float64, mode="w+", shape=(total_values,))
    offset = 0
    for case in cases:
        values = case["omega_star"]
        for start in range(0, values.shape[0], 8):
            chunk = np.abs(values[start:start + 8]).ravel()
            combined[offset:offset + chunk.size] = chunk
            offset += chunk.size
    combined.flush()
    vmax = float(np.percentile(combined, PERCENTILE, overwrite_input=True))
    del combined
finally:
    temporary_path.unlink(missing_ok=True)

norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
cmap = LinearSegmentedColormap.from_list("blue_aqua_red", [
    (0.00, "#0033A0"), (0.20, "#0066CC"), (0.38, "#3399E6"),
    (0.47, "#86C5E8"), (0.50, AQUA), (0.53, "#F29A7E"),
    (0.62, "#E64B35"), (0.80, "#C51B1D"), (1.00, "#8B0000")])

diameter, u_inf = cases[0]["D"], cases[0]["U_inf"]
ny, nx = cases[0]["omega_star"].shape[1:]
X = (np.arange(nx) - 5.0 * diameter) / diameter
Y = (np.arange(ny) - ny / 2.0) / diameter
xi = np.flatnonzero((X >= XLIM[0]) & (X <= XLIM[1]))
yi = np.flatnonzero((Y >= YLIM[0]) & (Y <= YLIM[1]))
views = [case["omega_star"][:, yi[0]:yi[-1] + 1, xi[0]:xi[-1] + 1] for case in cases]
if OUTPUT.exists():
    raise FileExistsError(f"Refusing to overwrite existing file: {OUTPUT}")
frame = 0

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                     "axes.labelcolor": INK, "xtick.color": "#3A4147",
                     "ytick.color": "#3A4147"})
fig = plt.figure(figsize=(10.7, 11.0), facecolor="#F5F6F7")
grid = fig.add_gridspec(3, 2, left=.075, right=.925, bottom=.095, top=.865,
                       width_ratios=(1, .018), hspace=.16, wspace=.035)
axes = [fig.add_subplot(grid[row, 0]) for row in range(3)]
cax = fig.add_subplot(grid[:, 1])
descriptions = ("Steady wake", "Periodic vortex shedding", "Developed vortex street")
images = []
for panel, (ax, case, view, indices, description, letter) in enumerate(
        zip(axes, cases, views, common_indices, descriptions, "abc")):
    ax.set_facecolor(AQUA)
    image = ax.imshow(view[indices[frame]], origin="lower", aspect="equal",
                      extent=(*XLIM, *YLIM), cmap=cmap, norm=norm,
                      interpolation="bilinear", zorder=1)
    images.append(image)
    ax.set_axisbelow(True)
    ax.grid(color="#5F6B73", alpha=.12, linewidth=.5, linestyle="--")
    ax.add_patch(Circle((0, 0), .5, facecolor=AQUA, edgecolor="#263238",
                        linewidth=.8, zorder=3))
    ax.set(xlim=XLIM, ylim=YLIM, ylabel="y/D",
           xticks=np.arange(-2, 25, 2), yticks=np.arange(-4, 5, 2))
    ax.tick_params(axis="both", labelsize=8, length=3, width=.6)
    for spine in ax.spines.values():
        spine.set(linewidth=.7, color="#3A4147")
    label = rf"({letter})  $\bf{{Re\ =\ {case['Re']}}}$" + "\n" + description
    ax.text(.012, .955, label,
            transform=ax.transAxes, va="top", color="white", fontsize=8.5,
            linespacing=1.35, bbox=dict(boxstyle="square,pad=0.38",
            facecolor=NAVY, edgecolor="none", alpha=.92), zorder=4)
    if panel < 2:
        ax.tick_params(labelbottom=False)
    else:
        ax.set_xlabel("x/D", fontsize=9)

colorbar = fig.colorbar(images[0], cax=cax)
colorbar.outline.set_linewidth(.6)
colorbar.outline.set_edgecolor("#3A4147")
colorbar.ax.tick_params(labelsize=8, length=2.5, width=.5)
colorbar.ax.set_title(r"$\omega_z^*$", fontsize=18, pad=8, color=INK)
fig.text(cax.get_position().x0 + cax.get_position().width / 2, .068,
         r"$\omega_z^* = \frac{\omega_z D}{U_\infty}$",
         ha="center", va="center", fontsize=20, color="#4A535A")

fig.suptitle("Reynolds Number Effects on Circular-Cylinder Wake Dynamics",
             x=.5, y=.965, fontsize=15, fontweight="semibold", color="#172027")
subtitle = fig.text(.5, .925, rf"D2Q9 LBM (BGK)   |   D = {diameter} lu   |   "
         rf"$U_\infty$ = {u_inf:.4f} lu/ts   |   $t^*$ = {common_times[frame]:.1f}",
         ha="center", va="center", fontsize=9.5, color="#46515A")
panel_position = axes[0].get_position()
flow_y = panel_position.y1 + .026
fig.text(panel_position.x0, flow_y, r"$U_\infty$", ha="left", va="center",
         fontsize=16, color=NAVY)
axes[0].annotate("", xy=(panel_position.x0 + .087, flow_y),
                 xytext=(panel_position.x0 + .037, flow_y),
                 xycoords=fig.transFigure, textcoords=fig.transFigure,
                 arrowprops=dict(arrowstyle="-|>", color=NAVY, lw=1.6,
                                 mutation_scale=13),
                 annotation_clip=False)
fig.add_artist(Rectangle((0, 0), 1, .05, transform=fig.transFigure,
                         facecolor="#ECEFF1", edgecolor="none", zorder=0))
fig.text(.5, .025, "D2Q9 LBM (BGK)  •  Circular cylinder  •  D = 20 lu  •  "
         "U∞ = 0.0625 lu/ts  •  Common normalized-vorticity scale",
         ha="center", va="center", fontsize=8, color="#46515A")

def update(animation_frame):
    for image, view, indices in zip(images, views, common_indices):
        image.set_data(view[indices[animation_frame]])
    subtitle.set_text(
        rf"D2Q9 LBM (BGK)   |   D = {diameter} lu   |   "
        rf"$U_\infty$ = {u_inf:.4f} lu/ts   |   "
        rf"$t^*$ = {common_times[animation_frame]:.1f}"
    )
    return (*images, subtitle)


OUTPUT.parent.mkdir(parents=True, exist_ok=True)
animation = FuncAnimation(fig, update, frames=len(common_times),
                          interval=1000 / FPS, blit=False)
animation.save(OUTPUT, writer=PillowWriter(fps=FPS), dpi=120)
plt.close(fig)
print(f"frames={len(common_times)}")
print(f"fps={FPS}")
print(f"duration_seconds={len(common_times) / FPS:.12g}")
print(f"t_star_min={common_times.min():.12g}")
print(f"t_star_max={common_times.max():.12g}")
print(f"vmax_global={vmax:.12g}")
print(f"output={OUTPUT.resolve()}")
