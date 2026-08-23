from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.patches import Circle

checkpoint_path = Path("data/re100_d20_longdomain_tstar120_state.npz")
variants = [
    {
        "name": "white_green_blue",
        "output": Path("data/re100_d20_vorticity_final_white_green_blue.gif"),
        "background": "#FFFFFF",
        "edge": "black",
        "foreground": "black",
        "colors": [
            (0.00, "#08306B"), (0.20, "#2171B5"), (0.38, "#6BAED6"),
            (0.47, "#C6DBEF"), (0.50, "#FFFFFF"),
            (0.53, "#C7E9C0"), (0.62, "#74C476"),
            (0.80, "#238B45"), (1.00, "#005A32"),
        ],
    },
]
for variant in variants:
    if variant["output"].exists():
        raise FileExistsError(f"Refusing to overwrite existing file: {variant['output']}")

with np.load(checkpoint_path) as state:
    omega_frames = state["omega_frames"].copy()
    omega_t_star = state["omega_t_star"].copy()
    diameter = int(state["D"])
    reynolds_number = float(state["Re"])
if not np.isfinite(omega_frames).all():
    raise ValueError("The D=20 checkpoint contains non-finite omega frames")

frame_count, ny, nx = omega_frames.shape
xc, yc = 5.0 * diameter, ny / 2.0
xlim, ylim = (-2.0, 24.0), (-4.0, 4.0)
X = (np.arange(nx) - xc) / diameter
Y = (np.arange(ny) - yc) / diameter
x_indices = np.flatnonzero((X >= xlim[0]) & (X <= xlim[1]))
y_indices = np.flatnonzero((Y >= ylim[0]) & (Y <= ylim[1]))
omega_view = omega_frames[:, y_indices[0]:y_indices[-1] + 1,
                          x_indices[0]:x_indices[-1] + 1]
vmax = float(np.percentile(np.abs(omega_frames), 99.5))
norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
fps = 10


def save_variant(variant):
    background = variant["background"]
    foreground = variant["foreground"]
    cmap = LinearSegmentedColormap.from_list(variant["name"], variant["colors"])
    fig, ax = plt.subplots(figsize=(12, 5), constrained_layout=True)
    fig.patch.set_facecolor(background)
    ax.set_facecolor(background)
    image = ax.imshow(
        omega_view[0], origin="lower", aspect="equal", extent=(*xlim, *ylim),
        cmap=cmap, norm=norm, interpolation="bilinear",
    )
    ax.add_patch(Circle(
        (0.0, 0.0), 0.5, facecolor=background, edgecolor=variant["edge"],
        linewidth=1.0, zorder=3,
    ))
    colorbar = fig.colorbar(image, ax=ax, pad=0.02)
    colorbar.set_label(r"Vorticity $\omega_z$", color=foreground)
    colorbar.ax.tick_params(colors=foreground)
    colorbar.outline.set_edgecolor(foreground)
    ax.set(xlabel="x/D", ylabel="y/D", xlim=xlim, ylim=ylim)
    ax.xaxis.label.set_color(foreground)
    ax.yaxis.label.set_color(foreground)
    ax.tick_params(colors=foreground)
    for spine in ax.spines.values():
        spine.set_edgecolor(foreground)
    title = ax.set_title(
        f"Cylinder Flow \N{EM DASH} Re = {reynolds_number:.0f} "
        f"\N{EM DASH} t* = {omega_t_star[0]:.1f}", color=foreground,
    )

    def update(i):
        image.set_data(omega_view[i])
        title.set_text(
            f"Cylinder Flow \N{EM DASH} Re = {reynolds_number:.0f} "
            f"\N{EM DASH} t* = {omega_t_star[i]:.1f}"
        )
        return image, title

    animation = FuncAnimation(fig, update, frames=frame_count,
                              interval=1000 / fps, blit=False)
    animation.save(variant["output"], writer=PillowWriter(fps=fps), dpi=100)
    plt.close(fig)


for variant in variants:
    save_variant(variant)
    print(f"name={variant['output'].name}")
    print(f"vmax={vmax:.12g}")
    print("percentile=99.5")
    print(f"frames={frame_count}")
    print(f"fps={fps}")
    print(f"duration_seconds={frame_count / fps:.12g}")
    print(f"output={variant['output'].resolve()}")
