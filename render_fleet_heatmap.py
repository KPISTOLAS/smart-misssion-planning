"""Standalone presentation figure: 3-UAV fleet trajectories on the IoT
vegetation-stress heatmap (the left-hand map panel, on its own).

Consumes drone_field_export.mat from the real MapGenerator + PriorityPlanner run.
"""
import numpy as np
import scipy.io as sio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

d = sio.loadmat("/workspace/drone_field_export.mat", squeeze_me=True)
H = d["H"].astype(float)
obstacle = d["obstacle"].astype(bool)
pos = np.asarray(d["pos"], dtype=float)          # [T, U, 2] (row, col)
starts = np.atleast_2d(np.asarray(d["starts"], dtype=float))
dx = float(d["dx"])
N, M = H.shape
U = pos.shape[1] if pos.ndim == 3 else 1

FLEET = ["#1f4e8c", "#c1121f", "#e08e00"]


def trim(path):
    chg = np.any(np.diff(path, axis=0) != 0, axis=1)
    L = (np.nonzero(chg)[0][-1] + 2) if chg.any() else 1
    p = path[:L]
    keep = np.concatenate([[True], np.any(np.diff(p, axis=0) != 0, axis=1)])
    return p[keep]


paths = [trim(pos[:, u, :]) for u in range(U)]


def to_xy(rc):
    return (rc[..., 1] - 0.5) * dx, (rc[..., 0] - 0.5) * dx


def draw_quad(ax, x, y, r, color):
    arm = r * 1.05
    for sx, sy in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
        ax.plot([x, x + sx * arm], [y, y + sy * arm], color=color, lw=2.2,
                solid_capstyle="round", zorder=9)
    for sx, sy in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
        ax.add_patch(Circle((x + sx * arm, y + sy * arm), r * 0.6,
                            facecolor="white", edgecolor=color, lw=1.7, zorder=10))
    ax.add_patch(Circle((x, y), r * 0.5, facecolor=color,
                        edgecolor="white", lw=1.3, zorder=11))


extent = [0, M * dx, 0, N * dx]
flown_frac = 0.55

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 13})
fig, ax = plt.subplots(figsize=(12.5, 8.6), dpi=200)
fig.patch.set_facecolor("white")

im = ax.imshow(H, origin="lower", extent=extent, cmap="RdYlGn_r",
               vmin=0, vmax=4, interpolation="bilinear", alpha=0.95, aspect="equal")
cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
cb.set_label("Vegetation stress index  $H$", fontsize=13, labelpad=10)
cb.set_ticks([0, 1, 2, 3, 4])
cb.ax.set_yticklabels(["0 healthy", "1", "2", "3", "4 critical"])

rr, cc = np.where(obstacle)
if rr.size:
    ax.scatter((cc - 0.5) * dx, (rr - 0.5) * dx, s=10, marker="^",
               c="#274018", alpha=0.5, linewidths=0, zorder=2)

for u, p in enumerate(paths):
    x, y = to_xy(p)
    nf = max(2, int(len(p) * flown_frac))
    ax.plot(x[nf - 1:], y[nf - 1:], "--", color=FLEET[u], lw=1.4, alpha=0.55,
            zorder=3, dashes=(5, 4))
    ax.plot(x[:nf], y[:nf], "-", color=FLEET[u], lw=2.7, zorder=5,
            label=f"UAV-{u + 1}")
    sx, sy = to_xy(starts[u])
    ax.scatter([sx], [sy], s=110, marker="s", facecolor="white",
               edgecolor=FLEET[u], linewidths=2.1, zorder=7)
    ax.add_patch(Circle((x[nf - 1], y[nf - 1]), dx * 2.3, facecolor="white",
                        alpha=0.25, zorder=6))
    draw_quad(ax, x[nf - 1], y[nf - 1], r=dx * 1.05, color=FLEET[u])

ax.set_xlim(extent[:2]); ax.set_ylim(extent[2:])
ax.set_xlabel("Easting (m)", fontsize=13)
ax.set_ylabel("Northing (m)", fontsize=13)
fig.suptitle("3-UAV Fleet Trajectories on the IoT Plant-Health Stress Map",
             fontsize=16.5, fontweight="bold", y=0.99)
ax.set_title(
    f"Capacitated-Voronoi workload partition   |   {M}\u00d7{N} grid @ {dx:.0f} m "
    f"({M*dx/1000:.1f}\u00d7{N*dx/1000:.1f} km)",
    fontsize=11.5, color="#555555", pad=10)
ax.legend(loc="upper right", framealpha=0.92, fontsize=12)
ax.grid(True, color="white", alpha=0.15, lw=0.6)
for s in ax.spines.values():
    s.set_edgecolor("#cccccc")

fig.tight_layout(rect=[0, 0, 1, 0.96])
out = "/workspace/fleet_stress_map_matlab.png"
fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
print("wrote", out)
