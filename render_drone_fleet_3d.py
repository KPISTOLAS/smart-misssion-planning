"""Presentation figure: 3-UAV fleet over the PHM field.

Left  : 2D vegetation-stress heatmap with fleet trajectories.
Right : 3D terrain overlay with the fleet's elevation-aware flight paths.

Consumes drone_field_export.mat from the real MapGenerator + PriorityPlanner run.
"""
import numpy as np
import scipy.io as sio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from mpl_toolkits.mplot3d.art3d import Line3DCollection

d = sio.loadmat("/workspace/drone_field_export.mat", squeeze_me=True)
H = d["H"].astype(float)
Z = d["Z_m"].astype(float)
obstacle = d["obstacle"].astype(bool)
pos = np.asarray(d["pos"], dtype=float)          # [T, U, 2] (row, col)
starts = np.atleast_2d(np.asarray(d["starts"], dtype=float))
dx = float(d["dx"])
N, M = H.shape
U = pos.shape[1] if pos.ndim == 3 else 1
AGL = 32.0                                        # flight height above terrain (m)

FLEET = ["#1f4e8c", "#c1121f", "#e08e00"]         # UAV-1/2/3 colors
FLEET_L = ["#4f80c0", "#e05a5a", "#f2b53a"]


def trim(path):
    """Drop trailing padding (repeated last cell) and internal duplicates."""
    chg = np.any(np.diff(path, axis=0) != 0, axis=1)
    L = (np.nonzero(chg)[0][-1] + 2) if chg.any() else 1
    p = path[:L]
    keep = np.concatenate([[True], np.any(np.diff(p, axis=0) != 0, axis=1)])
    return p[keep]


paths = [trim(pos[:, u, :]) for u in range(U)]


def to_xy(rc):
    return (rc[..., 1] - 0.5) * dx, (rc[..., 0] - 0.5) * dx


def terr_z(rc):
    r = np.clip(np.round(rc[..., 0]).astype(int) - 1, 0, N - 1)
    c = np.clip(np.round(rc[..., 1]).astype(int) - 1, 0, M - 1)
    return Z[r, c]


extent = [0, M * dx, 0, N * dx]
flown_frac = 0.55

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12})
fig = plt.figure(figsize=(17.5, 8.2), dpi=200)
fig.patch.set_facecolor("white")

# ============================ Panel 1: 2D heatmap ============================
ax1 = fig.add_subplot(1, 2, 1)
im = ax1.imshow(H, origin="lower", extent=extent, cmap="RdYlGn_r",
                vmin=0, vmax=4, interpolation="bilinear", alpha=0.95, aspect="equal")
cb = fig.colorbar(im, ax=ax1, fraction=0.046, pad=0.02)
cb.set_label("Vegetation stress index  $H$", fontsize=12)
cb.set_ticks([0, 1, 2, 3, 4])
cb.ax.set_yticklabels(["0", "1", "2", "3", "4"])

rr, cc = np.where(obstacle)
if rr.size:
    ax1.scatter((cc - 0.5) * dx, (rr - 0.5) * dx, s=9, marker="^",
                c="#274018", alpha=0.5, linewidths=0, zorder=2)


def draw_quad(ax, x, y, r, color):
    arm = r * 1.05
    for sx, sy in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
        ax.plot([x, x + sx * arm], [y, y + sy * arm], color=color, lw=2.0,
                solid_capstyle="round", zorder=9)
    for sx, sy in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
        ax.add_patch(Circle((x + sx * arm, y + sy * arm), r * 0.6,
                            facecolor="white", edgecolor=color, lw=1.6, zorder=10))
    ax.add_patch(Circle((x, y), r * 0.5, facecolor=color,
                        edgecolor="white", lw=1.2, zorder=11))


for u, p in enumerate(paths):
    x, y = to_xy(p)
    nf = max(2, int(len(p) * flown_frac))
    ax1.plot(x[nf - 1:], y[nf - 1:], "--", color=FLEET[u], lw=1.3, alpha=0.55,
             zorder=3, dashes=(5, 4))
    ax1.plot(x[:nf], y[:nf], "-", color=FLEET[u], lw=2.5, zorder=5,
             label=f"UAV-{u + 1}")
    sx, sy = to_xy(starts[u])
    ax1.scatter([sx], [sy], s=90, marker="s", facecolor="white",
                edgecolor=FLEET[u], linewidths=2.0, zorder=7)
    ax1.add_patch(Circle((x[nf - 1], y[nf - 1]), dx * 2.2, facecolor="white",
                         alpha=0.25, zorder=6))
    draw_quad(ax1, x[nf - 1], y[nf - 1], r=dx * 1.0, color=FLEET[u])

ax1.set_xlim(extent[:2]); ax1.set_ylim(extent[2:])
ax1.set_xlabel("Easting (m)"); ax1.set_ylabel("Northing (m)")
ax1.set_title("Fleet trajectories on IoT stress heatmap", fontsize=13.5, pad=8)
ax1.legend(loc="upper right", framealpha=0.9, fontsize=11)

# ============================ Panel 2: 3D terrain ============================
ax2 = fig.add_subplot(1, 2, 2, projection="3d", computed_zorder=False)
Xg, Yg = np.meshgrid((np.arange(M) + 0.5) * dx, (np.arange(N) + 0.5) * dx)
Zc = Z - Z.min()                                  # land baseline at 0 m
zrng = max(Zc.max(), 1.0)
surf = ax2.plot_surface(Xg, Yg, Zc, cmap="gist_earth",
                        vmin=-0.45 * zrng, vmax=zrng * 1.05, linewidth=0,
                        antialiased=True, alpha=0.95, rstride=1, cstride=1,
                        zorder=1)
cb2 = fig.colorbar(surf, ax=ax2, fraction=0.03, pad=0.06, shrink=0.6)
cb2.set_label("Relative elevation (m)", fontsize=12)
cb2.set_ticks(np.round(np.linspace(0, zrng, 5)))

for u, p in enumerate(paths):
    x, y = to_xy(p)
    z = (terr_z(p) - Z.min()) + AGL
    nf = max(2, int(len(p) * flown_frac))
    pts = np.array([x[:nf], y[:nf], z[:nf]]).T.reshape(-1, 1, 3)
    segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
    ax2.add_collection3d(Line3DCollection(segs, colors=FLEET[u], linewidths=3.2,
                                          zorder=6))
    ax2.plot(x[nf - 1:], y[nf - 1:], z[nf - 1:], "--", color=FLEET_L[u], lw=1.3,
             alpha=0.7, zorder=6)
    ax2.scatter([x[nf - 1]], [y[nf - 1]], [z[nf - 1]], s=70, color=FLEET[u],
                edgecolor="white", linewidths=1.2, depthshade=False, zorder=10,
                label=f"UAV-{u + 1}")
    sx, sy = to_xy(starts[u])
    sz = (terr_z(starts[u]) - Z.min()) + AGL
    ax2.scatter([sx], [sy], [sz], s=70, marker="s", facecolor="white",
                edgecolor=FLEET[u], linewidths=1.8, depthshade=False, zorder=9)

ax2.set_xlabel("Easting (m)", labelpad=10)
ax2.set_ylabel("Northing (m)", labelpad=10)
ax2.set_zlabel("Elevation (m)", labelpad=6)
ax2.set_title("3D terrain overlay: elevation-aware fleet paths",
              fontsize=13.5, pad=2)
ax2.view_init(elev=34, azim=-58)
ax2.set_box_aspect((1.0, 0.78, 0.34))
ax2.set_zlim(0, max(zrng + AGL + 5, 60))
ax2.legend(loc="upper left", fontsize=10.5, framealpha=0.85)
ax2.grid(True, alpha=0.25)

fig.suptitle("Coordinated 3-UAV Fleet Surveying the IoT Plant-Health Field",
             fontsize=18, fontweight="bold", y=0.99)
fig.text(0.5, 0.935,
         f"MapGenerator + PriorityPlanner (capacitated Voronoi partition)   |   "
         f"{M}\u00d7{N} grid @ {dx:.0f} m   |   {AGL:.0f} m AGL flight",
         ha="center", fontsize=11.5, color="#555555")

fig.tight_layout(rect=[0, 0, 1, 0.92])
out = "/workspace/drone_fleet_3d_matlab.png"
fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
print("wrote", out)
