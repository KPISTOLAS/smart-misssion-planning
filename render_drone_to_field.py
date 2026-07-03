"""Render a presentation-quality figure of the UAV ingressing the plant-health field.

Consumes drone_field_export.mat produced by the real MATLAB/Octave pipeline
(MapGenerator + PriorityPlanner) and draws the vegetation-stress heatmap, the
tree/obstacle field, and the drone's flown + planned trajectory.
"""
import numpy as np
import scipy.io as sio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch
from matplotlib.collections import LineCollection

d = sio.loadmat("/workspace/drone_field_export.mat", squeeze_me=True)
H = d["H"].astype(float)
obstacle = d["obstacle"].astype(bool)
pos = np.asarray(d["pos"], dtype=float)          # [T, U, 2] or [T, 2]
starts = np.atleast_2d(np.asarray(d["starts"], dtype=float))
dx = float(d["dx"])
N, M = H.shape

if pos.ndim == 3:
    traj = pos[:, 0, :]        # single lead UAV (row, col)
else:
    traj = pos

# Collapse consecutive duplicate padding at the tail.
keep = np.concatenate([[True], np.any(np.diff(traj, axis=0) != 0, axis=1)])
traj = traj[keep]

# cell (row,col) -> metric (x,y)
def to_xy(rc):
    r = rc[..., 0]
    c = rc[..., 1]
    return (c - 0.5) * dx, (r - 0.5) * dx

tx, ty = to_xy(traj)

# Portion already flown vs. still planned ("moving to / into the field").
flown_frac = 0.34
nflow = max(2, int(len(traj) * flown_frac))

extent = [0, M * dx, 0, N * dx]

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 13,
})

fig, ax = plt.subplots(figsize=(13.5, 8.4), dpi=200)
fig.patch.set_facecolor("white")

# --- Field: vegetation stress heatmap (green healthy -> red stressed) ---
im = ax.imshow(H, origin="lower", extent=extent, cmap="RdYlGn_r",
               vmin=0, vmax=4, interpolation="bilinear", alpha=0.95, aspect="equal")

cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
cbar.set_label("Vegetation stress index  $H$", fontsize=13, labelpad=10)
cbar.set_ticks([0, 1, 2, 3, 4])
cbar.ax.set_yticklabels(["0 healthy", "1", "2", "3", "4 critical"])

# --- Trees / obstacles ---
rr, cc = np.where(obstacle)
if rr.size:
    ox = (cc - 0.5) * dx
    oy = (rr - 0.5) * dx
    ax.scatter(ox, oy, s=10, marker="^", c="#274018", alpha=0.55,
               linewidths=0, zorder=2)

# --- Planned (remaining) path: thin dashed ---
ax.plot(tx[nflow - 1:], ty[nflow - 1:], "--", color="#1b3a5b", lw=1.4,
        alpha=0.6, zorder=3, dashes=(5, 4))

# --- Flown path: bold gradient-ish solid with subtle glow ---
pts = np.array([tx[:nflow], ty[:nflow]]).T.reshape(-1, 1, 2)
segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
lc = LineCollection(segs, colors="#0b1e33", linewidths=5.5, alpha=0.28, zorder=4)
ax.add_collection(lc)
ax.plot(tx[:nflow], ty[:nflow], "-", color="#0d3b66", lw=2.6, zorder=5)

# --- Direction arrows along flown path ---
for f in (0.5, 0.85):
    k = max(1, int(nflow * f))
    if k + 1 < nflow:
        ax.add_patch(FancyArrowPatch(
            (tx[k - 1], ty[k - 1]), (tx[k], ty[k]),
            arrowstyle="-|>", mutation_scale=22, color="#0d3b66", zorder=6))

# --- Depot / launch point ---
sx, sy = to_xy(starts[0])
ax.scatter([sx], [sy], s=190, marker="s", facecolor="white",
           edgecolor="#0d3b66", linewidths=2.2, zorder=7)
ax.annotate("Launch / depot", (sx, sy), textcoords="offset points",
            xytext=(-16, 20), ha="right", fontsize=12, fontweight="bold",
            color="#0d3b66",
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#0d3b66", alpha=0.9),
            arrowprops=dict(arrowstyle="->", color="#0d3b66", lw=1.3))


def draw_quad(ax, x, y, r, color="#10233b"):
    """Stylized top-down quadcopter marker."""
    arm = r * 1.05
    for dxr, dyr in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
        ax.plot([x, x + dxr * arm], [y, y + dyr * arm], color=color,
                lw=2.4, solid_capstyle="round", zorder=9)
    for dxr, dyr in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
        rot = Circle((x + dxr * arm, y + dyr * arm), r * 0.62,
                     facecolor="#2e6fb0", edgecolor=color, lw=1.6,
                     alpha=0.85, zorder=10)
        ax.add_patch(rot)
    ax.add_patch(Circle((x, y), r * 0.5, facecolor=color,
                        edgecolor="white", lw=1.4, zorder=11))


# --- Drone at current (leading) position ---
cxp, cyp = tx[nflow - 1], ty[nflow - 1]
halo = Circle((cxp, cyp), dx * 2.6, facecolor="white", alpha=0.28, zorder=7)
ax.add_patch(halo)
draw_quad(ax, cxp, cyp, r=dx * 1.15)
ax.annotate("UAV in transit", (cxp, cyp), textcoords="offset points",
            xytext=(16, -30), fontsize=12.5, fontweight="bold", color="#10233b",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#10233b", alpha=0.9),
            arrowprops=dict(arrowstyle="->", color="#10233b", lw=1.4))

# --- Cosmetics ---
ax.set_xlim(extent[0], extent[1])
ax.set_ylim(extent[2], extent[3])
ax.set_xlabel("Easting (m)", fontsize=13)
ax.set_ylabel("Northing (m)", fontsize=13)
fig.suptitle("Autonomous UAV Ingressing the IoT Plant-Health Monitoring Field",
             fontsize=17, fontweight="bold", y=0.985)
ax.set_title(
    f"Hybrid priority coverage planner   |   {M}\u00d7{N} grid @ {dx:.0f} m cells "
    f"({M*dx/1000:.1f}\u00d7{N*dx/1000:.1f} km)",
    fontsize=11.5, color="#555555", pad=12)
ax.grid(True, color="white", alpha=0.15, lw=0.6)
for s in ax.spines.values():
    s.set_edgecolor("#cccccc")

fig.tight_layout(rect=[0, 0, 1, 0.96])
out = "/workspace/drone_to_field_matlab.png"
fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
print("wrote", out)
