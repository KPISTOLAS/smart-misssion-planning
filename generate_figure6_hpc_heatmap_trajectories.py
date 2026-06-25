import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.ticker import FuncFormatter
from shapely.geometry import Point, Polygon


def create_synthetic_boundary(rng, center=(10.0, 8.0), mean_radius=6.0, n_vertices=24):
    """Create a realistic irregular polygon in local x/y coordinates (km)."""
    angles = np.linspace(0, 2 * np.pi, n_vertices, endpoint=False)
    angles += rng.normal(0.0, 0.06, size=n_vertices)
    angles = np.sort(angles)

    harmonic = (
        0.55 * np.sin(2 * angles + 0.7)
        + 0.40 * np.sin(3 * angles - 1.1)
        + 0.25 * np.sin(5 * angles + 0.3)
    )
    random_component = rng.normal(0.0, 0.35, size=n_vertices)
    radii = mean_radius + harmonic + random_component
    radii = np.clip(radii, mean_radius * 0.62, mean_radius * 1.42)

    x = center[0] + radii * np.cos(angles)
    y = center[1] + radii * np.sin(angles)
    poly = Polygon(np.column_stack([x, y])).buffer(0)
    if not poly.is_valid:
        raise ValueError("Failed to create a valid synthetic polygon boundary.")
    return poly


def sample_points_in_polygon(poly, n_points, rng):
    """Uniformly sample n_points inside polygon via rejection sampling."""
    minx, miny, maxx, maxy = poly.bounds
    points = []
    while len(points) < n_points:
        xs = rng.uniform(minx, maxx, size=200)
        ys = rng.uniform(miny, maxy, size=200)
        for x, y in zip(xs, ys):
            p = Point(x, y)
            if poly.contains(p):
                points.append((x, y))
                if len(points) == n_points:
                    break
    return np.array(points)


def build_grid(poly, nx=40, ny=40):
    """Build regular grid and inside-mask for cell centers."""
    minx, miny, maxx, maxy = poly.bounds
    pad_x = 0.03 * (maxx - minx)
    pad_y = 0.03 * (maxy - miny)
    minx, maxx = minx - pad_x, maxx + pad_x
    miny, maxy = miny - pad_y, maxy + pad_y

    x_edges = np.linspace(minx, maxx, nx + 1)
    y_edges = np.linspace(miny, maxy, ny + 1)

    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
    y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])
    xx, yy = np.meshgrid(x_centers, y_centers)

    inside = np.zeros_like(xx, dtype=bool)
    for i in range(ny):
        for j in range(nx):
            inside[i, j] = poly.contains(Point(xx[i, j], yy[i, j]))

    return x_edges, y_edges, xx, yy, inside


def gaussian_field(xx, yy, centers, amplitudes, sigmas):
    field = np.zeros_like(xx, dtype=float)
    for (cx, cy), amp, (sx, sy) in zip(centers, amplitudes, sigmas):
        exponent = -(((xx - cx) ** 2) / (2 * sx**2) + ((yy - cy) ** 2) / (2 * sy**2))
        field += amp * np.exp(exponent)
    return field


def normalize_inside(values, inside_mask):
    v = values.copy()
    valid = v[inside_mask]
    vmin = valid.min()
    vmax = valid.max()
    if vmax - vmin < 1e-12:
        v[inside_mask] = 0.0
    else:
        v[inside_mask] = (valid - vmin) / (vmax - vmin)
    v[~inside_mask] = np.nan
    return v


def cubic_bezier(p0, p1, p2, p3, n=120):
    t = np.linspace(0.0, 1.0, n)
    one_minus = 1.0 - t
    curve = (
        (one_minus**3)[:, None] * p0
        + (3 * one_minus**2 * t)[:, None] * p1
        + (3 * one_minus * t**2)[:, None] * p2
        + (t**3)[:, None] * p3
    )
    return curve


def make_trajectory(start, end, rng):
    """Generate a smooth synthetic UAV path from launch to target."""
    p0 = np.array(start, dtype=float)
    p3 = np.array(end, dtype=float)
    direction = p3 - p0
    distance = np.linalg.norm(direction)
    if distance < 1e-9:
        return np.vstack([p0, p3])
    direction /= distance
    normal = np.array([-direction[1], direction[0]])

    bend_scale = 0.14 * distance
    bend1 = rng.uniform(-1.0, 1.0) * bend_scale
    bend2 = rng.uniform(-1.0, 1.0) * bend_scale

    p1 = p0 + 0.32 * (p3 - p0) + bend1 * normal
    p2 = p0 + 0.72 * (p3 - p0) + bend2 * normal
    return cubic_bezier(p0, p1, p2, p3, n=150)


def make_trajectory_via_waypoint(start, waypoint, end, rng):
    """Create a smooth path that passes through a required waypoint."""
    first_leg = make_trajectory(start, waypoint, rng)
    second_leg = make_trajectory(waypoint, end, rng)
    return np.vstack([first_leg[:-1], second_leg])


def catmull_rom_chain(control_points, samples_per_seg=90):
    """Smoothly interpolate a path through all control points."""
    cps = np.asarray(control_points, dtype=float)
    if len(cps) < 2:
        return cps.copy()
    if len(cps) == 2:
        return np.vstack([cps[0], cps[1]])

    segments = []
    for i in range(len(cps) - 1):
        p0 = cps[max(i - 1, 0)]
        p1 = cps[i]
        p2 = cps[i + 1]
        p3 = cps[min(i + 2, len(cps) - 1)]

        t = np.linspace(0.0, 1.0, samples_per_seg, endpoint=(i == len(cps) - 2))
        t2 = t * t
        t3 = t2 * t

        # Standard Catmull-Rom basis (uniform parameterization).
        seg = 0.5 * (
            (2.0 * p1)
            + (-p0 + p2) * t[:, None]
            + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t2[:, None]
            + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t3[:, None]
        )
        segments.append(seg)

    return np.vstack(segments)


def select_spread_points(points, scores, n_select, min_dist):
    """
    Select high-score points with spatial separation to avoid over-clustering.
    Points are expected to be sorted by descending score.
    """
    selected = []
    selected_scores = []
    for idx, pt in enumerate(points):
        if len(selected) >= n_select:
            break
        if not selected:
            selected.append(pt)
            selected_scores.append(scores[idx])
            continue
        distances = np.linalg.norm(np.array(selected) - pt, axis=1)
        if np.all(distances >= min_dist):
            selected.append(pt)
            selected_scores.append(scores[idx])
    return np.array(selected), np.array(selected_scores)


def smooth_field(field: np.ndarray, inside_mask: np.ndarray, sigma: float = 1.1) -> np.ndarray:
    """Light Gaussian smoothing inside the boundary for a cleaner heatmap."""
    try:
        from scipy.ndimage import gaussian_filter
    except ImportError:
        return field

    work = np.nan_to_num(field, nan=0.0)
    work[~inside_mask] = 0.0
    smoothed = gaussian_filter(work, sigma=sigma)
    smoothed[~inside_mask] = np.nan
    return smoothed


def main():
    seed = 20260506
    rng = np.random.default_rng(seed)

    boundary = create_synthetic_boundary(rng)
    x_edges, y_edges, xx, yy, inside = build_grid(boundary, nx=72, ny=72)

    hotspot_centers = sample_points_in_polygon(boundary, n_points=5, rng=rng)
    hotspot_amps = rng.uniform(0.8, 1.4, size=5)
    hotspot_sigmas = np.column_stack(
        [rng.uniform(0.85, 1.55, size=5), rng.uniform(0.75, 1.45, size=5)]
    )
    base_field = gaussian_field(xx, yy, hotspot_centers, hotspot_amps, hotspot_sigmas)

    low_freq_trend = 0.15 * np.sin(0.45 * xx) * np.cos(0.38 * yy)
    noise = rng.normal(0.0, 0.05, size=xx.shape)
    hpc_raw = base_field + low_freq_trend + noise
    hpc_norm = normalize_inside(hpc_raw, inside)
    hpc_plot = smooth_field(hpc_norm, inside, sigma=1.0)

    inside_values = hpc_norm[inside]
    threshold = np.quantile(inside_values, 0.95)
    top_mask = inside & (hpc_norm >= threshold)
    top_points = np.column_stack([xx[top_mask], yy[top_mask]])
    top_scores = hpc_norm[top_mask]
    sorted_idx = np.argsort(top_scores)[::-1]
    top_points = top_points[sorted_idx]
    top_scores = top_scores[sorted_idx]

    n_uav = 4
    uav_colors = ["#1575A8", "#E7A400", "#0FA07A", "#8E3B96"]
    uav_linestyles = ["-", "-", "-", "--"]

    minx, miny, maxx, maxy = boundary.bounds
    launch_points = np.array(
        [
            [minx - 0.35, miny + 0.18 * (maxy - miny)],
            [minx - 0.20, miny + 0.44 * (maxy - miny)],
            [minx + 0.10, miny - 0.28],
            [minx + 0.30 * (maxx - minx), miny - 0.35],
        ]
    )
    cell_w = np.abs(x_edges[1] - x_edges[0])
    cell_h = np.abs(y_edges[1] - y_edges[0])
    spread_min_dist = 1.8 * np.sqrt(cell_w**2 + cell_h**2)

    display_points, _ = select_spread_points(
        top_points, top_scores, n_select=45, min_dist=spread_min_dist
    )
    targets, _ = select_spread_points(
        top_points, top_scores, n_select=n_uav, min_dist=4.0 * spread_min_dist
    )
    if len(targets) < n_uav:
        targets = top_points[:n_uav]

    bottom_top_priority_cell = top_points[np.argmin(top_points[:, 1])]
    topmost_top_priority_cell = top_points[np.argmax(top_points[:, 1])]
    passing_uav_idx = int(np.argmin(launch_points[:, 1]))

    trajectories = []
    for i in range(n_uav):
        if i == passing_uav_idx:
            control_pts = np.vstack(
                [
                    launch_points[i],
                    bottom_top_priority_cell,
                    topmost_top_priority_cell,
                    targets[i],
                ]
            )
            trajectories.append(catmull_rom_chain(control_pts, samples_per_seg=80))
        else:
            trajectories.append(make_trajectory(launch_points[i], targets[i], rng))

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 9,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.labelsize": 9,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "legend.fontsize": 7.5,
        }
    )

    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    extent = [x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]]
    heat = ax.imshow(
        hpc_plot,
        origin="lower",
        extent=extent,
        cmap="cividis",
        vmin=0.0,
        vmax=1.0,
        interpolation="bilinear",
        aspect="equal",
        zorder=1,
    )

    bx, by = boundary.exterior.xy
    boundary_patch = MplPolygon(
        np.column_stack([bx, by]),
        closed=True,
        fill=False,
        edgecolor="black",
        linewidth=1.05,
        zorder=4,
    )
    ax.add_patch(boundary_patch)
    heat.set_clip_path(boundary_patch)

    # Hide outside area by overlaying a white patch with a transparent hole effect via clipping.
    outside_cover = MplPolygon(
        np.column_stack([bx, by]),
        closed=True,
        facecolor="none",
        edgecolor="none",
    )
    ax.add_patch(outside_cover)

    ax.scatter(
        display_points[:, 0],
        display_points[:, 1],
        s=18,
        c="#1A1A1A",
        marker="*",
        linewidths=0.4,
        edgecolors="white",
        alpha=0.92,
        zorder=5,
        label="Top-priority cells",
    )

    ax.scatter(
        launch_points[:, 0],
        launch_points[:, 1],
        s=58,
        c=uav_colors,
        marker="^",
        edgecolors="white",
        linewidths=0.9,
        zorder=6,
        label="UAV launch points",
    )

    legend_lines = [
        Line2D([0], [0], color="black", lw=1.05, label="Synthetic boundary"),
        Line2D(
            [0],
            [0],
            marker="*",
            color="w",
            markerfacecolor="#1A1A1A",
            markeredgecolor="white",
            markeredgewidth=0.5,
            markersize=8,
            linestyle="None",
            label="Top-priority cells",
        ),
        Line2D(
            [0],
            [0],
            marker="^",
            color="w",
            markerfacecolor="#888888",
            markeredgecolor="white",
            markeredgewidth=0.6,
            markersize=7,
            linestyle="None",
            label="UAV launch points",
        ),
    ]

    for i, path in enumerate(trajectories):
        ax.plot(
            path[:, 0],
            path[:, 1],
            color=uav_colors[i],
            linestyle=uav_linestyles[i],
            linewidth=2.1,
            alpha=0.92,
            zorder=7 + i,
            path_effects=[pe.Stroke(linewidth=3.0, foreground="white"), pe.Normal()],
        )

        idx_a = int(0.78 * len(path))
        idx_b = int(0.93 * len(path))
        ax.annotate(
            "",
            xy=(path[idx_b, 0], path[idx_b, 1]),
            xytext=(path[idx_a, 0], path[idx_a, 1]),
            arrowprops=dict(
                arrowstyle="-|>",
                color=uav_colors[i],
                lw=1.8,
                mutation_scale=12,
                linestyle=uav_linestyles[i],
            ),
            zorder=12,
        )

        label_idx = max(8, int(0.12 * len(path)))
        ax.text(
            path[label_idx, 0],
            path[label_idx, 1],
            f"U{i + 1}",
            fontsize=7.5,
            fontweight="bold",
            color=uav_colors[i],
            ha="center",
            va="center",
            zorder=13,
            path_effects=[pe.Stroke(linewidth=2.0, foreground="white"), pe.Normal()],
        )

        legend_lines.append(
            Line2D(
                [0],
                [0],
                color=uav_colors[i],
                lw=2.1,
                linestyle=uav_linestyles[i],
                label=f"UAV {i + 1}",
            )
        )

    cbar = fig.colorbar(heat, ax=ax, fraction=0.042, pad=0.02)
    cbar.set_label("Normalized HPC priority", fontsize=9)
    cbar.set_ticks(np.linspace(0.0, 1.0, 6))
    cbar.ax.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{x:g}"))
    cbar.ax.tick_params(labelsize=8.5)

    ax.set_xlabel(r"Local $X$ (km)")
    ax.set_ylabel(r"Local $Y$ (km)")
    ax.tick_params(axis="both", which="both", labelsize=8.5)
    ax.grid(True, linestyle="-", linewidth=0.35, color="#E0E0E0", alpha=0.65, zorder=0)
    ax.set_aspect("equal", adjustable="box")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#333333")
    ax.spines["bottom"].set_color("#333333")

    fig.legend(
        handles=legend_lines,
        loc="upper center",
        bbox_to_anchor=(0.52, 0.99),
        ncol=3,
        frameon=False,
        handlelength=1.6,
        columnspacing=0.9,
        handletextpad=0.45,
    )

    fig.subplots_adjust(left=0.10, right=0.88, bottom=0.10, top=0.86)
    fig.savefig("Figure6_HPC_Heatmap_Trajectories.png", dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig("Figure6_HPC_Heatmap_Trajectories.pdf", dpi=300, bbox_inches="tight")
    fig.savefig("Figure6_HPC_Heatmap_Trajectories.eps", dpi=300, bbox_inches="tight")
    plt.close(fig)

    print("Saved Figure6_HPC_Heatmap_Trajectories.png (300 dpi)")
    print("Saved Figure6_HPC_Heatmap_Trajectories.pdf")
    print("Saved Figure6_HPC_Heatmap_Trajectories.eps")


if __name__ == "__main__":
    main()
