import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.ticker import FuncFormatter, MultipleLocator
from shapely.geometry import Point, Polygon


def create_synthetic_boundary(rng, center, radius_x, radius_y, n_vertices=24):
    """Create a realistic irregular polygon in local x/y coordinates (km).

    The radial perturbations are expressed as unit fractions so the boundary
    shape stays self-similar at any physical scale, and independent
    ``radius_x``/``radius_y`` let the field be elongated to match the aspect of
    the study area (so it fills the frame instead of floating in whitespace).
    """
    angles = np.linspace(0, 2 * np.pi, n_vertices, endpoint=False)
    angles += rng.normal(0.0, 0.06, size=n_vertices)
    angles = np.sort(angles)

    harmonic = (
        0.092 * np.sin(2 * angles + 0.7)
        + 0.067 * np.sin(3 * angles - 1.1)
        + 0.042 * np.sin(5 * angles + 0.3)
    )
    random_component = rng.normal(0.0, 0.058, size=n_vertices)
    unit_radii = np.clip(1.0 + harmonic + random_component, 0.62, 1.42)

    x = center[0] + radius_x * unit_radii * np.cos(angles)
    y = center[1] + radius_y * unit_radii * np.sin(angles)
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


def build_grid(poly, domain_bounds, nx, ny):
    """Build a regular grid over an explicit domain and inside-mask for cell centers.

    ``domain_bounds`` is ``(minx, miny, maxx, maxy)`` in km and defines the true
    study-area extent, so the axes reflect the physical size of the mapped region
    rather than the padded bounding box of the synthetic boundary. Using ``nx``/``ny``
    equal to the real cell counts keeps every cell at the intended ground resolution.
    """
    minx, miny, maxx, maxy = domain_bounds

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

    # Real study-area geometry (see runIEEEStudy.m / MapGenerator.build): a
    # 54 x 72 grid of square cells at dx = 18 m. This fixes the axes to the true
    # 0.972 km x 1.296 km footprint instead of the previous ~16 km x 14 km, which
    # came from generating the synthetic boundary at an arbitrary km scale.
    dx_m = 18.0
    n_cells_x = 54
    n_cells_y = 72
    domain_w = n_cells_x * dx_m / 1000.0  # 0.972 km
    domain_h = n_cells_y * dx_m / 1000.0  # 1.296 km
    domain_bounds = (0.0, 0.0, domain_w, domain_h)

    # Elongate the field to match the portrait study area so it fills the frame,
    # while keeping a margin so launch markers placed just outside the boundary
    # still fall within the plotted domain.
    boundary_center = (domain_w / 2.0, domain_h / 2.0)
    boundary_radius_x = 0.31 * domain_w
    boundary_radius_y = 0.315 * domain_h

    boundary = create_synthetic_boundary(
        rng, boundary_center, boundary_radius_x, boundary_radius_y
    )
    x_edges, y_edges, xx, yy, inside = build_grid(
        boundary, domain_bounds, nx=n_cells_x, ny=n_cells_y
    )

    hotspot_centers = sample_points_in_polygon(boundary, n_points=5, rng=rng)
    hotspot_amps = rng.uniform(0.8, 1.4, size=5)
    # Hotspot widths scale with the boundary size (fractions of its radius) so the
    # heatmap keeps the same visual character at the corrected physical scale.
    hotspot_sigmas = np.column_stack(
        [
            rng.uniform(0.142, 0.258, size=5) * boundary_radius_x,
            rng.uniform(0.125, 0.242, size=5) * boundary_radius_y,
        ]
    )
    base_field = gaussian_field(xx, yy, hotspot_centers, hotspot_amps, hotspot_sigmas)

    # Low-frequency trend expressed in cycles across the domain so it renders the
    # same gentle gradient regardless of the absolute domain size.
    low_freq_trend = (
        0.15
        * np.sin(2 * np.pi * 0.85 * (xx - domain_bounds[0]) / domain_w)
        * np.cos(2 * np.pi * 0.73 * (yy - domain_bounds[1]) / domain_h)
    )
    noise = rng.normal(0.0, 0.05, size=xx.shape)
    hpc_raw = base_field + low_freq_trend + noise
    hpc_norm = normalize_inside(hpc_raw, inside)
    hpc_plot = smooth_field(hpc_norm, inside, sigma=1.0)

    inside_values = hpc_norm[inside]
    threshold = np.quantile(inside_values, 0.90)
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
    b_w = maxx - minx
    b_h = maxy - miny
    # Offsets are fractions of the boundary bounding box so the launch markers sit
    # just outside the boundary while remaining inside the plotted study area.
    launch_points = np.array(
        [
            [minx - 0.029 * b_w, miny + 0.18 * b_h],
            [minx - 0.017 * b_w, miny + 0.44 * b_h],
            [minx + 0.008 * b_w, miny - 0.023 * b_h],
            [minx + 0.30 * b_w, miny - 0.029 * b_h],
        ]
    )
    cell_w = np.abs(x_edges[1] - x_edges[0])
    cell_h = np.abs(y_edges[1] - y_edges[0])
    spread_min_dist = 1.8 * np.sqrt(cell_w**2 + cell_h**2)

    display_points, _ = select_spread_points(
        top_points, top_scores, n_select=30, min_dist=1.9 * spread_min_dist
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
            "mathtext.fontset": "stix",
            "font.size": 9,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.labelsize": 10,
            "axes.linewidth": 0.7,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "legend.fontsize": 8,
        }
    )

    fig, ax = plt.subplots(figsize=(4.9, 5.3))
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

    # Study-area boundary with a soft white casing underneath for depth.
    bx, by = boundary.exterior.xy
    boundary_xy = np.column_stack([bx, by])
    ax.plot(
        bx, by,
        color="white",
        linewidth=2.6,
        alpha=0.85,
        solid_joinstyle="round",
        solid_capstyle="round",
        zorder=3,
    )
    boundary_patch = MplPolygon(
        boundary_xy,
        closed=True,
        fill=False,
        edgecolor="#1A1A1A",
        linewidth=1.3,
        joinstyle="round",
        zorder=4,
    )
    ax.add_patch(boundary_patch)
    heat.set_clip_path(boundary_patch)

    # High-priority cells: white stars with a dark rim + halo so they read clearly
    # over both the dark background and the bright hotspots.
    ax.scatter(
        display_points[:, 0],
        display_points[:, 1],
        s=22,
        c="white",
        marker="*",
        linewidths=0.55,
        edgecolors="#1A1A1A",
        alpha=0.98,
        zorder=5,
        path_effects=[pe.withStroke(linewidth=1.3, foreground="#404040")],
    )

    # UAV routes: colored strokes over a white halo for contrast on the heatmap.
    for i, path in enumerate(trajectories):
        ax.plot(
            path[:, 0],
            path[:, 1],
            color=uav_colors[i],
            linestyle=uav_linestyles[i],
            linewidth=1.9,
            alpha=0.97,
            solid_capstyle="round",
            zorder=7 + i,
            path_effects=[pe.Stroke(linewidth=3.1, foreground="white"), pe.Normal()],
        )

        idx_a = int(0.86 * len(path))
        idx_b = min(len(path) - 1, int(0.965 * len(path)))
        ax.annotate(
            "",
            xy=(path[idx_b, 0], path[idx_b, 1]),
            xytext=(path[idx_a, 0], path[idx_a, 1]),
            arrowprops=dict(
                arrowstyle="-|>",
                color=uav_colors[i],
                lw=1.7,
                mutation_scale=11,
                linestyle=uav_linestyles[i],
                shrinkA=0,
                shrinkB=0,
            ),
            zorder=12,
        )

    # Launch (start) and target (goal) markers, colored per UAV.
    ax.scatter(
        launch_points[:, 0],
        launch_points[:, 1],
        s=62,
        c=uav_colors,
        marker="^",
        edgecolors="white",
        linewidths=1.0,
        zorder=13,
    )
    ax.scatter(
        targets[:, 0],
        targets[:, 1],
        s=54,
        c=uav_colors,
        marker="o",
        edgecolors="white",
        linewidths=1.0,
        zorder=13,
    )

    cbar = fig.colorbar(heat, ax=ax, fraction=0.046, pad=0.03)
    cbar.set_label(r"Normalized priority $W$", fontsize=9.5, labelpad=6)
    cbar.set_ticks(np.linspace(0.0, 1.0, 5))
    cbar.ax.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{x:g}"))
    cbar.ax.tick_params(labelsize=8, width=0.7, length=3)
    cbar.outline.set_linewidth(0.7)

    ax.set_xlim(x_edges[0], x_edges[-1])
    ax.set_ylim(y_edges[0], y_edges[-1])
    ax.set_xlabel(r"Local $X$ (km)")
    ax.set_ylabel(r"Local $Y$ (km)")
    ax.xaxis.set_major_locator(MultipleLocator(0.25))
    ax.yaxis.set_major_locator(MultipleLocator(0.25))
    ax.tick_params(axis="both", which="both", labelsize=8.5, width=0.7, length=3, color="#5A5A5A")
    ax.grid(True, linestyle=(0, (3, 3)), linewidth=0.3, color="#BEBEBE", alpha=0.55, zorder=0)
    ax.set_aspect("equal", adjustable="box")

    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_color("#444444")
        ax.spines[side].set_linewidth(0.7)

    route_handles = [
        Line2D([0], [0], color=uav_colors[i], lw=1.9, linestyle=uav_linestyles[i], label=f"UAV {i + 1}")
        for i in range(n_uav)
    ]
    marker_handles = [
        Line2D([0], [0], color="#1A1A1A", lw=1.3, label="Boundary"),
        Line2D(
            [0], [0], marker="*", linestyle="None", markersize=7,
            markerfacecolor="#F4F4F4", markeredgecolor="#242424", markeredgewidth=0.5,
            label="High-priority cell",
        ),
        Line2D(
            [0], [0], marker="^", linestyle="None", markersize=7,
            markerfacecolor="#7A7A7A", markeredgecolor="white", markeredgewidth=0.8,
            label="Launch",
        ),
        Line2D(
            [0], [0], marker="o", linestyle="None", markersize=6.5,
            markerfacecolor="#7A7A7A", markeredgecolor="white", markeredgewidth=0.8,
            label="Target",
        ),
    ]

    fig.legend(
        handles=route_handles + marker_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.01),
        ncol=4,
        frameon=False,
        handlelength=1.7,
        columnspacing=1.1,
        handletextpad=0.5,
        labelspacing=0.55,
    )

    fig.subplots_adjust(left=0.13, right=0.99, bottom=0.17, top=0.99)
    fig.savefig("Figure6_HPC_Heatmap_Trajectories.png", dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig("Figure6_HPC_Heatmap_Trajectories.pdf", bbox_inches="tight")
    fig.savefig("Figure6_HPC_Heatmap_Trajectories.eps", bbox_inches="tight")
    plt.close(fig)

    print("Saved Figure6_HPC_Heatmap_Trajectories.png (600 dpi)")
    print("Saved Figure6_HPC_Heatmap_Trajectories.pdf")
    print("Saved Figure6_HPC_Heatmap_Trajectories.eps")


if __name__ == "__main__":
    main()
