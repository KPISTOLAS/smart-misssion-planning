from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Polygon as MplPolygon, Rectangle
from shapely import affinity
from shapely.geometry import MultiPolygon, Point, Polygon, box


SEED = 20260511


def irregular_polygon(
    center_x: float,
    center_y: float,
    base_radius: float,
    n_vertices: int,
    rng: np.random.Generator,
) -> Polygon:
    angles = np.linspace(0.0, 2.0 * np.pi, n_vertices, endpoint=False)
    # Smooth perturbations make the synthetic boundary look map-like.
    radial = (
        1.0
        + 0.18 * np.sin(angles * 3.0 + 0.6)
        + 0.09 * np.cos(angles * 5.0 - 0.4)
        + rng.normal(0.0, 0.03, size=n_vertices)
    )
    radii = base_radius * radial
    x = center_x + radii * np.cos(angles)
    y = center_y + radii * np.sin(angles)
    poly = Polygon(np.column_stack([x, y])).buffer(0)
    if not poly.is_valid:
        poly = poly.buffer(0)
    return poly


def geometry_to_patches(geom: Polygon | MultiPolygon, **kwargs) -> list[MplPolygon]:
    parts: list[Polygon]
    if isinstance(geom, Polygon):
        parts = [geom]
    elif isinstance(geom, MultiPolygon):
        parts = list(geom.geoms)
    else:
        return []
    patches: list[MplPolygon] = []
    for part in parts:
        x, y = part.exterior.xy
        patches.append(MplPolygon(np.column_stack([x, y]), closed=True, **kwargs))
    return patches


def sample_points_in_polygon(
    poly: Polygon,
    count: int,
    rng: np.random.Generator,
) -> np.ndarray:
    minx, miny, maxx, maxy = poly.bounds
    points: list[tuple[float, float]] = []
    while len(points) < count:
        cand_x = rng.uniform(minx, maxx)
        cand_y = rng.uniform(miny, maxy)
        if poly.contains(Point(cand_x, cand_y)):
            points.append((cand_x, cand_y))
    return np.asarray(points)


def add_north_arrow(ax: plt.Axes, x: float, y: float, length: float) -> None:
    ax.annotate(
        "",
        xy=(x, y + length),
        xytext=(x, y),
        arrowprops=dict(arrowstyle="-|>", lw=1.2, color="black"),
        zorder=20,
    )
    ax.text(x, y + length + 1.2, "N", ha="center", va="bottom", fontsize=9, fontstyle="italic")


def add_scale_bar(ax: plt.Axes, x: float, y: float, length: float, label: str) -> None:
    ax.plot([x, x + length], [y, y], color="black", lw=2.0, zorder=20)
    ax.plot([x, x], [y - 0.7, y + 0.7], color="black", lw=1.2, zorder=20)
    ax.plot([x + length, x + length], [y - 0.7, y + 0.7], color="black", lw=1.2, zorder=20)
    ax.text(x + length / 2.0, y + 2.15, label, ha="center", va="bottom", fontsize=8)


def build_context_map(output_dir: Path) -> list[Path]:
    rng = np.random.default_rng(SEED)

    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.labelweight": "semibold",
            "axes.titlesize": 10,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "savefig.bbox": "tight",
        }
    )

    region = irregular_polygon(55.0, 45.0, 30.0, n_vertices=220, rng=rng)
    region = affinity.scale(region, xfact=1.35, yfact=0.92, origin="center")
    region = affinity.rotate(region, 7.5, origin="center")

    minx, miny, maxx, maxy = region.bounds
    width = maxx - minx
    x1 = minx + 0.34 * width
    x2 = minx + 0.67 * width

    zone_a = region.intersection(box(minx - 10.0, miny - 10.0, x1, maxy + 10.0))
    zone_b = region.intersection(box(x1, miny - 10.0, x2, maxy + 10.0))
    zone_c = region.intersection(box(x2, miny - 10.0, maxx + 10.0, maxy + 10.0))

    sector_centers = sample_points_in_polygon(region, count=6, rng=rng)
    sector_radius = 5.2
    sectors = [Point(cx, cy).buffer(sector_radius, resolution=64).intersection(region) for cx, cy in sector_centers]
    uav_points = sample_points_in_polygon(region, count=13, rng=rng)

    fig, ax = plt.subplots(figsize=(8.8, 6.5))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    zone_colors = {
        "A": "#009E73",  # bluish green
        "B": "#56B4E9",  # sky blue
        "C": "#E69F00",  # orange
    }

    for patch in geometry_to_patches(zone_a, facecolor=zone_colors["A"], edgecolor="white", lw=0.55, alpha=0.33):
        ax.add_patch(patch)
    for patch in geometry_to_patches(zone_b, facecolor=zone_colors["B"], edgecolor="white", lw=0.55, alpha=0.31):
        ax.add_patch(patch)
    for patch in geometry_to_patches(zone_c, facecolor=zone_colors["C"], edgecolor="white", lw=0.55, alpha=0.31):
        ax.add_patch(patch)

    for patch in geometry_to_patches(region, facecolor="none", edgecolor="#202020", lw=1.0):
        patch.set_zorder(8)
        ax.add_patch(patch)

    for i, sector in enumerate(sectors, start=1):
        for patch in geometry_to_patches(sector, facecolor="none", edgecolor="#6A3D9A", lw=1.25, alpha=0.98):
            patch.set_linestyle("--")
            patch.set_zorder(10)
            ax.add_patch(patch)
        label_x = float(sector_centers[i - 1, 0])
        label_y = float(sector_centers[i - 1, 1])
        if label_x > (maxx - 0.14 * (maxx - minx)):
            label_x -= 1.8
        if label_y > (maxy - 0.10 * (maxy - miny)):
            label_y -= 0.7
        ax.text(
            label_x,
            label_y,
            f"S{i}",
            ha="center",
            va="center",
            fontsize=8.2,
            color="#3B2055",
            bbox=dict(boxstyle="round,pad=0.16", fc="white", ec="none", alpha=0.92),
            clip_on=False,
            zorder=11,
        )

    ax.scatter(
        uav_points[:, 0],
        uav_points[:, 1],
        s=33,
        c="#D55E00",
        marker="^",
        edgecolors="white",
        linewidths=0.6,
        zorder=12,
    )

    x_margin, y_margin = 7.5, 6.0
    ax.set_xlim(minx - x_margin, maxx + x_margin)
    ax.set_ylim(miny - y_margin, maxy + y_margin)
    ax.set_aspect("equal")
    ax.set_xlabel(r"Easting ($\mathit{arb.\ units}$)")
    ax.set_ylabel(r"Northing ($\mathit{arb.\ units}$)")
    xticks = np.linspace(minx, maxx, 5)
    yticks = np.linspace(miny, maxy, 5)
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)
    ax.tick_params(axis="both", which="major", direction="out", length=2.4, width=0.65, pad=1.8)
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)

    add_north_arrow(ax, x=minx - 4.0, y=maxy - 12.0, length=8.5)
    add_scale_bar(ax, x=minx + 2.0, y=miny - 3.2, length=12.0, label="6 km")

    inset = ax.inset_axes([0.68, 0.64, 0.29, 0.31])
    inset.set_facecolor("#FAFAFA")
    outer_context = irregular_polygon(55.0, 44.0, 45.0, n_vertices=280, rng=rng)
    outer_context = affinity.scale(outer_context, xfact=1.35, yfact=0.95, origin="center")
    outer_context = affinity.rotate(outer_context, 5.0, origin="center")

    for patch in geometry_to_patches(outer_context, facecolor="#E2E2E2", edgecolor="#9A9A9A", lw=0.9, alpha=0.78):
        inset.add_patch(patch)

    for patch in geometry_to_patches(region, facecolor="none", edgecolor="#CC0000", lw=1.05):
        inset.add_patch(patch)
    inset.text(minx + 1.4, maxy + 1.9, "Study area", color="#CC0000", fontsize=7.2, fontweight="bold")
    outer_minx, outer_miny, outer_maxx, outer_maxy = outer_context.bounds
    inset.set_xlim(outer_minx - 4.0, outer_maxx + 4.0)
    inset.set_ylim(outer_miny - 4.0, outer_maxy + 4.0)
    inset.set_aspect("equal")
    inset.set_xticks([])
    inset.set_yticks([])
    inset.set_title("Synthetic regional context", fontsize=7, pad=1.5)
    for spine in inset.spines.values():
        spine.set_linewidth(0.8)

    legend_items = [
        Patch(facecolor=zone_colors["A"], edgecolor="white", alpha=0.33, label="Zone A: Dense vegetation"),
        Patch(facecolor=zone_colors["B"], edgecolor="white", alpha=0.31, label="Zone B: Mixed terrain"),
        Patch(facecolor=zone_colors["C"], edgecolor="white", alpha=0.31, label="Zone C: Sparse/open terrain"),
        Line2D([0], [0], color="#202020", lw=1.0, label="Study-area boundary"),
        Line2D([0], [0], color="#6A3D9A", lw=1.25, linestyle="--", label="Mission sectors (S1-S6)"),
        Line2D(
            [0],
            [0],
            marker="^",
            markersize=7,
            markerfacecolor="#D55E00",
            markeredgecolor="white",
            linestyle="None",
            label="Representative UAV/sensor points",
        ),
    ]
    ax.legend(
        handles=legend_items,
        loc="lower right",
        frameon=False,
        fontsize=8,
        borderpad=0.6,
    )

    png_path = output_dir / "Figure11_StudyArea_ContextMap.png"
    pdf_path = output_dir / "Figure11_StudyArea_ContextMap.pdf"
    fig.savefig(png_path, dpi=300, facecolor="white")
    fig.savefig(pdf_path, facecolor="white")
    plt.close(fig)
    return [png_path, pdf_path]


def main() -> None:
    root = Path(__file__).resolve().parent
    outputs = build_context_map(root)
    for path in outputs:
        print(f"Saved: {path}")


if __name__ == "__main__":
    main()
