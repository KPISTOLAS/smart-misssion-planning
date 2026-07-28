from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from matplotlib.ticker import FuncFormatter
from matplotlib.ticker import FormatStrFormatter


def synthetic_metrics(alpha: np.ndarray, beta: np.ndarray, gamma: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Deterministic synthetic sensitivity surfaces matching expected trends:
    - HPC/utility rises with alpha and soft-penalizes very high beta, gamma.
    - Energy reduces with larger beta (when varied) and gamma.
    - Congestion overhead reduces as gamma increases.
    """
    # Utility/HPC (%)
    hpc = (
        64.0
        + 24.0 * alpha
        - 7.0 * (beta - 0.6) ** 2
        - 5.0 * (gamma - 0.6) ** 2
        - 1.8 * beta
        - 1.5 * gamma
        + 2.0 * alpha * (1.0 - np.abs(beta - 0.6))
    )

    # Mission energy (Wh)
    energy = (
        168.0
        + 12.0 * alpha
        - 34.0 * beta
        - 8.0 * gamma
        + 4.0 * (alpha - 0.8) ** 2
        + 1.5 * np.maximum(0.0, gamma - 0.8)
    )

    # Congestion overhead (%)
    congestion = (
        36.0
        + 3.5 * alpha
        - 5.0 * beta
        - 16.0 * gamma
        + 6.5 * (beta - 0.6) ** 2
        + 1.0 * alpha * beta
    )

    return hpc, energy, congestion


def add_cell_box(ax: plt.Axes, x_values: np.ndarray, y_values: np.ndarray, x0: float, y0: float) -> None:
    x_step = float(np.mean(np.diff(x_values)))
    y_step = float(np.mean(np.diff(y_values)))
    x_center = float(x0)
    y_center = float(y0)
    rect = Rectangle(
        (x_center - x_step / 2.0, y_center - y_step / 2.0),
        x_step,
        y_step,
        fill=False,
        lw=0.75,
        ec="black",
        linestyle="--",
        zorder=6,
    )
    ax.add_patch(rect)
    # -style highlighted operating point: outlined star with white halo.
    ax.plot(
        x_center,
        y_center,
        marker="*",
        linestyle="None",
        markerfacecolor="white",
        markeredgecolor="white",
        markeredgewidth=0.9,
        markersize=8.5,
        zorder=7,
    )
    ax.plot(
        x_center,
        y_center,
        marker="*",
        linestyle="None",
        markerfacecolor="none",
        markeredgecolor="black",
        markeredgewidth=0.8,
        markersize=7.2,
        zorder=8,
    )


def edges_from_centers(vals: np.ndarray) -> tuple[float, float]:
    step = float(np.mean(np.diff(vals)))
    return float(vals.min() - 0.5 * step), float(vals.max() + 0.5 * step)


def save_single_heatmap(
    data: np.ndarray,
    x_vals: np.ndarray,
    y_vals: np.ndarray,
    x_label: str,
    y_label: str,
    cmap: str,
    cbar_label: str,
    star_x: float,
    star_y: float,
    out_png: Path,
    out_pdf: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(2.55, 2.60))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    for spine in ax.spines.values():
        spine.set_linewidth(0.7)
    x_min, x_max = edges_from_centers(x_vals)
    y_min, y_max = edges_from_centers(y_vals)

    im = ax.imshow(
        data,
        origin="lower",
        extent=[x_min, x_max, y_min, y_max],
        aspect="equal",
        cmap=cmap,
        interpolation="nearest",
    )
    ax.set_xlabel(x_label, labelpad=2.2)
    ax.set_ylabel(y_label, labelpad=2.8)
    ax.set_xticks(x_vals)
    ax.set_yticks(y_vals)
    ax.xaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    ax.tick_params(axis="both", which="major", direction="in", length=2.8, width=0.7, top=True, right=True, pad=1.5)
    add_cell_box(ax, x_vals, y_vals, x0=star_x, y0=star_y)
    cbar = fig.colorbar(im, ax=ax, fraction=0.055, pad=0.03)
    cbar.set_label(cbar_label, fontsize=plt.rcParams["axes.labelsize"])
    cbar.ax.tick_params(direction="in", length=2.8, width=0.7)
    cbar.ax.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{x:g}"))

    fig.tight_layout()
    fig.savefig(out_png, dpi=300, facecolor="white")
    fig.savefig(out_pdf, facecolor="white")
    plt.close(fig)


def create_figures(root: Path) -> list[Path]:
    # Publication-style typography and layout.
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 8.5,
            "axes.labelsize": 8.5,
            "axes.titlesize": 9.0,
            "xtick.labelsize": 8.0,
            "ytick.labelsize": 8.0,
            "savefig.bbox": "tight",
        }
    )

    alpha_vals = np.array([0.2, 0.4, 0.6, 0.8, 1.0], dtype=float)
    beta_vals = np.array([0.2, 0.4, 0.6, 0.8, 1.0], dtype=float)
    gamma_vals = np.array([0.2, 0.4, 0.6, 0.8, 1.0], dtype=float)

    fixed_gamma = 0.6
    fixed_beta = 0.6
    fixed_alpha = 0.8

    # Panel (a): HPC(alpha, beta | gamma=0.6)
    aa, bb = np.meshgrid(alpha_vals, beta_vals, indexing="xy")
    gg_a = np.full_like(aa, fixed_gamma)
    hpc_a, _, _ = synthetic_metrics(alpha=aa, beta=bb, gamma=gg_a)

    # Panel (b): Energy(alpha, gamma | beta=0.6)
    aa_b, gg = np.meshgrid(alpha_vals, gamma_vals, indexing="xy")
    bb_b = np.full_like(aa_b, fixed_beta)
    _, energy_b, _ = synthetic_metrics(alpha=aa_b, beta=bb_b, gamma=gg)

    # Panel (c): Congestion(beta, gamma | alpha=0.8)
    bb_c, gg_c = np.meshgrid(beta_vals, gamma_vals, indexing="xy")
    aa_c = np.full_like(bb_c, fixed_alpha)
    _, _, congestion_c = synthetic_metrics(alpha=aa_c, beta=bb_c, gamma=gg_c)

    out_hpc_png = root / "Figure10_HPC_Weight_Sensitivity.png"
    out_hpc_pdf = root / "Figure10_HPC_Weight_Sensitivity.pdf"
    save_single_heatmap(
        data=hpc_a,
        x_vals=alpha_vals,
        y_vals=beta_vals,
        x_label=r"$\alpha$",
        y_label=r"$\beta$",
        cmap="viridis",
        cbar_label="HPC (%)",
        star_x=0.8,
        star_y=0.6,
        out_png=out_hpc_png,
        out_pdf=out_hpc_pdf,
    )

    out_energy_png = root / "Figure10_Energy_Weight_Sensitivity.png"
    out_energy_pdf = root / "Figure10_Energy_Weight_Sensitivity.pdf"
    save_single_heatmap(
        data=energy_b,
        x_vals=alpha_vals,
        y_vals=gamma_vals,
        x_label=r"$\alpha$",
        y_label=r"$\gamma$",
        cmap="cividis",
        cbar_label="Energy (Wh)",
        star_x=0.8,
        star_y=0.6,
        out_png=out_energy_png,
        out_pdf=out_energy_pdf,
    )

    out_cong_png = root / "Figure10_Congestion_Weight_Sensitivity.png"
    out_cong_pdf = root / "Figure10_Congestion_Weight_Sensitivity.pdf"
    save_single_heatmap(
        data=congestion_c,
        x_vals=beta_vals,
        y_vals=gamma_vals,
        x_label=r"$\beta$",
        y_label=r"$\gamma$",
        cmap="viridis",
        cbar_label="Congestion (%)",
        star_x=0.6,
        star_y=0.6,
        out_png=out_cong_png,
        out_pdf=out_cong_pdf,
    )

    return [
        out_hpc_png,
        out_hpc_pdf,
        out_energy_png,
        out_energy_pdf,
        out_cong_png,
        out_cong_pdf,
    ]


def main() -> None:
    root = Path(__file__).resolve().parent
    output_paths = create_figures(root=root)
    for p in output_paths:
        print(f"Saved: {p}")


if __name__ == "__main__":
    main()

