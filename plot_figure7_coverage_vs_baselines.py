from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np


def _apply_ieee_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 9,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "figure.dpi": 120,
            "savefig.bbox": "tight",
            "savefig.facecolor": "white",
            "axes.linewidth": 0.6,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
        }
    )


def _style_axes(ax: plt.Axes) -> None:
    ax.set_facecolor("white")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for key in ("left", "bottom"):
        ax.spines[key].set_color("#333333")
        ax.spines[key].set_linewidth(0.6)
    ax.tick_params(axis="both", length=3, width=0.6, colors="#333333")
    ax.grid(axis="y", color="#E8E8E8", linestyle="-", linewidth=0.55, zorder=0)
    ax.grid(axis="x", visible=False)
    ax.set_axisbelow(True)


def annotate_bars(ax: plt.Axes, bars, *, dy: float = 0.6, fontsize: float = 7.0) -> None:
    for bar in bars:
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            h + dy,
            f"{h:.0f}",
            ha="center",
            va="bottom",
            fontsize=fontsize,
            color="#333333",
        )


def add_significance_markers(
    ax: plt.Axes,
    x: np.ndarray,
    proposed_vals: np.ndarray,
    baseline_a_vals: np.ndarray,
    baseline_b_vals: np.ndarray,
    pvals_vs_a: Optional[np.ndarray] = None,
    pvals_vs_b: Optional[np.ndarray] = None,
) -> None:
    if pvals_vs_a is None or pvals_vs_b is None:
        return

    for i in range(len(x)):
        if pvals_vs_a[i] < 0.05 and pvals_vs_b[i] < 0.05:
            y_ref = max(proposed_vals[i], baseline_a_vals[i], baseline_b_vals[i])
            ax.text(x[i], y_ref + 1.5, "*", ha="center", va="bottom", fontsize=10, fontweight="bold")


def create_figure7(
    out_png: Path,
    out_pdf: Path,
) -> None:
    u_labels = ["2", "3", "4", "5"]
    x = np.arange(len(u_labels), dtype=float)

    coverage_proposed = np.array([88, 93, 95, 96], dtype=float)
    coverage_greedy = np.array([79, 84, 86, 87], dtype=float)
    coverage_voronoi = np.array([82, 87, 89, 90], dtype=float)

    std_proposed = None
    std_greedy = None
    std_voronoi = None
    pvals_vs_greedy = None
    pvals_vs_voronoi = None

    _apply_ieee_style()

    color_proposed = "#0FA07A"
    color_greedy = "#E7A400"
    color_voronoi = "#1575A8"

    fig, ax = plt.subplots(figsize=(3.6, 2.5))
    fig.patch.set_facecolor("white")

    width = 0.23
    pos_proposed = x - width
    pos_greedy = x
    pos_voronoi = x + width

    bar_kw = dict(linewidth=0, zorder=3)
    bars_proposed = ax.bar(
        pos_proposed,
        coverage_proposed,
        width=width,
        label="IUEF-EM (proposed)",
        color=color_proposed,
        **bar_kw,
    )
    bars_greedy = ax.bar(
        pos_greedy,
        coverage_greedy,
        width=width,
        label="Greedy",
        color=color_greedy,
        alpha=0.92,
        **bar_kw,
    )
    bars_voronoi = ax.bar(
        pos_voronoi,
        coverage_voronoi,
        width=width,
        label="Voronoi–greedy",
        color=color_voronoi,
        alpha=0.92,
        **bar_kw,
    )

    if std_proposed is not None:
        ax.errorbar(
            pos_proposed,
            coverage_proposed,
            yerr=std_proposed,
            fmt="none",
            ecolor="#333333",
            capsize=2,
            lw=0.6,
            zorder=4,
        )
    if std_greedy is not None:
        ax.errorbar(pos_greedy, coverage_greedy, yerr=std_greedy, fmt="none", ecolor="#333333", capsize=2, lw=0.6, zorder=4)
    if std_voronoi is not None:
        ax.errorbar(pos_voronoi, coverage_voronoi, yerr=std_voronoi, fmt="none", ecolor="#333333", capsize=2, lw=0.6, zorder=4)

    annotate_bars(ax, bars_proposed, dy=0.5, fontsize=7.2)
    annotate_bars(ax, bars_greedy, dy=0.5, fontsize=6.8)
    annotate_bars(ax, bars_voronoi, dy=0.5, fontsize=6.8)

    ax.set_xlabel("Swarm size (UAVs)")
    ax.set_ylabel("Coverage quality (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(u_labels)
    ax.set_ylim(70, 100)
    ax.set_yticks(np.arange(70, 101, 5))
    _style_axes(ax)

    add_significance_markers(
        ax=ax,
        x=pos_proposed,
        proposed_vals=coverage_proposed,
        baseline_a_vals=coverage_greedy,
        baseline_b_vals=coverage_voronoi,
        pvals_vs_a=pvals_vs_greedy,
        pvals_vs_b=pvals_vs_voronoi,
    )

    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.14),
        ncol=3,
        frameon=False,
        handlelength=1.2,
        columnspacing=0.9,
        handletextpad=0.4,
    )

    fig.subplots_adjust(left=0.16, right=0.98, bottom=0.20, top=0.82)
    fig.savefig(out_png, dpi=300, facecolor="white")
    fig.savefig(out_pdf, facecolor="white")
    plt.close(fig)


def main() -> None:
    root = Path(__file__).resolve().parent
    out_png = root / "Figure7_CoverageQuality_vs_Baselines.png"
    out_pdf = root / "Figure7_CoverageQuality_vs_Baselines.pdf"
    create_figure7(out_png=out_png, out_pdf=out_pdf)

    print(f"Saved: {out_png}")
    print(f"Saved: {out_pdf}")


if __name__ == "__main__":
    main()
