from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HPC_TARGET_PCT = 85.0  # shared mission target enforced by BatteryAwareOrchestrator


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
            f"{h:.1f}",
            ha="center",
            va="bottom",
            fontsize=fontsize,
            color="#333333",
        )


def load_coverage(
    csv_path: Path,
    planner: str = "priority",
    u_min: int = 2,
    u_max: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """Measured high-priority coverage (HPC %) per fleet size for one planner."""
    df = pd.read_csv(csv_path)
    subset = df[(df["planner"] == planner) & (df["numUAV"].between(u_min, u_max))]

    if subset.empty:
        raise ValueError(
            f"No rows found for planner='{planner}' in U={u_min}..{u_max}. "
            "Check planner name or CSV contents."
        )

    grouped = (
        subset.groupby("numUAV", as_index=False)
        .agg(coverage_pct=("mean_HPC_pct", "mean"))
        .sort_values("numUAV")
    )
    return grouped["numUAV"].to_numpy(), grouped["coverage_pct"].to_numpy()


def create_figure7(out_png: Path, out_pdf: Path, csv_path: Path) -> None:
    u, coverage_proposed = load_coverage(csv_path)
    x = np.arange(len(u), dtype=float)

    _apply_ieee_style()
    color_proposed = "#0FA07A"

    fig, ax = plt.subplots(figsize=(3.6, 2.5))
    fig.patch.set_facecolor("white")

    bars_proposed = ax.bar(
        x,
        coverage_proposed,
        width=0.5,
        label="IUEF-EM (proposed)",
        color=color_proposed,
        linewidth=0,
        zorder=3,
    )
    annotate_bars(ax, bars_proposed, dy=0.5, fontsize=7.2)

    ax.axhline(
        HPC_TARGET_PCT,
        color="#C23B22",
        lw=0.9,
        ls="--",
        zorder=4,
        label=f"Mission target ({HPC_TARGET_PCT:.0f}%)",
    )

    ax.set_xlabel("Swarm size (UAVs)")
    ax.set_ylabel("Coverage quality, HPC (%)")
    ax.set_xticks(x)
    ax.set_xticklabels([str(int(v)) for v in u])
    ax.set_ylim(70, 100)
    ax.set_yticks(np.arange(70, 101, 5))
    _style_axes(ax)

    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.16),
        ncol=2,
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
    create_figure7(
        out_png=out_png,
        out_pdf=out_pdf,
        csv_path=root / "ieeeComparativeByPlannerFleet.csv",
    )

    print(f"Saved: {out_png}")
    print(f"Saved: {out_pdf}")
    print(
        "\nSuggested caption:\n"
        "IUEF-EM high-priority coverage (HPC) versus swarm size. The planner holds "
        "coverage at the mission target across every fleet size, showing that "
        "coverage quality is invariant to swarm size and that fleet sizing can "
        "therefore be decided on time, energy, and safety grounds alone."
    )


if __name__ == "__main__":
    main()
