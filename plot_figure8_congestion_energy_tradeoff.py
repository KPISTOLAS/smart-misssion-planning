from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _apply_publication_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 9,
            "axes.labelsize": 9,
            "legend.fontsize": 7.5,
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


def load_tradeoff(
    csv_path: Path,
    planner: str = "priority",
    u_min: int = 1,
    u_max: int = 8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Measured fleet energy (MJ) and conflict rate (%) per fleet size."""
    df = pd.read_csv(csv_path)
    subset = df[(df["planner"] == planner) & (df["numUAV"].between(u_min, u_max))]

    if subset.empty:
        raise ValueError(
            f"No rows found for planner='{planner}' in U={u_min}..{u_max}. "
            "Check planner name or CSV contents."
        )

    grouped = (
        subset.groupby("numUAV", as_index=False)
        .agg(
            energy_MJ=("mean_fleet_energy_MJ", "mean"),
            conflict=("mean_any_collision", "mean"),
        )
        .sort_values("numUAV")
    )

    return (
        grouped["numUAV"].to_numpy(),
        grouped["energy_MJ"].to_numpy(),
        grouped["conflict"].to_numpy() * 100.0,
    )


def create_figure(out_png: Path, out_pdf: Path, csv_path: Path) -> None:
    u, energy_mj, conflict_pct = load_tradeoff(csv_path)

    color_energy = "#1575A8"
    color_conflict = "#E7A400"

    _apply_publication_style()

    fig, ax_left = plt.subplots(figsize=(3.6, 2.55))
    fig.patch.set_facecolor("white")
    ax_left.set_facecolor("white")
    ax_right = ax_left.twinx()

    # Conflict-free operating region: largest fleet sizes with zero collisions.
    safe = u[conflict_pct <= 0]
    u_safe_max = int(safe.max()) if safe.size else int(u.min())
    ax_left.axvspan(u.min() - 0.2, u_safe_max + 0.3, color="#D9D9D9", alpha=0.45,
                    lw=0, zorder=0)
    ax_left.text(
        (u.min() + u_safe_max) / 2,
        0.97,
        f"Conflict-free\nU \u2264 {u_safe_max}",
        transform=ax_left.get_xaxis_transform(),
        ha="center",
        va="top",
        fontsize=7,
        color="#444444",
        linespacing=1.05,
        zorder=1,
    )

    line_kw = dict(lw=1.35, zorder=4)
    energy_line = ax_left.plot(
        u, energy_mj, "-o", color=color_energy, ms=4.0,
        label="Fleet energy per mission", **line_kw,
    )[0]
    conflict_line = ax_right.plot(
        u, conflict_pct, "-s", color=color_conflict, ms=3.8,
        label="Conflict rate", **line_kw,
    )[0]

    ax_left.set_xlabel("Swarm size (UAVs)")
    ax_left.set_ylabel("Fleet energy per mission (MJ)", color=color_energy)
    ax_right.set_ylabel("Conflict rate (%)", color="#333333")

    ax_left.set_xlim(u.min() - 0.2, u.max() + 0.2)
    ax_left.set_xticks(u)
    e_lo = np.floor(energy_mj.min() - 0.5)
    e_hi = np.ceil(energy_mj.max() + 0.5)
    ax_left.set_ylim(e_lo, e_hi)
    ax_right.set_ylim(0, 105)
    ax_right.set_yticks(np.arange(0, 101, 20))

    ax_left.grid(axis="y", color="#E8E8E8", linestyle="-", linewidth=0.55, zorder=0)
    ax_left.set_axisbelow(True)

    ax_left.spines["top"].set_visible(False)
    ax_left.spines["right"].set_visible(False)
    ax_left.spines["left"].set_color(color_energy)
    ax_left.spines["bottom"].set_color("#333333")
    ax_left.spines["left"].set_linewidth(0.6)
    ax_left.spines["bottom"].set_linewidth(0.6)

    ax_right.spines["top"].set_visible(False)
    ax_right.spines["left"].set_visible(False)
    ax_right.spines["right"].set_color("#333333")
    ax_right.spines["bottom"].set_visible(False)
    ax_right.spines["right"].set_linewidth(0.6)

    ax_left.tick_params(axis="x", length=3, width=0.6, colors="#333333")
    ax_left.tick_params(axis="y", length=3, width=0.6, colors=color_energy,
                        labelcolor=color_energy)
    ax_right.tick_params(axis="y", length=3, width=0.6, colors="#333333",
                         labelcolor="#333333")

    # The shaded region is labelled in-plot, so it is left out of the legend.
    handles = [energy_line, conflict_line]
    labels = [h.get_label() for h in handles]

    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.10),
        ncol=2,
        frameon=False,
        handlelength=1.6,
        columnspacing=0.9,
        handletextpad=0.45,
    )

    fig.subplots_adjust(left=0.17, right=0.84, bottom=0.20, top=0.86)
    fig.savefig(out_png, dpi=300, facecolor="white")
    fig.savefig(out_pdf, facecolor="white")
    plt.close(fig)


def main() -> None:
    root = Path(__file__).resolve().parent
    out_png = root / "Figure8_CongestionEnergy_Tradeoff.png"
    out_pdf = root / "Figure8_CongestionEnergy_Tradeoff.pdf"

    create_figure(
        out_png=out_png,
        out_pdf=out_pdf,
        csv_path=root / "ieeeComparativeByPlannerFleet.csv",
    )

    print(f"Saved: {out_png}")
    print(f"Saved: {out_pdf}")
    print(
        "\nSuggested caption:\n"
        "IUEF-EM congestion-energy trade-off versus swarm size. Total fleet energy "
        "grows monotonically with the number of UAVs because per-agent overheads are "
        "additive, while the conflict rate stays at zero up to U = 3 and then rises "
        "steeply. Enlarging the fleet beyond the conflict-free region therefore costs "
        "energy and safety without improving coverage."
    )


if __name__ == "__main__":
    main()
