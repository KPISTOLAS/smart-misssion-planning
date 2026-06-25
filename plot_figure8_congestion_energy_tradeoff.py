from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


def _apply_ieee_style() -> None:
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


def create_figure(out_png: Path, out_pdf: Path) -> None:
    u = np.arange(1, 9)
    energy_wh = np.array([182, 124, 101, 94, 92, 91, 91, 92], dtype=float)
    congestion_pct = np.array([3, 6, 10, 15, 22, 30, 39, 51], dtype=float)
    detour_pct = np.array([2, 4, 7, 11, 16, 22, 29, 37], dtype=float)

    energy_std = None
    congestion_std = None
    detour_std = None

    color_energy = "#1575A8"
    color_congestion = "#E7A400"
    color_detour = "#0FA07A"

    _apply_ieee_style()

    fig, ax_left = plt.subplots(figsize=(3.6, 2.55))
    fig.patch.set_facecolor("white")
    ax_left.set_facecolor("white")
    ax_right = ax_left.twinx()

    # Neutral band so it does not clash with the detour (teal) line.
    ax_left.axvspan(2.7, 4.3, color="#D9D9D9", alpha=0.45, lw=0, zorder=0)
    ax_left.text(
        3.5,
        0.97,
        "Recommended\nU = 3–4",
        transform=ax_left.get_xaxis_transform(),
        ha="center",
        va="top",
        fontsize=7,
        color="#444444",
        linespacing=1.05,
        zorder=1,
    )

    line_kw = dict(lw=1.35, zorder=4)
    if energy_std is not None:
        energy_line = ax_left.errorbar(
            u,
            energy_wh,
            yerr=energy_std,
            fmt="-o",
            color=color_energy,
            ms=4.0,
            capsize=2.0,
            label="Energy per mission",
            **line_kw,
        )
    else:
        energy_line = ax_left.plot(
            u,
            energy_wh,
            "-o",
            color=color_energy,
            ms=4.0,
            label="Energy per mission",
            **line_kw,
        )[0]

    if congestion_std is not None:
        congestion_line = ax_right.errorbar(
            u,
            congestion_pct,
            yerr=congestion_std,
            fmt="-s",
            color=color_congestion,
            ms=3.8,
            capsize=2.0,
            label="Congestion overhead",
            **line_kw,
        )
    else:
        congestion_line = ax_right.plot(
            u,
            congestion_pct,
            "-s",
            color=color_congestion,
            ms=3.8,
            label="Congestion overhead",
            **line_kw,
        )[0]

    if detour_std is not None:
        detour_line = ax_right.errorbar(
            u,
            detour_pct,
            yerr=detour_std,
            fmt="--^",
            color=color_detour,
            ms=3.6,
            capsize=2.0,
            label="Detour overhead",
            **line_kw,
        )
    else:
        detour_line = ax_right.plot(
            u,
            detour_pct,
            "--^",
            color=color_detour,
            ms=3.6,
            label="Detour overhead",
            **line_kw,
        )[0]

    ax_left.set_xlabel("Swarm size (UAVs)")
    ax_left.set_ylabel("Energy per mission (Wh)", color=color_energy)
    ax_right.set_ylabel("Overhead (%)", color="#333333")

    ax_left.set_xlim(0.8, 8.2)
    ax_left.set_xticks(u)
    ax_left.set_ylim(85, 190)
    ax_right.set_ylim(0, 55)
    ax_right.set_yticks(np.arange(0, 56, 10))

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
    ax_left.tick_params(axis="y", length=3, width=0.6, colors=color_energy, labelcolor=color_energy)
    ax_right.tick_params(axis="y", length=3, width=0.6, colors="#333333", labelcolor="#333333")

    band_patch = mpatches.Patch(facecolor="#D9D9D9", edgecolor="none", alpha=0.45, label="Recommended U = 3–4")
    handles = [energy_line, congestion_line, detour_line, band_patch]
    labels = [h.get_label() for h in handles[:3]] + ["Recommended U = 3–4"]

    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.16),
        ncol=2,
        frameon=False,
        handlelength=1.6,
        columnspacing=0.9,
        handletextpad=0.45,
    )

    fig.subplots_adjust(left=0.17, right=0.84, bottom=0.20, top=0.78)
    fig.savefig(out_png, dpi=300, facecolor="white")
    fig.savefig(out_pdf, facecolor="white")
    plt.close(fig)


def main() -> None:
    root = Path(__file__).resolve().parent
    out_png = root / "Figure8_CongestionEnergy_Tradeoff.png"
    out_pdf = root / "Figure8_CongestionEnergy_Tradeoff.pdf"

    create_figure(out_png=out_png, out_pdf=out_pdf)

    print(f"Saved: {out_png}")
    print(f"Saved: {out_pdf}")


if __name__ == "__main__":
    main()
