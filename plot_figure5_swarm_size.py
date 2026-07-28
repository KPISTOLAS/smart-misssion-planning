from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_measured_data(
    csv_path: Path,
    planner: str = "priority",
    u_min: int = 1,
    u_max: int = 8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Load mission data for a given planner and U-range.

    Returns:
      U, mission_time_mean_min, risk_mean, mission_time_std_min, risk_std, coverage_mean_pct
    """
    df = pd.read_csv(csv_path)
    subset = df[(df["planner"] == planner) & (df["numUAV"].between(u_min, u_max))].copy()
    subset = subset.sort_values("numUAV")

    if subset.empty:
        raise ValueError(
            f"No rows found for planner='{planner}' in U={u_min}..{u_max}. "
            "Check planner name or CSV contents."
        )

    # If repeated-trial rows exist, aggregate mean and std per U.
    grouped = subset.groupby("numUAV", as_index=False).agg(
        mission_time_s_mean=("mean_mission_time_s", "mean"),
        mission_time_s_std=("mean_mission_time_s", "std"),
        risk_mean=("mean_any_collision", "mean"),
        risk_std=("mean_any_collision", "std"),
    )

    # Coverage is optional. If present, use it as a proxy metric.
    coverage_mean_pct = None
    if "mean_HPC_pct" in subset.columns:
        cov_grouped = subset.groupby("numUAV", as_index=False).agg(
            coverage_mean_pct=("mean_HPC_pct", "mean")
        )
        grouped = grouped.merge(cov_grouped, on="numUAV", how="left")
        coverage_mean_pct = grouped["coverage_mean_pct"].to_numpy()

    u = grouped["numUAV"].to_numpy()
    mission_time_mean_min = grouped["mission_time_s_mean"].to_numpy() / 60.0
    mission_time_std_min = grouped["mission_time_s_std"].to_numpy() / 60.0
    risk_mean = grouped["risk_mean"].to_numpy()
    risk_std = grouped["risk_std"].to_numpy()

    if np.isnan(mission_time_std_min).all():
        mission_time_std_min = None
    if np.isnan(risk_std).all():
        risk_std = None

    return u, mission_time_mean_min, risk_mean, mission_time_std_min, risk_std, coverage_mean_pct


def create_figure(
    u: np.ndarray,
    mission_time_min: np.ndarray,
    risk_index: np.ndarray,
    mission_time_std: Optional[np.ndarray],
    risk_std: Optional[np.ndarray],
    out_png: Path,
    out_pdf: Path,
) -> None:
    # IEEE-friendly style
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 10,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "figure.dpi": 120,
            "savefig.bbox": "tight",
        }
    )

    # IEEE-like clean style with requested line colors.
    fig, ax = plt.subplots(figsize=(6.2, 3.5))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    color_time = "#0070B0"  # previous blue
    color_risk = "#E09800"  # previous orange

    time_range = float(np.nanmax(mission_time_min) - np.nanmin(mission_time_min))
    risk_range = float(np.nanmax(risk_index) - np.nanmin(risk_index))
    time_range = time_range if time_range > 0 else 1.0
    risk_range = risk_range if risk_range > 0 else 1.0

    mission_time_norm = (mission_time_min - np.nanmin(mission_time_min)) / time_range
    risk_norm = (risk_index - np.nanmin(risk_index)) / risk_range

    # Mission time curve
    if mission_time_std is not None:
        line_time = ax.errorbar(
            u,
            mission_time_norm,
            yerr=mission_time_std / time_range,
            fmt="-o",
            color=color_time,
            lw=1.4,
            ms=3.6,
            capsize=2,
            label="Mission time (IUEF-EM)",
            zorder=3,
        )
    else:
        line_time = ax.plot(
            u,
            mission_time_norm,
            "-o",
            color=color_time,
            lw=1.4,
            ms=3.6,
            label="Mission time (IUEF-EM)",
            zorder=3,
        )[0]

    # Collision risk curve
    if risk_std is not None:
        line_risk = ax.errorbar(
            u,
            risk_norm,
            yerr=risk_std / risk_range,
            fmt="--s",
            color=color_risk,
            lw=1.3,
            ms=3.4,
            capsize=2,
            label="Collision risk (IUEF-EM)",
            zorder=3,
        )
    else:
        line_risk = ax.plot(
            u,
            risk_norm,
            "--s",
            color=color_risk,
            lw=1.3,
            ms=3.4,
            label="Collision risk (IUEF-EM)",
            zorder=3,
        )[0]

    # Operating point: fastest mission among the collision-free fleet sizes.
    safe = np.flatnonzero(risk_index <= 0)
    u_opt_idx = safe[int(np.argmin(mission_time_norm[safe]))] if safe.size else int(
        np.argmin(mission_time_norm)
    )
    ax.axvline(u[u_opt_idx], color="#555555", lw=0.8, ls=":", alpha=0.75, zorder=2)
    ax.annotate(
        f"U = {int(u[u_opt_idx])}\nmin time,\nno conflicts",
        xy=(u[u_opt_idx], mission_time_norm[u_opt_idx]),
        xytext=(-30, 92),
        textcoords="offset points",
        fontsize=7.2,
        color="#333333",
        linespacing=1.15,
        arrowprops=dict(arrowstyle="->", color="#555555", lw=0.7),
    )

    ax.set_xlabel("Swarm Size (UAV count)")
    ax.set_ylabel("Normalized Metric Value (0-1)")
    ax.set_xlim(min(u) - 0.25, max(u) + 0.25)
    ax.set_xticks(u)
    ax.set_ylim(-0.03, 1.03)
    ax.grid(axis="y", linestyle=":", alpha=0.2, linewidth=0.55)

    # Black axes and ticks for print consistency.
    for spine in ax.spines.values():
        spine.set_color("black")
        spine.set_linewidth(0.8)
    ax.tick_params(axis="both", colors="black")

    # Legend outside to avoid overlap with data.
    handles = [line_time, line_risk]
    labels = [h.get_label() for h in handles]
    ax.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.18), ncol=2, frameon=False)

    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(out_png, dpi=300)
    fig.savefig(out_pdf)
    plt.close(fig)


def main() -> None:
    root = Path(__file__).resolve().parent
    csv_path = root / "ieeeComparativeByPlannerFleet.csv"

    # Use measured values when available
    u, mission_time_min, risk_index, mission_time_std, risk_std, _coverage_pct = load_measured_data(
        csv_path=csv_path,
        planner="priority",
        u_min=1,
        u_max=8,
    )

    out_png = root / "Figure5_SwarmSize_vs_MissionPerformance.png"
    out_pdf = root / "Figure5_SwarmSize_vs_MissionPerformance.pdf"
    create_figure(
        u=u,
        mission_time_min=mission_time_min,
        risk_index=risk_index,
        mission_time_std=mission_time_std,
        risk_std=risk_std,
        out_png=out_png,
        out_pdf=out_pdf,
    )

    caption = (
        "IUEF-EM mission performance versus swarm size (normalized for single-axis "
        "comparison). Mission time falls steeply up to U = 3 and then rises again as "
        "inter-agent contention outweighs added parallelism, while coordination risk "
        "stays at zero through U = 3 before climbing sharply. The two curves jointly "
        "identify U = 3 as the operating point: minimum mission time among the "
        "collision-free fleet sizes."
    )
    print(f"Saved: {out_png}")
    print(f"Saved: {out_pdf}")
    print("\nSuggested caption:")
    print(caption)


if __name__ == "__main__":
    main()
