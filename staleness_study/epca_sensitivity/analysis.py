"""Statistical analysis and LaTeX table generation for sensitivity studies."""

from __future__ import annotations

from dataclasses import asdict
import numpy as np

from epca_staleness.experiments import SweepResult, _ci95, operating_bounds


def extract_operating_bound(sweep: SweepResult,
                            hpc_thr: float = 65.0,
                            coll_thr: float = 0.40) -> dict:
    """Largest mean τ satisfying HPC ≥ thr and collision ≤ thr."""
    return operating_bounds(sweep, hpc_thr=hpc_thr, coll_thr=coll_thr)


def summarise_trials(trials) -> dict:
    """Aggregate a list of SensitivityMissionResult objects."""
    hpc = np.array([t.hpc_pct for t in trials])
    coll = np.array([t.collision_rate for t in trials])
    near = np.array([t.near_miss_rate for t in trials])
    energy = np.array([t.energy_per_uav_hour for t in trials])
    mae = np.array([t.inference_mae for t in trials])
    whpc = np.array([t.whpc_pct for t in trials])
    targ = np.array([t.targeting_error_pct for t in trials])
    m_h, lo_h, hi_h = _ci95(hpc)
    m_c, lo_c, hi_c = _ci95(coll)
    return dict(
        hpc_mean=m_h, hpc_lo=lo_h, hpc_hi=hi_h,
        whpc_mean=float(np.mean(whpc)),
        coll_mean=float(np.mean(coll)), coll_lo=lo_c, coll_hi=hi_c,
        near_mean=float(np.mean(near)),
        energy_mean=float(np.mean(energy)),
        inference_mae_mean=float(np.mean(mae)),
        targeting_error_mean=float(np.mean(targ)),
        n=len(trials),
    )


def latex_table_tau_bounds(rows: dict,
                           caption: str = "Operating $\\bar{\\tau}$ bound sensitivity",
                           label: str = "tab:tau_bound_sensitivity") -> str:
    """Generate LaTeX table: rows keyed by parameter value, columns max τ / HPC / collision.

    ``rows`` format: ``{param_value: operating_bound_dict}``.
    """
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"Configuration & $\bar{\tau}_{\max}$ & HPC@bound (\%) & Coll@bound \\",
        r"\midrule",
    ]
    for key, bound in rows.items():
        if not bound.get("feasible", False):
            lines.append(f"{key} & -- & -- & -- \\\\")
        else:
            lines.append(
                f"{key} & {bound['max_tau']:.0f} & {bound['hpc_at_bound']:.1f} & {bound['coll_at_bound']:.3f} \\\\"
            )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def latex_table_hotspot_u(summary: dict) -> str:
    """Combined table: τ bound vs hotspot density and fleet size U."""
    parts = []
    if "hotspot_density" in summary:
        bounds = {k: v["operating_bound"] for k, v in summary["hotspot_density"].items()}
        parts.append(latex_table_tau_bounds(
            bounds,
            caption="Operating $\\bar{\\tau}$ bound vs. hotspot density (HPC$>$65\\%, coll$<$0.4)",
            label="tab:tau_hotspot",
        ))
    if "fleet_size" in summary:
        bounds = {f"$U={k}$": v["operating_bound"] for k, v in summary["fleet_size"].items()}
        parts.append(latex_table_tau_bounds(
            bounds,
            caption="Operating $\\bar{\\tau}$ bound vs. fleet size $U$",
            label="tab:tau_fleet",
        ))
    return "\n\n".join(parts)


def latex_table_priority_quality(summary: dict) -> str:
    """Perfect vs imperfect priority comparison table."""
    pq = summary.get("priority_quality", {})
    if not pq:
        return ""
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Planner robustness: perfect vs. imperfect priority fields ($\bar{\tau}=45$, medium link)}",
        r"\label{tab:priority_quality}",
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        r"Priority source & HPC (\%) & WHPC (\%) & Collision & Targeting err. (\%) & Inf. MAE \\",
        r"\midrule",
    ]
    labels = {"perfect": "Perfect $W_i$", "imperfect_mild": "Noisy (10\\%)",
              "imperfect_severe": "Noisy (20\\%)"}
    for key, row in pq.items():
        lbl = labels.get(key, key)
        lines.append(
            f"{lbl} & {row['hpc_mean']:.1f} & {row.get('whpc_mean', 0):.1f} & {row['coll_mean']:.3f} & "
            f"{row.get('targeting_error_mean', 0):.1f} & {row.get('inference_mae_mean', 0):.2f} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def insight_sentences(summary: dict) -> list[str]:
    """Generate deployment-relevant insight bullets from summary JSON."""
    insights = []
    hd = summary.get("hotspot_density", {})
    if hd:
        taus = {k: v["operating_bound"].get("max_tau") for k, v in hd.items()
                if v.get("operating_bound", {}).get("feasible")}
        if len(taus) >= 2:
            # Lower max_tau = tighter bound = less tolerant of staleness.
            tightest = min(taus, key=lambda k: taus[k] or 0)
            loosest = max(taus, key=lambda k: taus[k] or 0)
            if taus[tightest] != taus[loosest]:
                insights.append(
                    f"Hotspot density '{tightest}' yields the tightest operating τ bound "
                    f"({taus[tightest]:.0f} steps vs {taus[loosest]:.0f} for '{loosest}'), "
                    f"because denser or more heterogeneous hotspot fields require fresher "
                    f"priority beliefs to maintain HPC above 65%."
                )
    fs = summary.get("fleet_size", {})
    if fs:
        taus = {int(k): v["operating_bound"].get("max_tau") for k, v in fs.items()
                if v["operating_bound"].get("feasible")}
        if taus:
            best_u = max(taus, key=lambda u: taus[u] or 0)
            insights.append(
                f"Fleet size U={best_u} achieves the largest operating τ bound "
                f"({taus[best_u]:.0f} steps), reflecting improved hotspot visitation capacity."
            )
    pq = summary.get("priority_quality", {})
    if pq and "perfect" in pq and "imperfect_severe" in pq:
        dh = pq["perfect"]["hpc_mean"] - pq["imperfect_severe"]["hpc_mean"]
        if abs(dh) > 0.5:
            direction = "reduces" if dh > 0 else "increases"
            insights.append(
                f"Severe detector/forecaster noise (20% FP/FN) {direction} HPC by "
                f"~{abs(dh):.1f} pp relative to perfect priority fields at matched staleness."
            )
    return insights
