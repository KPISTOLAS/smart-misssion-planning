"""R3 closed-loop forecaster ablation (MLP vs persistence vs AR(1))."""

from __future__ import annotations

import csv
from pathlib import Path
import numpy as np

from epca_staleness.stats_export import aggregate_metric, compare_variants

from .closed_loop import ClosedLoopConfig, SyncPolicy, run_closed_loop
from epca_staleness.channel import NTNChannel
from epca_staleness.environment import build_priority_field
from epca_staleness.experiments import default_params

SEED_R3_FORECASTER = 8000


def run_r3_forecaster_ablation(
    n_mc: int = 150,
    link: str = "medium",
    base_tau: float = 45.0,
    seed_base: int = SEED_R3_FORECASTER,
) -> dict:
    """R3 closed-loop under MLP, persistence, and AR(1) forecasters."""
    modes = ("mlp", "persistence", "ar1")
    raw_hpc = {m: [] for m in modes}
    raw_coll = {m: [] for m in modes}
    raw_mae = {m: [] for m in modes}

    for j in range(n_mc):
        seed = seed_base + j
        for mode in modes:
            field = build_priority_field(50, 50, seed=seed)
            channel = NTNChannel(link, rng=seed + 17, base_tau_override=base_tau)
            tau_ref = channel.expected_tau(2000)
            cfg = ClosedLoopConfig(
                horizon=600,
                policy=SyncPolicy.PERIODIC,
                tau_ref=tau_ref,
                forecaster_mode=mode,
            )
            r = run_closed_loop(
                field, cfg,
                staleness_params=default_params(rng=seed),
                channel=NTNChannel(link, rng=seed + 17, base_tau_override=base_tau),
                rng=seed + 31,
            )
            raw_hpc[mode].append(r.hpc_pct)
            raw_coll[mode].append(r.collision_rate)
            raw_mae[mode].append(r.inference_mae)

    hpc_rows = {m: aggregate_metric(m, raw_hpc[m]) for m in modes}
    coll_rows = {m: aggregate_metric(f"{m}_coll", raw_coll[m]) for m in modes}
    ref = hpc_rows["mlp"]
    comparisons = compare_variants(ref, {k: v for k, v in hpc_rows.items() if k != "mlp"}, "HPC_pct")

    return {
        "n_mc": n_mc,
        "seed_base": seed_base,
        "link": link,
        "base_tau": base_tau,
        "hpc_rows": hpc_rows,
        "collision_rows": coll_rows,
        "mae_mean": {m: float(np.mean(raw_mae[m])) for m in modes},
        "comparisons_vs_mlp": comparisons,
    }


def export_r3_forecaster_csv(result: dict, out_dir: Path | str) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "table_R3_forecaster_ablation.csv"
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["forecaster", "n_seeds", "HPC_mean", "HPC_ci95_low", "HPC_ci95_high",
                     "collision_mean", "inference_mae_mean", "seed_base"])
        sb = result["seed_base"]
        for mode in ("mlp", "persistence", "ar1"):
            row = result["hpc_rows"][mode]
            coll = result["collision_rows"][mode]
            w.writerow([
                mode, row.n_seeds, f"{row.mean:.2f}", f"{row.ci95_low:.2f}", f"{row.ci95_high:.2f}",
                f"{coll.mean:.4f}", f"{result['mae_mean'][mode]:.4f}", sb,
            ])
        w.writerow([])
        w.writerow(["comparison", "reference", "wilcoxon_p", "cliffs_delta", "fdr_p"])
        for c in result["comparisons_vs_mlp"]:
            w.writerow([c.variant, c.reference, f"{c.wilcoxon_p:.4g}", f"{c.cliffs_delta:.4f}",
                        f"{c.fdr_p:.4g}" if c.fdr_p is not None else ""])
    return path
