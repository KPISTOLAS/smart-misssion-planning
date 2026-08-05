#!/usr/bin/env python3
"""Unified critical-fixes validation driver (review checklist).

Runs:
  1. Table II — YOLO Params/GFLOPs (all EPCA-Det variants)
  2. Calibrated staleness + NTN channel τ sweep (multi-seed MC)
  3. ±30% parameter perturbation on operating τ bound
  4. Closed-loop end-to-end (detector + forecaster → planner)
  5. Forecaster baselines (persistence, AR(1), ARIMA)
  6. Detector k-fold evaluation
  7. Planner baselines (IUEF-EM, DARP, priority-TSP)

Usage:
    python3 run_critical_fixes.py --quick     # smoke test (~5 min)
    python3 run_critical_fixes.py --n-mc 50   # publication run
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from epca_models.yolo_profile import write_table_ii
from epca_staleness.experiments import sweep_tau, default_params, operating_bounds
from epca_staleness.staleness import calibrated_defaults
from epca_staleness.channel import NTNChannel
from epca_sensitivity.perturbation import run_full_perturbation_study, latex_table_perturbation
from epca_closed_loop.experiments import compare_modes, run_single_trial
from epca_closed_loop.forecaster_baselines import evaluate_forecaster_baselines
from epca_closed_loop.data_synth import sample_iot_windows
from epca_closed_loop.detector_eval import run_detector_kfold
from epca_sensitivity.map_generator import SyntheticMapGenerator, SyntheticMapConfig


def _sanitize(obj):
    if hasattr(obj, "__dict__"):
        return {k: _sanitize(v) for k, v in vars(obj).items()}
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize(x) for x in obj]
    if isinstance(obj, float) and (obj != obj or abs(obj) == float("inf")):
        return None
    return obj


def run_planner_baselines(n_mc: int, seed_base: int = 40000) -> dict:
    """Quick IUEF-EM vs DARP vs priority-TSP (no staleness, fair planner compare)."""
    import numpy as np
    from epca_staleness.executor import evaluate_planner
    from epca_sensitivity.map_generator import SyntheticMapGenerator, SyntheticMapConfig
    gen = SyntheticMapGenerator()
    results = {}
    for name in ("iuef_em", "darp", "priority_tsp"):
        hpc, coll = [], []
        for j in range(n_mc):
            field = gen.generate(SyntheticMapConfig(seed=seed_base + j))
            m = evaluate_planner(name, field, 3, seed=seed_base + j)
            hpc.append(m.hpc_pct)
            coll.append(m.collision_rate)
        results[name] = dict(
            hpc_mean=float(np.mean(hpc)), hpc_std=float(np.std(hpc, ddof=1)),
            coll_mean=float(np.mean(coll)), coll_std=float(np.std(coll, ddof=1)),
        )
    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=Path("critical_fixes_output"))
    p.add_argument("--n-mc", type=int, default=50)
    p.add_argument("--n-seeds", type=int, default=10, help="RNG seeds for MC")
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    n_mc = 8 if args.quick else args.n_mc
    n_seeds = 3 if args.quick else args.n_seeds
    summary = dict(n_mc=n_mc, n_seeds=n_seeds, quick=args.quick)

    print("1/7 Table II — YOLO Params/GFLOPs ...")
    summary["table_ii"] = write_table_ii(out)

    print("2/7 Calibrated staleness + NTN τ sweep (multi-seed) ...")
    params = calibrated_defaults()
    summary["calibrated_params"] = dict(
        beta_M=params.beta_M, sigma_M=params.sigma_M, sigma_g=params.sigma_g,
    )
    channel = NTNChannel("medium")
    summary["ntn_channel"] = channel.describe()
    tau_sweep = sweep_tau("medium", n_mc=n_mc, params=params)
    summary["tau_sweep"] = dict(
        tau_mean=tau_sweep.tau_mean.tolist(),
        hpc_mean=tau_sweep.hpc_mean.tolist(),
        hpc_lo=tau_sweep.hpc_lo.tolist(),
        hpc_hi=tau_sweep.hpc_hi.tolist(),
        coll_mean=tau_sweep.coll_mean.tolist(),
        operating_bound=operating_bounds(tau_sweep),
    )

    print("3/7 ±30% parameter perturbation ...")
    pert = run_full_perturbation_study(n_mc=min(n_mc, 10), quick=args.quick)
    summary["perturbation_30pct"] = pert
    (out / "Table_Perturbation_30.tex").write_text(latex_table_perturbation(pert))

    print("4/7 Closed-loop end-to-end ...")
    summary["closed_loop"] = compare_modes("medium", 45.0, n_mc)

    print("5/7 Forecaster baselines ...")
    gen = SyntheticMapGenerator()
    field = gen.generate(SyntheticMapConfig(seed=42))
    iot = sample_iot_windows(field, rng=42)
    from epca_closed_loop.forecaster import MLPForecaster
    mlp = MLPForecaster()
    mlp_preds = mlp.predict(iot)
    import numpy as np
    mlp_mae = float(np.mean(np.abs(mlp_preds - iot.ground_truth_risk)))
    fb = evaluate_forecaster_baselines(iot)
    summary["forecaster_baselines"] = {
        b.name: dict(mae=b.mae, rmse=b.rmse) for b in fb
    }
    summary["forecaster_baselines"]["mlp"] = dict(mae=mlp_mae)

    print("6/7 Detector k-fold ...")
    det = run_detector_kfold(n_folds=3 if args.quick else 5,
                             n_seeds=n_seeds)
    summary["detector_kfold"] = dict(
        precision_mean=det.precision_mean, precision_std=det.precision_std,
        recall_mean=det.recall_mean, recall_std=det.recall_std,
        f1_mean=det.f1_mean, f1_std=det.f1_std,
        mae_mean=det.mae_mean, mae_std=det.mae_std,
    )

    print("7/7 Planner baselines (IUEF-EM, DARP, priority-TSP) ...")
    summary["planner_baselines"] = run_planner_baselines(min(n_mc, 15))

    (out / "critical_fixes_summary.json").write_text(
        json.dumps(_sanitize(summary), indent=2))
    print(f"Done. Outputs in {out.resolve()}")


if __name__ == "__main__":
    main()
