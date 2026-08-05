#!/usr/bin/env python3
"""Methodology validation driver (reviewer fixes 1–5).

Runs held-out NTN calibration, CI/significance tables, metadata AoI sweep,
R3 forecaster ablation, and per-fold detector F1 diagnostics.

Usage:
    python3 run_methodology_validation.py --quick     # N=20 smoke
    python3 run_methodology_validation.py --n-mc 150  # publication
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from epca_staleness.ntn_calibration_validation import run_calibration_validation
from epca_staleness.metadata_queue import run_metadata_aoi_sweep
from epca_staleness.methodology_stats import export_all_comparison_tables
from epca_closed_loop.forecaster_ablation import run_r3_forecaster_ablation, export_r3_forecaster_csv
from epca_closed_loop.detector_eval import run_detector_kfold, export_per_fold_diagnostics


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=Path("methodology_validation_output"))
    p.add_argument("--n-mc", type=int, default=150)
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()

    n_mc = 20 if args.quick else args.n_mc
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    summary = {"n_mc": n_mc, "quick": args.quick}

    print("1/5 NTN calibration held-out validation ...")
    summary["ntn_calibration"] = run_calibration_validation(out)

    print("2/5 CI + significance tables (R2-S3, R1/R2 ablations, dx sweep) ...")
    summary["comparison_tables"] = export_all_comparison_tables(out, n_mc=n_mc, quick=args.quick)

    print("3/5 Metadata queue AoI sweep (loss × RTT) ...")
    summary["metadata_aoi"] = run_metadata_aoi_sweep(out)

    print("4/5 R3 forecaster ablation (MLP vs persistence vs AR1) ...")
    r3 = run_r3_forecaster_ablation(n_mc=n_mc)
    summary["r3_forecaster"] = {
        "seed_base": r3["seed_base"],
        "mae_mean": r3["mae_mean"],
        "hpc": {k: dict(mean=v.mean, ci95_low=v.ci95_low, ci95_high=v.ci95_high)
                for k, v in r3["hpc_rows"].items()},
        "csv": str(export_r3_forecaster_csv(r3, out)),
    }

    print("5/5 Detector k-fold per-fold F1 diagnostics ...")
    det = run_detector_kfold(n_folds=5, n_seeds=10 if not args.quick else 3, seed_base=1000)
    summary["detector_kfold"] = export_per_fold_diagnostics(det, out)
    summary["detector_kfold"]["f1_mean"] = det.f1_mean
    summary["detector_kfold"]["f1_std"] = det.f1_std

    (out / "methodology_validation_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(f"\nDone → {out.resolve()}")
    print("CSVs:")
    for pat in (
        "table_ntn_calibration_validation.csv",
        "ntn_calibration_points.csv",
        "table_R2S3_sync_policy.csv",
        "table_R2S3_sync_policy_collision.csv",
        "table_R1_ablation.csv",
        "table_R2_ablation_tau50.csv",
        "table_grid_resolution_dx.csv",
        "table_metadata_aoi_sweep.csv",
        "table_R3_forecaster_ablation.csv",
        "table_detector_kfold_f1_per_fold.csv",
    ):
        fp = out / pat
        if fp.exists():
            print(f"  {fp}")


if __name__ == "__main__":
    main()
