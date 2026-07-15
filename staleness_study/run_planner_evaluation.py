#!/usr/bin/env python3
"""Run the full planner evaluation study for publication (Section V).

Usage
-----
    python run_planner_evaluation.py              # full study (N=50)
    python run_planner_evaluation.py --quick      # smoke test (N=8)
    python run_planner_evaluation.py --n-mc 60    # custom MC count

Outputs -> planner_output/
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from epca_staleness.planner_evaluation import (
    ablation_study,
    baseline_comparison,
    staleness_planner_sweep,
    sweep_fleet_size,
    stats_to_table,
    save_results,
)
from epca_staleness.planner_plots import (
    plot_swarm_size_sweep,
    plot_ablation_bars,
    plot_baseline_bars,
    plot_staleness_planner,
    write_ablation_latex_table,
)
from epca_staleness.registry import BASELINE_PLANNERS, ABLATION_PLANNERS


def _serialize_stats(stats_dict):
    return {k: asdict(v) for k, v in stats_dict.items()}


def main():
    parser = argparse.ArgumentParser(description="EPCA-M planner evaluation study")
    parser.add_argument("--out", default="planner_output")
    parser.add_argument("--n-mc", type=int, default=50)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    n_mc = 6 if args.quick else args.n_mc
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    fleet_planners = ["iuef_em", "darp", "priority_tsp", "greedy"] if args.quick else [
        "iuef_em", "darp", "priority_tsp", "lawnmower", "greedy"]
    U_vals = [1, 2, 3, 4] if args.quick else [1, 2, 3, 4, 5, 6, 8, 10]

    print(f"=== Planner evaluation (N={n_mc}) ===\n")

    # 1. Ablation study
    print("[1/4] Ablation study...")
    ablation = ablation_study(n_mc=n_mc)
    plot_ablation_bars(ablation, out)
    write_ablation_latex_table(ablation, out)
    ab_rows = stats_to_table(ablation)
    with open(out / "ablation_results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=ab_rows[0].keys())
        w.writeheader()
        w.writerows(ab_rows)
    print("  Ablation HPC:", {k: f"{v.hpc_mean:.1f}" for k, v in ablation.items()})

    # 2. Baseline comparison
    print("[2/4] Baseline comparison...")
    bl_planners = ["iuef_em", "darp", "priority_tsp", "lawnmower", "greedy"] if args.quick else None
    if bl_planners:
        from epca_staleness.planner_evaluation import monte_carlo, EvalScenario
        baselines = monte_carlo(bl_planners, EvalScenario(), n_mc=n_mc, seed_base=7000)
    else:
        baselines = baseline_comparison(n_mc=n_mc)
    plot_baseline_bars(baselines, out)
    bl_rows = stats_to_table(baselines)
    with open(out / "baseline_comparison.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=bl_rows[0].keys())
        w.writeheader()
        w.writerows(bl_rows)

    # 3. Fleet size sweep
    print("[3/4] Fleet size sweep...")
    fleet = sweep_fleet_size(fleet_planners, U_values=U_vals, n_mc=n_mc)
    plot_swarm_size_sweep(fleet, out)

    # 4. Staleness x planner
    print("[4/4] Staleness-planner sweep...")
    tau_grid = [20, 40, 60, 80] if args.quick else [15, 25, 35, 45, 60, 80, 100]
    stale = staleness_planner_sweep(
        planners=["iuef_em", "darp", "priority_tsp", "greedy"],
        tau_grid=tau_grid, n_mc=min(20, n_mc),
    )
    plot_staleness_planner(stale, out)

    payload = {
        "n_mc": n_mc,
        "ablation": _serialize_stats(ablation),
        "baselines": _serialize_stats(baselines),
        "fleet_sweep": {str(U): _serialize_stats(fleet[U]) for U in fleet},
        "staleness_sweep": {str(t): _serialize_stats(stale[t]) for t in stale},
    }
    save_results(payload, out)

    print(f"\nDone. Outputs in {out.resolve()}")
    print("\n--- Ablation summary (HPC %, collision) ---")
    for k, v in ablation.items():
        print(f"  {k:28s}  HPC={v.hpc_mean:5.1f}±{v.hpc_std:.1f}  coll={v.collision_mean:.3f}")
    print("\n--- Baseline ranking by mission score (WHPC × safety / energy) ---")
    ranked = sorted(baselines.items(), key=lambda x: -x[1].mission_score_mean)
    for k, v in ranked:
        print(f"  {k:22s}  score={v.mission_score_mean:.3f}  HPC={v.hpc_mean:5.1f}  "
              f"early={v.hpc_early_mean:5.1f}  coll={v.collision_mean:.3f}")


if __name__ == "__main__":
    main()
