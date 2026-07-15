#!/usr/bin/env python3
"""Run the enhanced EPCA-M staleness Monte-Carlo study.

Usage
-----
    # Full study (N=50 per sweep point, ~15-25 min on a laptop):
    python run_staleness_study.py

    # Quick smoke test (N=12, coarser tau grid, ~2 min):
    python run_staleness_study.py --quick

    # Custom output directory and MC count:
    python run_staleness_study.py --out results --n-mc 60

Outputs are written to ``output/`` (or ``--out``):
  * Figure_Staleness_TauSweep.{png,pdf}
  * Figure_Staleness_PolicyComparison.{png,pdf}
  * Figure_Staleness_Sensitivity.{png,pdf}
  * Figure_Staleness_Calibration.{png,pdf}
  * staleness_study_summary.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure package is importable when run as a script.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from epca_staleness.experiments import run_full_study


def main():
  parser = argparse.ArgumentParser(description="EPCA-M enhanced staleness Monte-Carlo study")
  parser.add_argument("--out", type=str, default="output", help="Output directory")
  parser.add_argument("--n-mc", type=int, default=50, help="Monte-Carlo trials per sweep point")
  parser.add_argument("--quick", action="store_true", help="Fast smoke test (N=12, coarse grid)")
  args = parser.parse_args()

  summary = run_full_study(out_dir=args.out, n_mc=args.n_mc, quick=args.quick)

  print("\n=== Operating bounds (HPC >= 65 %, collision < 0.4) ===")
  for link, b in summary["operating_bounds"].items():
    if b["feasible"]:
      print(f"  {link:8s}: max mean tau = {b['max_tau']:.0f} steps  "
            f"(HPC={b['hpc_at_bound']:.1f} %, coll={b['coll_at_bound']:.3f})")
    else:
      print(f"  {link:8s}: no feasible tau in sweep range")

  print("\n=== Calibrated parameters ===")
  p = summary["params"]
  print(f"  beta_M = {p['beta_M']:.5f}")
  print(f"  sigma_M = {p['sigma_M']:.3f}")
  print(f"  sigma_g = {p['sigma_g']:.4f}")


if __name__ == "__main__":
  main()
