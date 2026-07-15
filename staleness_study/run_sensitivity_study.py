#!/usr/bin/env python3
"""Run the EPCA-M synthetic-map sensitivity and operating-bound study.

Usage
-----
    python run_sensitivity_study.py              # full study (N=50)
    python run_sensitivity_study.py --quick      # smoke test
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from epca_sensitivity.experiments import run_full_sensitivity_study
from epca_sensitivity.plots import generate_all_figures


def main():
    p = argparse.ArgumentParser(description="EPCA-M sensitivity analysis study")
    p.add_argument("--out", type=Path, default=Path("sensitivity_output"))
    p.add_argument("--n-mc", type=int, default=50)
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()

    print(f"Running sensitivity study (N={args.n_mc}, quick={args.quick}) ...")
    summary = run_full_sensitivity_study(args.out, n_mc=args.n_mc, quick=args.quick)
    print("Generating figures and LaTeX tables ...")
    generate_all_figures(summary, args.out)
    print(f"Done. Outputs in {args.out.resolve()}")


if __name__ == "__main__":
    main()
