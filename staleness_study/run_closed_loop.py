#!/usr/bin/env python3
"""Run the closed-loop EPCA-M Monte-Carlo study (sensing -> planning).

Usage
-----
    python run_closed_loop.py              # full study (N=50)
    python run_closed_loop.py --quick      # smoke test (~3 min)
    python run_closed_loop.py --n-mc 30 --out closed_loop_output
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from epca_closed_loop.experiments import run_full_study
from epca_closed_loop.plots import generate_all_figures


def main():
    p = argparse.ArgumentParser(description="Closed-loop EPCA-M simulation study")
    p.add_argument("--out", type=Path, default=Path("closed_loop_output"))
    p.add_argument("--n-mc", type=int, default=50)
    p.add_argument("--quick", action="store_true", help="Reduced MC count and tau grid")
    args = p.parse_args()

    print(f"Running closed-loop study (N={args.n_mc}, quick={args.quick}) ...")
    summary = run_full_study(args.out, n_mc=args.n_mc, quick=args.quick)
    print("Generating figures ...")
    generate_all_figures(summary, args.out)
    print(f"Done. Outputs in {args.out.resolve()}")


if __name__ == "__main__":
    main()
