#!/usr/bin/env python3
"""Publication-grade study (N=50): regimes, staleness ablations, operating envelope.

Usage:
    python3 run_publication_study.py --quick       # smoke test
    python3 run_publication_study.py --n-mc 50   # paper run (~30-60 min)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from epca_staleness.publication import run_publication_study


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=Path("publication_output"))
    p.add_argument("--n-mc", type=int, default=50)
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()
    print(f"Publication study (N={args.n_mc}, quick={args.quick})")
    run_publication_study(args.out, n_mc=args.n_mc, quick=args.quick)
    print(f"Done → {args.out.resolve()}")


if __name__ == "__main__":
    main()
