#!/usr/bin/env python3
"""Model-rebuild driver: smoke (N=20) then production (N=200).

Order: calibrated model → stats harness → extended τ grid → link budget → figures.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from epca_staleness.experiments import run_full_study
from epca_staleness.publication import run_publication_study
from epca_staleness.run_manifest import write_run_manifest
from epca_staleness.sweep_config import TAU_SWEEP_N_MC, TAU_SWEEP_N_MC_SMOKE
from epca_staleness.staleness import emit_calibration_report
import link_budget


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=Path("model_rebuild_output"))
    p.add_argument("--smoke", action="store_true", help="N=20 smoke sanity run")
    p.add_argument("--n-mc", type=int, default=None)
    args = p.parse_args()

    n_mc = args.n_mc or (TAU_SWEEP_N_MC_SMOKE if args.smoke else TAU_SWEEP_N_MC)
    quick = args.smoke
    out = args.out

    print(f"Model rebuild study (N={n_mc}, smoke={quick})")
    emit_calibration_report(out / "calibration_report.json")
    write_run_manifest(out, seed_list=list(range(1000, 1000 + n_mc)))
    budget = link_budget.emit_link_budget(out)
    from epca_staleness.channel import sync_presets_from_budget
    sync_presets_from_budget(budget)
    print(f"  Link budget: good SNR={budget['classes']['good']['mean_snr_db']:.1f} dB, "
          f"p_out={budget['classes']['good']['p_out']:.3f}")
    if "aoi_surface" in budget:
        print(f"  AoI finding: {budget['aoi_surface']['finding']}")

    print("  Core staleness sweeps ...")
    run_full_study(out / "staleness", n_mc=n_mc, quick=quick)

    print("  Publication regimes ...")
    run_publication_study(out / "publication", n_mc=n_mc, quick=quick)

    print(f"Done → {out.resolve()}")


if __name__ == "__main__":
    main()
