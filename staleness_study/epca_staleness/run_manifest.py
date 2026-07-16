"""Run manifest: full reproducibility metadata per experiment."""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from epca_staleness.environment import priority_field_W_stats
from epca_staleness.iuef_em import IUEFEMOptions
from epca_staleness.staleness import calibrated_defaults, emit_calibration_report


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:
        return "unknown"


def _pkg_version(name: str) -> str:
    try:
        mod = __import__(name)
        return getattr(mod, "__version__", "unknown")
    except Exception:
        return "not_installed"


def build_manifest(seed_list: list[int] | None = None,
                   extra: dict[str, Any] | None = None) -> dict:
    """Collect every tunable parameter for a study run."""
    opts = IUEFEMOptions()
    params = calibrated_defaults()
    w_stats = priority_field_W_stats(seed=42)
    manifest = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_sha": _git_sha(),
        "python_version": sys.version,
        "platform": platform.platform(),
        "numpy_version": np.__version__,
        "torch_version": _pkg_version("torch"),
        "ultralytics_version": _pkg_version("ultralytics"),
        # Composite priority field (Eq. weight): W = αH + βσ − γO + hotspot
        "alpha": 1.0,
        "beta": 0.55,
        "gamma": 0.32,
        "W_stats_seed_42": w_stats,
        "W_normalization": (
            "H in {0..4} (use H/4 for [0,1]); sigma,O in [0,1]; "
            "hotspot Gaussians add 3-9 on peak cells so raw W is not in [0,1]. "
            "w_hi is a traversable quantile on raw W."
        ),
        # Insertion score (Algorithm 1): ratio form implemented in iuef_em.py
        "insertion_score_formula": "score(g) = W_g / Delta_L(g|pi_u) - kappa_d * O_g",
        "kappa_w": 1.0,
        "kappa_e": 0.0,
        "kappa_d": opts.lambda_cong,
        "insertion_note": (
            "kappa_w=1 and kappa_e=0: ordering is priority-over-distance minus "
            "congestion penalty; blend_gamma applies to target weights g_w only."
        ),
        "blend_gamma": opts.blend_gamma,
        "eta": opts.eta,
        "omega_d": 0.45,
        "omega_s": 0.35,
        "omega_h": opts.lambda_slope,
        "tau_prune": opts.horizon_cells,
        "lambda_cong": opts.lambda_cong,
        "H_c": opts.horizon_cells,
        "N_tgt_max": opts.max_targets,
        "d_safe_m": 25.0,
        "dx_m": 18.0,
        "dt_s": 1.0,
        "v_max_mps": 12.0,
        "beta_M": params.beta_M,
        "sigma_M": params.sigma_M,
        "sigma_g": params.sigma_g,
        "staleness_model": "absolute_age_exponential_retention",
        "age_formula": "age(t) = Delta_k (integer steps since sync)",
        "kappa_formula": "kappa(tau) = (tau-1)/2",
        "retention_formula": "R(Delta) = exp(-beta_M * Delta)",
        "ghost_formula": "p_tilde = p + N(0, sigma_g^2 * Delta)",
        "seed_list": seed_list or [],
    }
    if extra:
        manifest.update(extra)
    return manifest


def write_run_manifest(out_dir: Path | str, seed_list: list[int] | None = None,
                       extra: dict | None = None) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = build_manifest(seed_list, extra)
    path = out_dir / "run_manifest.json"
    path.write_text(json.dumps(manifest, indent=2))
    emit_calibration_report(out_dir / "calibration_report.json")
    return path
