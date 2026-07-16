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
    manifest = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_sha": _git_sha(),
        "python_version": sys.version,
        "platform": platform.platform(),
        "numpy_version": np.__version__,
        "torch_version": _pkg_version("torch"),
        "ultralytics_version": _pkg_version("ultralytics"),
        # Planner weights (Algorithm 1 / Eq. 1)
        "w_T": 1.0,
        "w_E": 0.3,
        "w_C": 0.2,
        "w_P": 0.5,
        "eta": opts.eta,
        "kappa_w": opts.blend_gamma,
        "kappa_d": 0.12,
        "kappa_e": 0.12,
        "omega_d": 0.45,
        "omega_s": 0.35,
        "omega_h": opts.lambda_slope,   # reconciled: lambda_slope ≡ omega_h
        "tau_prune": opts.horizon_cells,
        "lambda_blend": opts.blend_gamma,
        "lambda_cong": opts.lambda_cong,
        "H_c": opts.horizon_cells,
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
