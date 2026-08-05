"""Staleness / Age-of-Information model for EPCA-M (absolute-age formulation).

Model (reviewer-corrected)
--------------------------
* **Age** is absolute steps since last sync (not normalized by τ)::

      age(t) = Δ_k = steps since last synchronization event

* **Mean AoI** over a periodic interval of length τ (ages 0 … τ−1)::

      κ(τ) = (τ − 1) / 2          # grows linearly with τ

* **Map retention** (exponential fade from last synced truth, no compounding)::

      retention(Δ) = exp(−β_M · Δ)
      M̂(Δ) = max(0, M_sync · retention(Δ) + N(0, σ_M² · Δ))

  where ``M_sync`` is the priority field at the last sync (inference or truth).

* **Ghost drift** (2-D Brownian random walk in grid cells)::

      p̃ = p + N(0, σ_g² · Δ)   per coordinate
      RMSE(Δ) = σ_g · √(2Δ)    (Euclidean, cells)
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import numpy as np


# Reference calibration targets (solved, not tuned to legacy broken outputs)
R_TARGET = 0.60
DELTA_REF = 60
GHOST_RMSE_TARGET_CELLS = 10.0


@dataclass
class StalenessParams:
    """Degradation parameters for the absolute-age staleness model."""

    beta_M: float = 0.008510641896545124   # −ln(0.60)/60
    sigma_M: float = 0.08
    sigma_g: float = 0.9128709291755658    # 10 / √(2·60)


def retention(delta: float, beta_M: float) -> float:
    """Exponential map retention ``exp(−β_M · Δ)``."""
    return float(np.exp(-beta_M * max(float(delta), 0.0)))


def age_of(step_in_interval: int, tau: int | None = None) -> int:
    """Absolute AoI: integer steps since last sync (Δ_k).

    ``tau`` is accepted for API compatibility but not used — age is never
    normalized by the interval length.
    """
    return max(int(step_in_interval), 0)


def kappa(tau) -> np.ndarray:
    """Mean AoI over one interval when age is absolute: ``(τ−1)/2``."""
    tau = np.asarray(tau, dtype=float)
    return (tau - 1.0) / 2.0


def calibrate_beta_M(target_retention: float = R_TARGET,
                   delta_ref: float = DELTA_REF) -> float:
    """Solve ``β_M = −ln(R_target) / Δ_ref``."""
    if not (0.0 < target_retention < 1.0):
        raise ValueError("target_retention must lie in (0, 1).")
    return float(-np.log(target_retention) / max(float(delta_ref), 1.0))


def calibrate_ghost_sigma(target_rmse_cells: float = GHOST_RMSE_TARGET_CELLS,
                          delta_ref: float = DELTA_REF,
                          n_mc: int = 40000,
                          rng=None) -> float:
    """Calibrate σ_g so RMSE(Δ_ref) = target (Brownian ghost model)."""
    denom = np.sqrt(2.0 * max(float(delta_ref), 1.0))
    sigma_g = float(target_rmse_cells / denom)
    model = StalenessModel(StalenessParams(sigma_g=sigma_g), rng=rng)
    true_pos = np.zeros((n_mc, 2))
    ghosts = model.ghost_positions(true_pos, int(delta_ref))
    emp = float(np.sqrt(np.mean(np.sum(ghosts ** 2, axis=1))))
    if emp > 1e-9:
        sigma_g *= target_rmse_cells / emp
    return sigma_g


def calibrated_defaults(delta_ref: int = DELTA_REF,
                        ghost_rmse_cells: float = GHOST_RMSE_TARGET_CELLS,
                        map_retention: float = R_TARGET,
                        rng=None) -> StalenessParams:
    """Return analytically calibrated staleness parameters."""
    beta_M = calibrate_beta_M(map_retention, delta_ref)
    sigma_g = calibrate_ghost_sigma(ghost_rmse_cells, delta_ref, rng=rng)
    return StalenessParams(beta_M=beta_M, sigma_M=0.08, sigma_g=sigma_g)


def uncalibrated_defaults() -> StalenessParams:
    """Legacy paper defaults — retained only for regression comparison."""
    return StalenessParams(beta_M=0.65, sigma_M=0.08, sigma_g=0.30)


# Back-compat alias (old name referred to cumulative integral model)
def interval_retention(beta_M: float, tau: float, n_grid: int = 64) -> float:
    """Deprecated alias → :func:`retention` with absolute age Δ = τ."""
    return retention(tau, beta_M)


class StalenessModel:
    """Map fade and ghost drift under absolute age Δ (steps since sync)."""

    def __init__(self, params: StalenessParams | None = None, rng=None):
        self.params = params or StalenessParams()
        self.rng = np.random.default_rng(rng)

    def degraded_map(self, M_sync: np.ndarray, delta: int | float) -> np.ndarray:
        """Fade ``M_sync`` by absolute age Δ (no compounding off stale belief)."""
        d = max(float(delta), 0.0)
        p = self.params
        R = retention(d, p.beta_M)
        noise = self.rng.normal(0.0, p.sigma_M * np.sqrt(d), size=M_sync.shape)
        return np.maximum(0.0, M_sync * R + noise)

    def fade_map(self, M_sync: np.ndarray, age: float) -> np.ndarray:
        """Alias for :meth:`degraded_map` (``M_sync`` must be last-synced truth)."""
        return self.degraded_map(M_sync, age)

    def fade_map_multi(self, M_sync: np.ndarray, delta: float, n_steps: int) -> np.ndarray:
        """Apply degradation at fixed Δ (diagnostic helper)."""
        return self.degraded_map(M_sync, delta)

    def ghost_positions(self, true_pos: np.ndarray, age: float,
                        tau: int | None = None) -> np.ndarray:
        """Brownian ghost drift: variance ``σ_g² · Δ`` per coordinate."""
        p = self.params
        var = (p.sigma_g ** 2) * max(float(age), 0.0)
        std = np.sqrt(max(var, 0.0))
        return np.asarray(true_pos, dtype=float) + self.rng.normal(0.0, std, size=np.shape(true_pos))

    def ghost_rmse(self, age: float, tau: int | None = None) -> float:
        """Analytic 2-D Euclidean RMSE in cells: ``σ_g √(2Δ)``."""
        return self.params.sigma_g * np.sqrt(2.0 * max(float(age), 0.0))

    def ghost_rmse_m(self, age: float, dx_m: float = 18.0) -> float:
        """Ghost RMSE in metres."""
        return self.ghost_rmse(age) * dx_m


def emit_calibration_report(out_path: Path | str | None = None,
                          delta_ref: float = DELTA_REF,
                          r_target: float = R_TARGET,
                          ghost_rmse_cells: float = GHOST_RMSE_TARGET_CELLS,
                          dx_m: float = 18.0,
                          d_safe_m: float = 25.0) -> dict:
    """Write calibration derivation to ``calibration_report.json``."""
    beta_M = calibrate_beta_M(r_target, delta_ref)
    sigma_g = calibrate_ghost_sigma(ghost_rmse_cells, delta_ref)
    rmse_60_cells = sigma_g * np.sqrt(2.0 * delta_ref)
    rmse_60_m = rmse_60_cells * dx_m

    report = {
        "R_target": r_target,
        "Delta_ref": delta_ref,
        "beta_M": beta_M,
        "beta_M_derivation": f"beta_M = -ln({r_target}) / {delta_ref}",
        "sigma_g": sigma_g,
        "ghost_rmse_target_cells": ghost_rmse_cells,
        "ghost_rmse_at_Delta_ref_cells": float(rmse_60_cells),
        "ghost_rmse_at_Delta_ref_m": float(rmse_60_m),
        "d_safe_m": d_safe_m,
        "ghost_rmse_exceeds_d_safe": bool(rmse_60_m > d_safe_m),
        "retention_at_20": retention(20, beta_M),
        "retention_at_60": retention(delta_ref, beta_M),
        "retention_at_160": retention(160, beta_M),
        "kappa_at_40": float(kappa(40)),
        "kappa_at_80": float(kappa(80)),
        "kappa_ratio_80_40": float(kappa(80) / kappa(40)),
        "p_out_per_class": None,  # filled by link_budget.py when available
    }

    if out_path is not None:
        p = Path(out_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(report, indent=2))
    return report


# Legacy names kept for imports
def calibrate_map_fade(target_retention: float = R_TARGET,
                       tau_ref: int = DELTA_REF,
                       age_ref: float | None = None) -> float:
    return calibrate_beta_M(target_retention, tau_ref)


def calibrate_ghost_sigma_legacy(target_rmse_cells: float,
                                 tau_ref: int = DELTA_REF,
                                 age_ref: float | None = None,
                                 n_mc: int = 40000,
                                 rng=None) -> float:
    return calibrate_ghost_sigma(target_rmse_cells, tau_ref, n_mc, rng)


def calibrate_map_noise(target_std: float, age_ref: float | None = None,
                        tau_ref: int = DELTA_REF) -> float:
    d = float(age_ref if age_ref is not None else tau_ref)
    return float(target_std / np.sqrt(max(d, 1e-12)))
