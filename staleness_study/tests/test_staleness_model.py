"""Unit tests for the absolute-age staleness model."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from epca_staleness.staleness import (
    R_TARGET,
    DELTA_REF,
    StalenessModel,
    StalenessParams,
    age_of,
    calibrate_beta_M,
    calibrate_ghost_sigma,
    emit_calibration_report,
    kappa,
    retention,
)


def test_age_is_absolute_not_normalized():
    assert age_of(40, tau=80) == 40
    assert age_of(40, tau=200) == 40


def test_kappa_grows_linearly_with_tau():
    assert kappa(80) / kappa(40) == pytest.approx(2.0, rel=0.02)


def test_retention_calibration_at_delta_ref():
    beta_M = calibrate_beta_M(R_TARGET, DELTA_REF)
    assert retention(DELTA_REF, beta_M) == pytest.approx(R_TARGET, abs=0.005)


def test_beta_M_analytic_value():
    beta_M = calibrate_beta_M(0.60, 60)
    assert beta_M == pytest.approx(-np.log(0.60) / 60, rel=1e-6)
    assert beta_M == pytest.approx(0.00851, rel=0.01)


def test_retention_monotone_decreasing_in_delta():
    beta_M = calibrate_beta_M()
    assert retention(20, beta_M) > retention(60, beta_M) > retention(160, beta_M)


def test_ghost_rmse_scales_with_sqrt_delta():
    p = StalenessParams(sigma_g=1.0)
    m = StalenessModel(p, rng=42)
    assert m.ghost_rmse(60) / m.ghost_rmse(15) == pytest.approx(np.sqrt(4.0), rel=0.01)


def test_degraded_map_fades_from_sync_not_iterative_stale():
    """Belief at age Δ equals M_sync · exp(−β_M Δ) (single reference point)."""
    p = StalenessParams(beta_M=0.01, sigma_M=0.0)
    m = StalenessModel(p, rng=0)
    M_sync = np.ones((3, 3)) * 10.0
    d60 = m.degraded_map(M_sync, 60)
    expected = M_sync.mean() * retention(60, 0.01)
    assert d60.mean() == pytest.approx(expected, rel=0.01)
    # Iterative per-step compounding (old bug) would use previous stale as base each step.
    stale_iter = M_sync.copy()
    for _ in range(60):
        stale_iter = stale_iter * (1.0 - 0.01 * (1.0 / 60.0))  # old normalized-age step
    assert d60.mean() != pytest.approx(stale_iter.mean(), rel=0.05)


def test_calibration_report_emitted(tmp_path):
    report = emit_calibration_report(tmp_path / "calibration_report.json")
    assert report["retention_at_60"] == pytest.approx(0.60, abs=0.005)
    assert (tmp_path / "calibration_report.json").exists()
    loaded = json.loads((tmp_path / "calibration_report.json").read_text())
    assert loaded["kappa_ratio_80_40"] == pytest.approx(2.0, rel=0.02)
