"""Smoke tests for methodology validation modules."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from epca_staleness.ntn_calibration_validation import (
    generate_calibration_points,
    run_calibration_validation,
    split_train_test,
)
from epca_staleness.metadata_queue import run_metadata_aoi_sweep, effective_aoi_kappa
from epca_closed_loop.detector_eval import run_detector_kfold, export_per_fold_diagnostics


def test_calibration_generates_24_points():
    pts = generate_calibration_points()
    assert len(pts) == 24
    assert len({p.link_class for p in pts}) == 3


def test_train_test_split_holds_out_poor():
    pts = generate_calibration_points()
    train, test = split_train_test(pts)
    assert len(train) == 16
    assert len(test) == 8
    assert all(p.link_class == "poor" for p in test)


def test_calibration_validation_writes_csv(tmp_path):
    result = run_calibration_validation(tmp_path)
    assert "train_R2" in result
    assert (tmp_path / "table_ntn_calibration_validation.csv").exists()


def test_metadata_aoi_sweep_monotone_in_loss():
    k_lo = effective_aoi_kappa(45.0, 0.05, 300.0)
    k_hi = effective_aoi_kappa(45.0, 0.20, 300.0)
    assert k_hi > k_lo


def test_detector_per_fold_export(tmp_path):
    det = run_detector_kfold(n_folds=3, n_seeds=2, seed_base=42)
    diag = export_per_fold_diagnostics(det, tmp_path)
    assert "fold_f1_means" in diag
    assert (tmp_path / "table_detector_kfold_f1_per_fold.csv").exists()
