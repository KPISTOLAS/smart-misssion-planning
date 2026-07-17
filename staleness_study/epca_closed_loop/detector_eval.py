"""Detector evaluation with multi-seed k-fold on synthetic Plant Health corpus."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
import numpy as np

from epca_sensitivity.map_generator import SyntheticMapGenerator, SyntheticMapConfig
from .data_synth import sample_uav_images
from .detector import EPCADetector, DetectorConfig


@dataclass
class DetectorFoldResult:
    fold: int
    seed: int
    precision: float
    recall: float
    f1: float
    mae_stress: float
    n_images: int


@dataclass
class DetectorKFoldSummary:
    n_folds: int
    n_seeds: int
    precision_mean: float
    precision_std: float
    recall_mean: float
    recall_std: float
    f1_mean: float
    f1_std: float
    mae_mean: float
    mae_std: float
    per_fold: list[DetectorFoldResult]


def _evaluate_batch(batch, dets, stress_thr: float = 0.5) -> tuple[float, float, float, float]:
    """Precision/recall/F1/MAE for stress detection vs ground truth."""
    gt_pos = batch.ground_truth_stress >= stress_thr
    n = len(batch.ground_truth_stress)
    if n == 0:
        return 0, 0, 0, 0
    pred_stress = np.array([d.stress_score for d in dets], dtype=float)
    pred_pos = pred_stress >= stress_thr
    tp = int(np.sum(gt_pos & pred_pos))
    fp = int(np.sum(~gt_pos & pred_pos))
    fn = int(np.sum(gt_pos & ~pred_pos))
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)
    mae = float(np.mean(np.abs(pred_stress - batch.ground_truth_stress)))
    return float(prec), float(rec), float(f1), mae


def export_per_fold_diagnostics(
    summary: DetectorKFoldSummary,
    out_dir: Path | str,
) -> dict:
    """Per-fold F1 scores, outlier flags (IQR rule), CSV export."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Aggregate per fold (mean over seeds within fold).
    fold_f1: dict[int, list[float]] = {}
    for fr in summary.per_fold:
        fold_f1.setdefault(fr.fold, []).append(fr.f1)
    fold_means = {f: float(np.mean(v)) for f, v in fold_f1.items()}

    vals = np.array(list(fold_means.values()))
    q1, q3 = np.percentile(vals, [25, 75])
    iqr = q3 - q1
    lo_fence = q1 - 1.5 * iqr
    hi_fence = q3 + 1.5 * iqr
    outliers = [f for f, m in fold_means.items() if m < lo_fence or m > hi_fence]

    csv_path = out_dir / "table_detector_kfold_f1_per_fold.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["fold", "n_seeds_in_fold", "f1_mean", "f1_std", "outlier_flag",
                     "global_f1_mean", "global_f1_std"])
        for fold in sorted(fold_f1.keys()):
            arr = np.array(fold_f1[fold])
            w.writerow([
                fold, len(arr), f"{arr.mean():.4f}", f"{arr.std(ddof=1):.4f}",
                "YES" if fold in outliers else "no",
                f"{summary.f1_mean:.4f}", f"{summary.f1_std:.4f}",
            ])
        w.writerow([])
        w.writerow(["seed", "fold", "f1", "precision", "recall", "mae_stress"])
        for fr in summary.per_fold:
            w.writerow([fr.seed, fr.fold, f"{fr.f1:.4f}", f"{fr.precision:.4f}",
                        f"{fr.recall:.4f}", f"{fr.mae_stress:.4f}"])

    return {
        "fold_f1_means": fold_means,
        "outlier_folds": outliers,
        "iqr_fence": [float(lo_fence), float(hi_fence)],
        "csv": str(csv_path),
        "interpretation": (
            f"F1 variance driven by fold(s) {outliers} as outliers (IQR rule)"
            if outliers else "No fold exceeds IQR outlier fence; variance is diffuse across folds"
        ),
    }


def run_detector_kfold(n_folds: int = 5,
                       n_seeds: int = 10,
                       seed_base: int = 1000) -> DetectorKFoldSummary:
    """K-fold detector eval across synthetic maps with multiple RNG seeds."""
    gen = SyntheticMapGenerator()
    detector = EPCADetector(DetectorConfig())
    folds: list[DetectorFoldResult] = []

    for fold in range(n_folds):
        for si in range(n_seeds):
            seed = seed_base + fold * 100 + si
            field = gen.generate(SyntheticMapConfig(seed=seed))
            positions = np.array([[field.N // 2, field.M // 2],
                                  [field.N // 4, field.M // 4],
                                  [3 * field.N // 4, 3 * field.M // 4]], dtype=float)
            batch = sample_uav_images(field, positions, views_per_uav=12, rng=seed + 7)
            dets = detector.predict(batch, (field.N, field.M))
            prec, rec, f1, mae = _evaluate_batch(batch, dets)
            folds.append(DetectorFoldResult(
                fold=fold, seed=seed, precision=prec, recall=rec, f1=f1,
                mae_stress=mae, n_images=len(batch.images),
            ))

    precs = np.array([f.precision for f in folds])
    recs = np.array([f.recall for f in folds])
    f1s = np.array([f.f1 for f in folds])
    maes = np.array([f.mae_stress for f in folds])
    return DetectorKFoldSummary(
        n_folds=n_folds, n_seeds=n_seeds,
        precision_mean=float(precs.mean()), precision_std=float(precs.std(ddof=1)),
        recall_mean=float(recs.mean()), recall_std=float(recs.std(ddof=1)),
        f1_mean=float(f1s.mean()), f1_std=float(f1s.std(ddof=1)),
        mae_mean=float(maes.mean()), mae_std=float(maes.std(ddof=1)),
        per_fold=folds,
    )
