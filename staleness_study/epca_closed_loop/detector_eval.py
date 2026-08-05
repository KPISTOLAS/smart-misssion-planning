"""Detector evaluation with multi-seed k-fold on synthetic Plant Health corpus."""

from __future__ import annotations

from dataclasses import dataclass
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
