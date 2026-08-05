"""Imperfect priority-field models for detector and forecaster error injection."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from epca_staleness.environment import PriorityField, _distance_to_obstacles


@dataclass
class ImperfectPriorityConfig:
    """Noise parameters for imperfect Tier-2 priority reconstruction."""

    false_positive_rate: float = 0.15
    false_negative_rate: float = 0.15
    confidence_noise_std: float = 0.12   # multiplicative noise on W
    localization_shift_cells: float = 2.0
    iot_noise_std: float = 0.10
    spike_probability: float = 0.05
    spike_magnitude: float = 3.0
    min_fp_distance_cells: float = 8.0  # FP peaks placed far from true hotspots
    enabled: bool = True


def _gaussian_peak(n: int, m: int, r: int, c: int, amp: float, width: float) -> np.ndarray:
    ii, jj = np.meshgrid(np.arange(n), np.arange(m), indexing="ij")
    return amp * np.exp(-((ii - r) ** 2 + (jj - c) ** 2) / (2 * width ** 2))


def _hotspot_distance_map(gt_high: np.ndarray, obstacle: np.ndarray) -> np.ndarray:
    """Distance (cells) from each traversable cell to nearest true hotspot."""
    n, m = gt_high.shape
    D = _distance_to_obstacles(~gt_high)  # distance to non-hotspot; invert logic
    # Multi-source BFS from hotspot cells.
    from collections import deque
    dist = np.full((n, m), np.inf)
    q = deque()
    for r, c in np.argwhere(gt_high):
        dist[r, c] = 0.0
        q.append((r, c))
    nbr = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
    while q:
        r, c = q.popleft()
        base = dist[r, c]
        for dr, dc in nbr:
            nr, nc = r + dr, c + dc
            if 0 <= nr < n and 0 <= nc < m and not obstacle[nr, nc] and base + 1 < dist[nr, nc]:
                dist[nr, nc] = base + 1
                q.append((nr, nc))
    dist[obstacle] = 0
    return dist


def corrupt_priority_field(field: PriorityField,
                           cfg: ImperfectPriorityConfig | None = None,
                           rng=None) -> tuple[np.ndarray, np.ndarray, dict]:
    """Return imperfect ``(W_hat, high_mask_hat, diagnostics)`` from ground truth."""
    cfg = cfg or ImperfectPriorityConfig()
    rng = np.random.default_rng(rng)
    if not cfg.enabled:
        return field.W.copy(), field.high_mask.copy(), dict(mode="perfect")

    n, m = field.N, field.M
    trav = field.traversable
    gt_high = field.high_mask
    W_true = field.W.copy()
    W_hat = W_true.copy()

    # False negatives: attenuate ALL hotspot cells (systematic detector recall loss).
    if gt_high.any():
        atten = 1.0 - cfg.false_negative_rate * rng.uniform(0.6, 1.0)
        W_hat[gt_high] *= atten

    # Additional random FN on subset (missed pockets).
    fn_mask = gt_high & (rng.random((n, m)) < cfg.false_negative_rate * 0.5)
    W_hat[fn_mask] *= rng.uniform(0.05, 0.25, size=int(fn_mask.sum()))

    # Localization shift: smear hotspot energy to neighbours (bbox offset).
    shift = int(round(cfg.localization_shift_cells))
    if shift != 0:
        smeared = np.zeros_like(W_hat)
        for r, c in np.argwhere(gt_high):
            for dr in range(-shift, shift + 1):
                for dc in range(-shift, shift + 1):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < n and 0 <= nc < m and trav[nr, nc]:
                        smeared[nr, nc] += W_true[r, c] * 0.25
        blend = 0.35
        W_hat = (1 - blend) * W_hat + blend * smeared

    # False positives: misleading peaks far from true hotspots.
    dist_to_hot = _hotspot_distance_map(gt_high, field.obstacle)
    fp_candidates = np.argwhere(trav & ~gt_high & (dist_to_hot >= cfg.min_fp_distance_cells))
    n_fp = max(3, int(cfg.false_positive_rate * 30))
    n_fp = min(n_fp, len(fp_candidates)) if len(fp_candidates) else 0
    fp_layer = np.zeros((n, m))
    if n_fp > 0:
        pick = fp_candidates[rng.choice(len(fp_candidates), size=n_fp, replace=False)]
        w_ref = float(np.percentile(W_true[gt_high], 90)) if gt_high.any() else 5.0
        for r, c in pick:
            amp = rng.uniform(1.2, 2.0) * w_ref
            fp_layer += _gaussian_peak(n, m, int(r), int(c), amp, rng.uniform(2.5, 5.0))

    # Forecaster noise + occasional spikes on non-hotspot cells.
    W_hat += rng.normal(0, cfg.iot_noise_std * W_true[trav].std(), size=W_hat.shape)
    spikes = (rng.random((n, m)) < cfg.spike_probability) & trav & ~gt_high
    W_hat[spikes] += cfg.spike_magnitude

    W_hat = W_hat + fp_layer
    W_hat *= (1.0 + rng.normal(0, cfg.confidence_noise_std, size=W_hat.shape))
    W_hat = np.maximum(0.0, W_hat)
    W_hat[~trav] = 0.0

    Wtrav = W_hat[trav]
    frac = field.meta.get("hotspot_frac", 0.12)
    w_hi = float(np.quantile(Wtrav, 1.0 - frac)) if Wtrav.size else 3.0
    high_hat = (W_hat >= w_hi) & trav

    diag = dict(
        mode="imperfect",
        n_false_negatives=int(fn_mask.sum()),
        n_false_positives=int(n_fp),
        n_spikes=int(spikes.sum()),
        w_hi_hat=w_hi,
        mae_hotspots=float(np.mean(np.abs(W_hat[gt_high] - W_true[gt_high]))) if gt_high.any() else 0.0,
    )
    return W_hat, high_hat, diag
