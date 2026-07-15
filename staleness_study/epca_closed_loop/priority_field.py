"""Priority-field fusion (Eq. 3) from Tier-2 detector + forecaster outputs.

Paper Eq. (3)::

    W_i = alpha * H_i + beta * sigma_i - gamma * O_i

In the closed loop:
  * H_i  — spatial stress from EPCA-Det-s (rasterised bbox confidence)
  * sigma_i — fused sensing uncertainty (detector entropy + forecaster variance)
  * O_i  — static obstacle-proximity penalty from the map
  * temporal risk from the MLP forecaster modulates H_i via a hotspot term
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from epca_staleness.environment import PriorityField

from .detector import Detection


@dataclass
class PriorityFusionConfig:
    """Weights for Eq. (3) and cross-modal fusion."""

    alpha: float = 1.0          # health / spatial stress weight
    beta: float = 0.55          # uncertainty weight
    gamma: float = 0.32         # traversability penalty weight
    w_spatial: float = 0.70     # detector contribution to H
    w_temporal: float = 0.30    # forecaster contribution to H
    hotspot_frac: float = 0.12  # top fraction flagged high-priority
    raster_sigma_cells: float = 1.8  # Gaussian splat radius for detections
    iot_interp_sigma: float = 6.0    # Gaussian spread for IoT station risk


def _gaussian_splat(grid: np.ndarray, r: int, c: int, value: float, sigma: float) -> None:
    """Add a Gaussian kernel centred at (r, c)."""
    n, m = grid.shape
    rad = int(max(2, round(3 * sigma)))
    for dr in range(-rad, rad + 1):
        for dc in range(-rad, rad + 1):
            rr, cc = r + dr, c + dc
            if 0 <= rr < n and 0 <= cc < m:
                d2 = dr * dr + dc * dc
                grid[rr, cc] += value * np.exp(-d2 / (2 * sigma * sigma))


def rasterize_detections(n: int, m: int,
                         detections: list[Detection],
                         obstacle: np.ndarray,
                         sigma_cells: float = 1.8) -> tuple[np.ndarray, np.ndarray]:
    """Rasterise detector outputs to spatial stress H_hat and uncertainty."""
    H_map = np.zeros((n, m), dtype=np.float32)
    conf_map = np.zeros((n, m), dtype=np.float32)
    for det in detections:
        val = det.stress_score * det.confidence
        _gaussian_splat(H_map, det.row, det.col, val, sigma_cells)
        _gaussian_splat(conf_map, det.row, det.col, det.confidence, sigma_cells)
    # Discretise to {0..4} health levels (EPCA-M convention).
    if H_map.max() > 0:
        thr = np.percentile(H_map[H_map > 0], [25, 50, 75])
        H_disc = np.zeros_like(H_map)
        H_disc[H_map > 0] = 1
        H_disc[H_map > thr[0]] = 2
        H_disc[H_map > thr[1]] = 3
        H_disc[H_map > thr[2]] = 4
    else:
        H_disc = H_map.copy()
    H_disc[obstacle] = 0
    # Uncertainty: low confidence -> high sigma.
    sigma_map = 1.0 - np.clip(conf_map / max(conf_map.max(), 1e-6), 0, 1)
    sigma_map = 0.15 + 0.85 * sigma_map
    sigma_map[obstacle] = 1.0
    return H_disc, sigma_map


def interpolate_iot_risk(n: int, m: int,
                         station_coords: np.ndarray,
                         risks: np.ndarray,
                         obstacle: np.ndarray,
                         sigma: float = 6.0) -> np.ndarray:
    """Spread forecaster station risks over the grid (inverse-distance Gaussian)."""
    risk_map = np.zeros((n, m), dtype=np.float32)
    if len(station_coords) == 0:
        return risk_map
    for (r, c), risk in zip(station_coords, risks):
        _gaussian_splat(risk_map, int(r), int(c), float(risk), sigma)
    if risk_map.max() > 0:
        risk_map /= risk_map.max()
    risk_map[obstacle] = 0.0
    return risk_map


def fuse_priority_field(field: PriorityField,
                        detections: list[Detection],
                        iot_coords: np.ndarray,
                        iot_risks: np.ndarray,
                        cfg: PriorityFusionConfig | None = None) -> tuple[np.ndarray, np.ndarray, dict]:
    """Fuse Tier-2 outputs into composite priority weight W and hotspot mask.

    Returns
    -------
    W : (N, M) fused priority field
    high_mask : (N, M) bool hotspot mask
    meta : diagnostic dict
    """
    cfg = cfg or PriorityFusionConfig()
    n, m = field.N, field.M
    obs = field.obstacle

    H_sp, sigma_det = rasterize_detections(n, m, detections, obs, cfg.raster_sigma_cells)
    risk_temp = interpolate_iot_risk(n, m, iot_coords, iot_risks, obs, cfg.iot_interp_sigma)

    # Cross-modal fusion of spatial (CV) and temporal (IoT) stress.
    H_cont = cfg.w_spatial * (H_sp / 4.0) + cfg.w_temporal * risk_temp
    H_fused = np.minimum(4, np.round(4.0 * H_cont)).astype(float)
    H_fused[obs] = 0

    # Forecaster variance proxy: spatial gradient of risk map.
    grad = np.gradient(risk_temp)
    forecaster_unc = np.clip(np.hypot(grad[0], grad[1]) * 2.0, 0, 1)
    sigma_fused = np.clip(0.5 * sigma_det + 0.5 * (0.2 + 0.8 * forecaster_unc), 0.05, 1.0)
    sigma_fused[obs] = 1.0

    O = field.O
    W = cfg.alpha * H_fused + cfg.beta * sigma_fused - cfg.gamma * O

    # Hotspot mask: top hotspot_frac of traversable cells by W.
    trav = ~obs
    Wtrav = W[trav]
    if Wtrav.size > 0:
        w_hi = float(np.quantile(Wtrav, 1.0 - cfg.hotspot_frac))
    else:
        w_hi = 3.0
    high_mask = (W >= w_hi) & trav

    meta = dict(
        alpha=cfg.alpha, beta=cfg.beta, gamma=cfg.gamma,
        w_hi=w_hi, n_detections=len(detections), n_iot_stations=len(iot_coords),
        mean_cv_stress=float(H_sp[trav].mean()) if trav.any() else 0.0,
        mean_iot_risk=float(risk_temp[trav].mean()) if trav.any() else 0.0,
    )
    return W, high_mask, meta


def build_field_from_inference(field: PriorityField,
                               detections: list[Detection],
                               iot_coords: np.ndarray,
                               iot_risks: np.ndarray,
                               cfg: PriorityFusionConfig | None = None) -> PriorityField:
    """Return a new PriorityField with inferred W / high_mask (static map kept)."""
    W, high_mask, meta = fuse_priority_field(field, detections, iot_coords, iot_risks, cfg)
    return PriorityField(
        N=field.N, M=field.M, dx=field.dx,
        H=np.minimum(4, np.round(W / max(cfg.alpha if cfg else 1.0, 1e-6))).astype(float),
        sigma=field.sigma,  # replaced below from fusion
        obstacle=field.obstacle, O=field.O, W=W,
        high_mask=high_mask, Z_m=field.Z_m,
        meta={**field.meta, **meta, "source": "closed_loop_inference"},
    )
