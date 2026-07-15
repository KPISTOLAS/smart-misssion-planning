"""Synthetic + semi-real sensor streams for closed-loop EPCA-M simulation.

Generates UAV RGB patches (Plant Health Tracker corpus logic) and IoT
time-series windows (Herbal Plant dataset logic) from a ground-truth
:class:`~epca_staleness.environment.PriorityField`.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from epca_staleness.environment import PriorityField


@dataclass
class UAVImageBatch:
    """RGB images captured along planned / executed UAV trajectories."""

    images: np.ndarray          # (B, H, W, 3) uint8
    cell_coords: np.ndarray     # (B, 2) row/col grid indices
    uav_ids: np.ndarray         # (B,) which UAV captured each frame
    ground_truth_stress: np.ndarray  # (B,) continuous stress in [0, 1]


@dataclass
class IoTWindowBatch:
    """Sliding windows of IoT features at fixed station locations."""

    features: np.ndarray        # (S, T, F) stations x time x features
    station_coords: np.ndarray  # (S, 2) row/col
    ground_truth_risk: np.ndarray  # (S,) temporal anomaly risk in [0, 1]


# Herbal Plant–style feature names (temperature, humidity, soil moisture, light).
IOT_FEATURE_NAMES = ("temperature_C", "humidity_pct", "soil_moisture", "light_lux")


def _stress_from_field(field: PriorityField) -> np.ndarray:
    """Continuous plant-stress proxy from discrete health + uncertainty."""
    H = field.H.astype(float)
    sigma = field.sigma.astype(float)
    stress = 0.65 * (H / 4.0) + 0.35 * sigma
    stress[field.obstacle] = 0.0
    return np.clip(stress, 0.0, 1.0)


def _rgb_from_stress(stress_val: float, rng) -> np.ndarray:
    """Plant Health Tracker–style colour: healthy green -> stressed yellow/brown."""
    h, w = 128, 128
    base_green = np.array([40, 140, 60], dtype=np.float32)
    stress_col = np.array([180, 120, 40], dtype=np.float32)
    col = (1.0 - stress_val) * base_green + stress_val * stress_col
    img = np.tile(col, (h, w, 1))
    # Leaf texture + sensor noise (semi-real appearance).
    tex = rng.normal(0, 12, size=(h, w, 3))
    spots = rng.random((h, w)) < (0.02 + 0.08 * stress_val)
    img[spots] = img[spots] * 0.6 + np.array([90, 70, 30])
    img = np.clip(img + tex, 0, 255).astype(np.uint8)
    return img


def sample_uav_images(field: PriorityField,
                      positions: np.ndarray,
                      uav_ids: np.ndarray | None = None,
                      views_per_uav: int = 8,
                      rng=None) -> UAVImageBatch:
    """Generate RGB frames for UAVs at their current (or recent) positions.

  Plant Health Tracker corpus logic: each frame is a top-down crop centred on
  the overflown cell with colour/texture correlated to local stress.
    """
    rng = np.random.default_rng(rng)
    stress = _stress_from_field(field)
    U = len(positions)
    if uav_ids is None:
        uav_ids = np.arange(U)

    imgs, coords, uids, gt = [], [], [], []
    n, m = field.N, field.M
    for u in range(U):
        r0, c0 = int(round(positions[u, 0])), int(round(positions[u, 1]))
        for _ in range(views_per_uav):
            # Jitter within a 3x3 neighbourhood (UAV footprint).
            dr, dc = rng.integers(-1, 2), rng.integers(-1, 2)
            r, c = int(np.clip(r0 + dr, 0, n - 1)), int(np.clip(c0 + dc, 0, m - 1))
            if field.obstacle[r, c]:
                continue
            s = float(stress[r, c])
            imgs.append(_rgb_from_stress(s, rng))
            coords.append([r, c])
            uids.append(uav_ids[u] if u < len(uav_ids) else u)
            gt.append(s)
    if not imgs:
        r, c = 2, 2
        imgs = [_rgb_from_stress(float(stress[r, c]), rng)]
        coords = [[r, c]]
        uids = [0]
        gt = [float(stress[r, c])]
    return UAVImageBatch(
        images=np.stack(imgs),
        cell_coords=np.array(coords, dtype=int),
        uav_ids=np.array(uids, dtype=int),
        ground_truth_stress=np.array(gt, dtype=float),
    )


def _place_iot_stations(field: PriorityField, n_stations: int, rng) -> np.ndarray:
    """Scatter IoT nodes on traversable cells (Herbal Plant field layout)."""
    trav = np.argwhere(field.traversable)
    if trav.size == 0:
        return np.zeros((0, 2), dtype=int)
    n = min(n_stations, len(trav))
    idx = rng.choice(len(trav), size=n, replace=False)
    return trav[idx]


def sample_iot_windows(field: PriorityField,
                       window_len: int = 24,
                       n_stations: int = 16,
                       rng=None) -> IoTWindowBatch:
    """Generate IoT sliding windows with Herbal Plant–style correlations.

  Features: temperature, humidity, soil moisture, light.  Anomaly risk rises
  with local stress and adds autocorrelated noise (drought / pest signatures).
    """
    rng = np.random.default_rng(rng)
    stress = _stress_from_field(field)
    stations = _place_iot_stations(field, n_stations, rng)
    S = len(stations)
    F = len(IOT_FEATURE_NAMES)
    feats = np.zeros((S, window_len, F), dtype=np.float32)
    gt_risk = np.zeros(S, dtype=np.float32)

    for s, (r, c) in enumerate(stations):
        base_stress = float(stress[r, c])
        # Baseline environmental state modulated by stress.
        temp0 = 22.0 + 8.0 * base_stress + rng.normal(0, 0.5)
        hum0 = 70.0 - 35.0 * base_stress + rng.normal(0, 2)
        soil0 = 0.45 - 0.25 * base_stress + rng.normal(0, 0.02)
        light0 = 800.0 + rng.normal(0, 50)
        ar = 0.85
        temp, hum, soil, light = temp0, hum0, soil0, light0
        for t in range(window_len):
            shock = rng.normal(0, 1) * (0.3 + 0.7 * base_stress)
            temp = ar * temp + (1 - ar) * temp0 + shock * 0.8
            hum = ar * hum + (1 - ar) * hum0 - shock * 2.5
            soil = ar * soil + (1 - ar) * soil0 - shock * 0.015
            light = ar * light + (1 - ar) * light0 + rng.normal(0, 30)
            feats[s, t] = [temp, hum, soil, light]
        # Ground-truth risk: stress + recent trend.
        trend = float(np.mean(feats[s, -6:, 0]) - np.mean(feats[s, :6, 0]))
        gt_risk[s] = float(np.clip(0.7 * base_stress + 0.3 * np.tanh(trend / 3.0), 0, 1))

    return IoTWindowBatch(
        features=feats,
        station_coords=stations,
        ground_truth_risk=gt_risk,
    )
