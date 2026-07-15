"""Forecaster baselines: persistence, AR(1), ARIMA vs MLP."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .data_synth import IoTWindowBatch


@dataclass
class ForecasterBaselineResult:
    name: str
    predictions: np.ndarray
    mae: float
    rmse: float


def _risk_from_series(series: np.ndarray) -> float:
    """Map a 1-D feature series to scalar risk in [0, 1]."""
    if len(series) < 2:
        return float(np.clip(series[-1] if len(series) else 0.5, 0, 1))
    trend = series[-1] - series[0]
    level = series[-1]
    return float(np.clip(0.5 * level + 0.5 * np.tanh(trend), 0, 1))


def predict_persistence(batch: IoTWindowBatch) -> np.ndarray:
    """Persistence: risk from last observation (temperature-led proxy)."""
    S = len(batch.station_coords)
    risks = np.zeros(S, dtype=np.float32)
    for s in range(S):
        temp = batch.features[s, :, 0]
        hum = batch.features[s, :, 1]
        proxy = 0.6 * (temp[-1] / 40.0) + 0.4 * (1.0 - hum[-1] / 100.0)
        risks[s] = float(np.clip(proxy, 0, 1))
    return risks


def predict_ar1(batch: IoTWindowBatch, phi: float = 0.85) -> np.ndarray:
    """AR(1) on combined stress proxy per station."""
    S = len(batch.station_coords)
    risks = np.zeros(S, dtype=np.float32)
    for s in range(S):
        w = batch.features[s]
        proxy = 0.4 * (w[:, 0] / 40) + 0.3 * (1 - w[:, 1] / 100) + 0.3 * (1 - w[:, 2])
        x = proxy[0]
        for t in range(1, len(proxy)):
            x = phi * x + (1 - phi) * proxy[t]
        risks[s] = float(np.clip(x, 0, 1))
    return risks


def predict_arima(batch: IoTWindowBatch, order: tuple = (1, 0, 1)) -> np.ndarray:
    """ARIMA(p,d,q) on temperature series; falls back to AR(1) if statsmodels missing."""
    S = len(batch.station_coords)
    risks = np.zeros(S, dtype=np.float32)
    try:
        from statsmodels.tsa.arima.model import ARIMA  # type: ignore
        for s in range(S):
            temp = batch.features[s, :, 0]
            try:
                fit = ARIMA(temp, order=order).fit()
                fc = float(fit.forecast(1)[0])
                risks[s] = float(np.clip(fc / 40.0, 0, 1))
            except Exception:
                risks[s] = float(np.clip(temp[-1] / 40.0, 0, 1))
    except ImportError:
        return predict_ar1(batch)
    return risks


def evaluate_forecaster_baselines(batch: IoTWindowBatch) -> list[ForecasterBaselineResult]:
    """Compare all forecaster baselines against ground-truth risk."""
    gt = batch.ground_truth_risk
    preds = {
        "persistence": predict_persistence(batch),
        "ar1": predict_ar1(batch),
        "arima": predict_arima(batch),
    }
    out = []
    for name, p in preds.items():
        out.append(ForecasterBaselineResult(
            name=name, predictions=p,
            mae=float(np.mean(np.abs(p - gt))),
            rmse=float(np.sqrt(np.mean((p - gt) ** 2))),
        ))
    return out
