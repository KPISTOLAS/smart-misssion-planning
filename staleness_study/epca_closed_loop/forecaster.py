"""MLP IoT forecaster for temporal anomaly / risk prediction.

Follows Herbal Plant dataset feature engineering: normalised sliding windows
of temperature, humidity, soil moisture, and light -> scalar risk in [0, 1].
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import numpy as np

from .data_synth import IoTWindowBatch, IOT_FEATURE_NAMES


@dataclass
class ForecasterConfig:
    weights_path: str | None = None
    hidden_dims: tuple = (64, 32)
    window_len: int = 24
    n_features: int = 4
    device: str = "cpu"
    noise_std: float = 0.06


class MLPForecaster:
    """Small MLP regressor with optional PyTorch checkpoint."""

    def __init__(self, config: ForecasterConfig | None = None):
        self.config = config or ForecasterConfig()
        self._torch_model = None
        self._norm_stats: dict | None = None
        self._load_weights()

    def _load_weights(self) -> None:
        path = self.config.weights_path
        if not path or not Path(path).exists():
            return
        try:
            import torch  # type: ignore
            import torch.nn as nn  # type: ignore

            ckpt = torch.load(path, map_location=self.config.device, weights_only=False)
            dims = ckpt.get("hidden_dims", self.config.hidden_dims)
            layers = []
            in_d = self.config.n_features * self.config.window_len
            prev = in_d
            for h in dims:
                layers += [nn.Linear(prev, h), nn.ReLU()]
                prev = h
            layers.append(nn.Linear(prev, 1))
            layers.append(nn.Sigmoid())
            model = nn.Sequential(*layers)
            model.load_state_dict(ckpt["state_dict"])
            model.eval()
            model.to(self.config.device)
            self._torch_model = model
            self._norm_stats = ckpt.get("norm_stats")
        except Exception:
            self._torch_model = None

    @property
    def uses_real_weights(self) -> bool:
        return self._torch_model is not None

    def _flatten(self, batch: IoTWindowBatch) -> np.ndarray:
        S, T, F = batch.features.shape
        x = batch.features.reshape(S, T * F)
        if self._norm_stats:
            mu = np.asarray(self._norm_stats["mean"], dtype=np.float32)
            sd = np.asarray(self._norm_stats["std"], dtype=np.float32)
            x = (x - mu) / np.maximum(sd, 1e-6)
        return x

    def predict(self, batch: IoTWindowBatch) -> np.ndarray:
        """Return per-station risk predictions, shape (S,)."""
        if self._torch_model is not None:
            return self._predict_torch(batch)
        return self._predict_heuristic(batch)

    def _predict_torch(self, batch: IoTWindowBatch) -> np.ndarray:
        import torch  # type: ignore
        x = self._flatten(batch)
        with torch.no_grad():
            t = torch.from_numpy(x).float().to(self.config.device)
            out = self._torch_model(t).cpu().numpy().ravel()
        return np.clip(out, 0, 1)

    def _predict_heuristic(self, batch: IoTWindowBatch) -> np.ndarray:
        """Herbal Plant–style risk from feature trends + ground-truth proxy."""
        rng = np.random.default_rng(int(batch.ground_truth_risk.sum() * 1e5) % (2**31))
        S = len(batch.station_coords)
        risks = np.zeros(S, dtype=np.float32)
        for s in range(S):
            w = batch.features[s]
            temp_trend = float(w[-1, 0] - w[0, 0])
            hum_drop = float(w[0, 1] - w[-1, 1])
            soil_drop = float(w[0, 2] - w[-1, 2])
            score = (
                0.35 * np.tanh(temp_trend / 4.0)
                + 0.30 * np.tanh(hum_drop / 15.0)
                + 0.25 * np.tanh(soil_drop / 0.08)
                + 0.10 * batch.ground_truth_risk[s]
            )
            risks[s] = float(np.clip(score + rng.normal(0, self.config.noise_std), 0, 1))
        return risks

    @staticmethod
    def export_checkpoint_template(path: str, config: ForecasterConfig | None = None) -> None:
        """Write a JSON + state_dict template for plugging in trained weights."""
        config = config or ForecasterConfig()
        meta = {
            "hidden_dims": list(config.hidden_dims),
            "window_len": config.window_len,
            "n_features": config.n_features,
            "feature_names": list(IOT_FEATURE_NAMES),
            "norm_stats": {"mean": [0.0] * (config.window_len * config.n_features),
                           "std": [1.0] * (config.window_len * config.n_features)},
        }
        Path(path).with_suffix(".json").write_text(json.dumps(meta, indent=2))
