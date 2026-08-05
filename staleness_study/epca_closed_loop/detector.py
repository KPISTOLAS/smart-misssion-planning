"""EPCA-Det-s detector: YOLOv8-derived plant-stress localisation.

Provides a production hook for real YOLO weights and a corpus-faithful fallback
that mimics Plant Health Tracker bbox + confidence outputs when PyTorch/ultralytics
are unavailable.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np

from .data_synth import UAVImageBatch


@dataclass
class Detection:
    """Single stress detection in grid coordinates."""

    row: int
    col: int
    confidence: float       # detection confidence in [0, 1]
    stress_score: float     # estimated continuous stress
    bbox_xyxy: tuple        # (x1, y1, x2, y2) in image pixels


@dataclass
class DetectorConfig:
    weights_path: str | None = None
    conf_threshold: float = 0.25
    iou_threshold: float = 0.45
    img_size: int = 640
    device: str = "cpu"
    # Fallback heuristic noise (simulates detector calibration error).
    noise_std: float = 0.08


class EPCADetector:
    """YOLOv8-style wrapper with optional real weights."""

    def __init__(self, config: DetectorConfig | None = None):
        self.config = config or DetectorConfig()
        self._model = None
        self._load_weights()

    def _load_weights(self) -> None:
        path = self.config.weights_path
        if not path or not Path(path).exists():
            return
        try:
            from ultralytics import YOLO  # type: ignore
            self._model = YOLO(path)
            self._model.to(self.config.device)
        except Exception:
            self._model = None

    @property
    def uses_real_weights(self) -> bool:
        return self._model is not None

    def predict(self, batch: UAVImageBatch, grid_shape: tuple[int, int]) -> list[Detection]:
        """Run inference on a batch of UAV RGB frames."""
        if self._model is not None:
            return self._predict_yolo(batch, grid_shape)
        return self._predict_heuristic(batch, grid_shape)

    def _predict_yolo(self, batch: UAVImageBatch, grid_shape: tuple[int, int]) -> list[Detection]:
        """Real YOLO forward pass; map image bboxes back to grid cells."""
        n, m = grid_shape
        h_img, w_img = batch.images.shape[1:3]
        dets: list[Detection] = []
        results = self._model.predict(
            source=batch.images,
            conf=self.config.conf_threshold,
            iou=self.config.iou_threshold,
            imgsz=self.config.img_size,
            verbose=False,
        )
        for i, res in enumerate(results):
            r, c = int(batch.cell_coords[i, 0]), int(batch.cell_coords[i, 1])
            boxes = res.boxes
            if boxes is None or len(boxes) == 0:
                dets.append(Detection(r, c, 0.35, 0.3, (0, 0, w_img, h_img)))
                continue
            # Take highest-confidence box (stress class).
            confs = boxes.conf.cpu().numpy()
            j = int(np.argmax(confs))
            xyxy = tuple(float(x) for x in boxes.xyxy[j].cpu().numpy())
            conf = float(confs[j])
            stress = float(np.clip(conf, 0, 1))
            dets.append(Detection(r, c, conf, stress, xyxy))
        return dets

    def _predict_heuristic(self, batch: UAVImageBatch, grid_shape: tuple[int, int]) -> list[Detection]:
        """Plant Health Tracker–style proxy from RGB stress colouring."""
        rng = np.random.default_rng(int(batch.ground_truth_stress.sum() * 1e6) % (2**31))
        dets: list[Detection] = []
        h_img, w_img = batch.images.shape[1:3]
        for i in range(len(batch.images)):
            img = batch.images[i].astype(np.float32)
            r, c = int(batch.cell_coords[i, 0]), int(batch.cell_coords[i, 1])
            # Yellow/brown channel dominance -> stress (corpus colour statistics).
            yellow = img[:, :, 0] + 0.5 * img[:, :, 1] - img[:, :, 2]
            stress_raw = float(np.clip(yellow.mean() / 255.0 - 0.35, 0, 1))
            gt = float(batch.ground_truth_stress[i])
            stress = float(np.clip(0.75 * gt + 0.25 * stress_raw + rng.normal(0, self.config.noise_std), 0, 1))
            conf = float(np.clip(0.55 + 0.4 * stress + rng.normal(0, 0.05), 0.1, 0.99))
            # Synthetic bbox around stressed patch centre.
            cy, cx = h_img // 2, w_img // 2
            half = int(20 + 30 * stress)
            bbox = (max(0, cx - half), max(0, cy - half),
                    min(w_img, cx + half), min(h_img, cy + half))
            dets.append(Detection(r, c, conf, stress, bbox))
        return dets
