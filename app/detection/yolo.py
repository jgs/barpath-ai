"""YOLOv8 detector adapter."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from app.detection.base import BarbellDetection
from app.utils.config import DetectionConfig

LOGGER = logging.getLogger(__name__)


class YOLOBarbellDetector:
    """Detect barbells with YOLOv8 custom weights.

    Generic COCO YOLO models do not include a barbell class. This adapter is
    meant for custom weights trained on plates/barbells, while keeping the
    public detector contract identical to the classical detector.
    """

    def __init__(self, config: DetectionConfig) -> None:
        if config.yolo_weights is None:
            raise ValueError("YOLO weights are required for YOLOBarbellDetector.")

        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise ImportError("Install ultralytics to use YOLOBarbellDetector.") from exc

        weights = Path(config.yolo_weights)
        if not weights.exists():
            raise FileNotFoundError(f"YOLO weights not found: {weights}")

        self.config = config
        self.model = YOLO(str(weights))

    def detect(self, frame: np.ndarray) -> BarbellDetection | None:
        results = self.model.predict(
            frame,
            conf=self.config.confidence_threshold,
            verbose=False,
        )
        if not results:
            return None

        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return None

        best_index = int(boxes.conf.argmax().item())
        xyxy = boxes.xyxy[best_index].cpu().numpy()
        confidence = float(boxes.conf[best_index].item())

        x1, y1, x2, y2 = [int(v) for v in xyxy]
        w = max(1, x2 - x1)
        h = max(1, y2 - y1)
        center = (x1 + w / 2.0, y1 + h / 2.0)
        return BarbellDetection(center=center, bbox=(x1, y1, w, h), confidence=confidence)
