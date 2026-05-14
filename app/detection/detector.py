"""Backward-compatible detection adapter.

New code should import from app.detection.base, app.detection.classical or
app.detection.yolo directly.
"""

from __future__ import annotations

import numpy as np

from app.detection.base import BarbellDetection
from app.detection.classical import ClassicalBarbellDetector
from app.utils.config import Config, DetectionConfig

class BarbellDetector:
    """Legacy wrapper around the OpenCV baseline detector."""

    def __init__(self, config: Config):
        self.config = config
        self._detector = ClassicalBarbellDetector(
            DetectionConfig(confidence_threshold=config.confidence_threshold)
        )

    def detect(self, frame: np.ndarray) -> BarbellDetection | None:
        """Detect the barbell in a frame."""

        return self._detector.detect(frame)
