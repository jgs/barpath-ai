import cv2
import numpy as np

from app.detection.base import BarbellDetection
from app.detection.template import TemplateLockedDetector
from app.utils.config import DetectionConfig


class OneShotDetector:
    def __init__(self) -> None:
        self.calls = 0

    def detect(self, frame: np.ndarray) -> BarbellDetection | None:
        self.calls += 1
        if self.calls > 1:
            return None
        return BarbellDetection(center=(70.0, 70.0), bbox=(50, 50, 40, 40), confidence=0.9)


def test_template_locked_detector_tracks_same_patch_without_redetection() -> None:
    first = np.zeros((180, 220, 3), dtype=np.uint8)
    second = np.zeros_like(first)
    cv2.circle(first, (70, 70), 18, (220, 220, 220), -1)
    cv2.circle(second, (80, 92), 18, (220, 220, 220), -1)

    detector = TemplateLockedDetector(
        OneShotDetector(),
        DetectionConfig(detector_refresh_interval=99, template_min_confidence=0.35),
    )

    detector.detect(first)
    tracked = detector.detect(second)

    assert tracked is not None
    assert 70 <= tracked.center[0] <= 90
    assert 82 <= tracked.center[1] <= 102


def test_template_locked_detector_preserves_output_bbox_size() -> None:
    first = np.zeros((220, 260, 3), dtype=np.uint8)
    second = np.zeros_like(first)
    cv2.circle(first, (100, 100), 26, (220, 220, 220), -1)
    cv2.circle(second, (112, 110), 26, (220, 220, 220), -1)

    detector = TemplateLockedDetector(
        OneShotDetector(),
        DetectionConfig(detector_refresh_interval=99, template_min_confidence=0.35),
    )

    detector.detect(first)
    tracked = detector.detect(second)

    assert tracked is not None
    assert tracked.bbox[2:] == (40, 40)
