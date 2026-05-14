import cv2
import numpy as np

from app.detection.classical import ClassicalBarbellDetector
from app.utils.config import DetectionConfig


def test_classical_detector_prefers_long_horizontal_bar() -> None:
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(frame, "1003 POUNDS", (110, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255, 255, 255), 3)
    cv2.line(frame, (80, 260), (570, 265), (190, 190, 190), 8)

    detector = ClassicalBarbellDetector(DetectionConfig(enable_plate_circle_detection=False))
    detection = detector.detect(frame)

    assert detection is not None
    assert 240 <= detection.center[1] <= 285
    assert detection.bbox[2] > 250
