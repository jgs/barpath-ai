import cv2
import numpy as np

from app.detection.classical import ClassicalBarbellDetector
from app.utils.config import DetectionConfig


def test_classical_detector_can_use_plate_center_when_bar_is_occluded() -> None:
    frame = np.full((720, 720, 3), 210, dtype=np.uint8)
    cv2.rectangle(frame, (70, 560), (650, 590), (45, 45, 45), -1)
    cv2.circle(frame, (540, 310), 82, (25, 25, 25), -1)
    cv2.circle(frame, (540, 310), 24, (105, 105, 105), -1)

    detector = ClassicalBarbellDetector(DetectionConfig(target="right_plate"))
    detection = detector.detect(frame)

    assert detection is not None
    assert detection.label == "plate"
    assert 285 <= detection.center[1] <= 335
