import cv2
import numpy as np

from app.detection.classical import ClassicalBarbellDetector, _suppress_overlay_markup
from app.utils.config import DetectionConfig


def test_plate_detector_prefers_side_plate_over_torso_like_circle() -> None:
    frame = np.full((720, 720, 3), 205, dtype=np.uint8)
    cv2.circle(frame, (210, 450), 170, (35, 35, 35), -1)
    cv2.circle(frame, (660, 420), 92, (20, 20, 20), -1)
    cv2.circle(frame, (660, 420), 26, (100, 100, 100), -1)

    detector = ClassicalBarbellDetector(DetectionConfig(target="right_plate"))
    detection = detector.detect(frame)

    assert detection is not None
    assert detection.label == "plate"
    assert detection.center[0] > 560


def test_plate_detector_prefers_large_high_plate_over_low_small_object() -> None:
    frame = np.full((720, 720, 3), 205, dtype=np.uint8)
    cv2.circle(frame, (635, 210), 78, (24, 24, 24), -1)
    cv2.circle(frame, (635, 210), 18, (120, 120, 120), -1)
    cv2.circle(frame, (600, 560), 35, (18, 18, 18), -1)

    detector = ClassicalBarbellDetector(DetectionConfig(target="right_plate"))
    detection = detector.detect(frame)

    assert detection is not None
    assert detection.center[1] < 300


def test_overlay_suppression_keeps_filled_colored_plate() -> None:
    frame = np.full((360, 360, 3), 80, dtype=np.uint8)
    cv2.circle(frame, (44, 180), 42, (40, 180, 40), -1)
    cv2.circle(frame, (290, 180), 48, (40, 240, 40), 3)

    cleaned = _suppress_overlay_markup(frame)

    assert int(cleaned[180, 44, 1]) > 145
    assert int(cleaned[132, 290, 1]) < 145


def test_plate_detector_uses_colored_partial_side_plate() -> None:
    frame = np.full((640, 360, 3), 90, dtype=np.uint8)
    cv2.circle(frame, (22, 285), 44, (30, 190, 30), -1)
    cv2.circle(frame, (22, 285), 25, (40, 40, 180), -1)
    cv2.circle(frame, (310, 285), 48, (40, 240, 40), 3)

    detector = ClassicalBarbellDetector(DetectionConfig(target="left_plate"))
    detection = detector.detect(frame)

    assert detection is not None
    assert detection.label == "plate"
    assert detection.center[0] < 90
    assert 240 <= detection.center[1] <= 330


def test_deadlift_plate_detector_accepts_low_side_plate() -> None:
    frame = np.full((640, 360, 3), 55, dtype=np.uint8)
    cv2.circle(frame, (296, 470), 58, (35, 35, 150), -1)
    cv2.circle(frame, (296, 470), 18, (120, 120, 120), -1)
    cv2.line(frame, (55, 470), (296, 470), (80, 80, 80), 6)

    detector = ClassicalBarbellDetector(
        DetectionConfig(
            target="plate",
            lift_type="deadlift",
            min_plate_center_y_ratio=0.10,
            max_plate_center_y_ratio=0.96,
            preferred_plate_radius_ratio=0.18,
        )
    )
    detection = detector.detect(frame)

    assert detection is not None
    assert detection.label == "plate"
    assert detection.center[0] > 230
    assert detection.center[1] > 400
