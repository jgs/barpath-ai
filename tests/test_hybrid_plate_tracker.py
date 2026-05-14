import cv2
import numpy as np

from app.detection.base import BarbellDetection
from app.detection.hybrid import HybridPlateTracker
from app.utils.config import DetectionConfig


class _CountingDetector:
    def __init__(self, detections: list[BarbellDetection | None]) -> None:
        self.detections = detections
        self.calls = 0

    def detect(self, frame: np.ndarray) -> BarbellDetection | None:
        detection = self.detections[min(self.calls, len(self.detections) - 1)]
        self.calls += 1
        return detection


def test_hybrid_plate_tracker_uses_optical_flow_between_detector_refreshes() -> None:
    frames = [_plate_frame((80 + index * 4, 120)) for index in range(8)]
    detector = _CountingDetector(
        [BarbellDetection(center=(80.0, 120.0), bbox=(52, 92, 56, 56), confidence=0.9, label="plate")]
    )
    tracker = HybridPlateTracker(
        detector,
        DetectionConfig(
            target="plate",
            hybrid_detector_interval=99,
            optical_flow_min_features=3,
            kalman_measurement_noise=0.05,
        ),
    )

    detections = [tracker.detect(frame) for frame in frames]

    assert detector.calls == 1
    assert detections[-1] is not None
    assert detections[-1].center[0] > detections[0].center[0] + 12
    assert abs(detections[-1].center[1] - 120.0) < 3.0
    assert tracker.track_id == 1


def test_hybrid_plate_tracker_rejects_far_detector_reacquisition_jump() -> None:
    frames = [_plate_frame((90 + index * 2, 140)) for index in range(14)]
    detector = _CountingDetector(
        [
            BarbellDetection(center=(90.0, 140.0), bbox=(62, 112, 56, 56), confidence=0.9, label="plate"),
            BarbellDetection(center=(260.0, 300.0), bbox=(232, 272, 56, 56), confidence=0.95, label="plate"),
        ]
    )
    tracker = HybridPlateTracker(
        detector,
        DetectionConfig(
            target="plate",
            hybrid_detector_interval=12,
            optical_flow_min_features=3,
            hybrid_max_jump_ratio=0.08,
            kalman_measurement_noise=0.05,
        ),
    )

    detections = [tracker.detect(frame) for frame in frames]

    assert detector.calls == 2
    assert detections[-1] is not None
    assert detections[-1].center[0] < 130
    assert detections[-1].center[1] < 155


def _plate_frame(center: tuple[int, int]) -> np.ndarray:
    frame = np.full((360, 360, 3), 35, dtype=np.uint8)
    cv2.circle(frame, center, 28, (70, 180, 70), -1)
    cv2.circle(frame, center, 10, (190, 190, 190), -1)
    cv2.line(frame, (center[0] - 22, center[1]), (center[0] + 22, center[1]), (20, 80, 20), 2)
    cv2.line(frame, (center[0], center[1] - 22), (center[0], center[1] + 22), (20, 80, 20), 2)
    return frame
