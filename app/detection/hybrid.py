"""Hybrid plate tracker using detector reacquisition and optical flow."""

from __future__ import annotations

import cv2
import numpy as np

from app.detection.base import BarbellDetection, Detector
from app.utils.config import DetectionConfig


class HybridPlateTracker:
    """Track one plate as a persistent object instead of redetecting every frame."""

    def __init__(self, detector: Detector, config: DetectionConfig) -> None:
        self.detector = detector
        self.config = config
        self.frame_index = 0
        self.track_id: int | None = None
        self.previous_gray: np.ndarray | None = None
        self.points: np.ndarray | None = None
        self.center: tuple[float, float] | None = None
        self.bbox_size: tuple[int, int] | None = None
        self.kalman = _build_kalman_filter(config)

    def detect(self, frame: np.ndarray) -> BarbellDetection | None:
        self.frame_index += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        should_refresh = self._should_refresh()

        flow_detection = self._track_with_optical_flow(gray, frame.shape[1], frame.shape[0])
        detector_detection = self.detector.detect(frame) if should_refresh or flow_detection is None else None

        chosen = self._choose_detection(flow_detection, detector_detection, frame)
        if chosen is None:
            self.previous_gray = gray
            return None

        measured_center = chosen.center
        if self.center is None:
            self._reset_kalman(measured_center)
            filtered_center = measured_center
        else:
            predicted = self.kalman.predict()
            predicted_center = (float(predicted[0, 0]), float(predicted[1, 0]))
            if _distance(measured_center, predicted_center) > _max_jump(frame.shape[1], frame.shape[0], self.config):
                measured_center = predicted_center
            measurement = np.array([[np.float32(measured_center[0])], [np.float32(measured_center[1])]])
            corrected = self.kalman.correct(measurement)
            filtered_center = (float(corrected[0, 0]), float(corrected[1, 0]))

        self.center = filtered_center
        self.bbox_size = (chosen.bbox[2], chosen.bbox[3])
        self.track_id = 1 if self.track_id is None else self.track_id
        output = _detection_at_center(chosen, filtered_center, self.bbox_size)
        self.points = _feature_points(gray, output.bbox, self.config)
        self.previous_gray = gray
        return output

    def _should_refresh(self) -> bool:
        if self.center is None:
            return True
        interval = max(1, self.config.hybrid_detector_interval)
        return self.frame_index % interval == 1

    def _track_with_optical_flow(
        self,
        gray: np.ndarray,
        frame_width: int,
        frame_height: int,
    ) -> BarbellDetection | None:
        if self.previous_gray is None or self.points is None or self.center is None or self.bbox_size is None:
            return None

        next_points, status, _ = cv2.calcOpticalFlowPyrLK(
            self.previous_gray,
            gray,
            self.points,
            None,
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03),
        )
        if next_points is None or status is None:
            return None

        valid_old = self.points[status.reshape(-1) == 1]
        valid_new = next_points[status.reshape(-1) == 1]
        if len(valid_new) < self.config.optical_flow_min_features:
            return None

        deltas = valid_new.reshape(-1, 2) - valid_old.reshape(-1, 2)
        median_delta = np.median(deltas, axis=0)
        tracked_center = (self.center[0] + float(median_delta[0]), self.center[1] + float(median_delta[1]))
        if _distance(tracked_center, self.center) > _max_jump(frame_width, frame_height, self.config):
            return None

        width, height = self.bbox_size
        x = int(round(tracked_center[0] - width / 2.0))
        y = int(round(tracked_center[1] - height / 2.0))
        return BarbellDetection(
            center=tracked_center,
            bbox=_clamp_bbox((x, y, width, height), frame_width, frame_height),
            confidence=0.72,
            label="plate",
        )

    def _choose_detection(
        self,
        flow_detection: BarbellDetection | None,
        detector_detection: BarbellDetection | None,
        frame: np.ndarray,
    ) -> BarbellDetection | None:
        normalized_detector = _normalize_plate_detection(detector_detection, frame, self.config)
        if flow_detection is None:
            return normalized_detector
        if normalized_detector is None or self.center is None:
            return flow_detection

        max_jump = _max_jump(frame.shape[1], frame.shape[0], self.config)
        if _distance(normalized_detector.center, flow_detection.center) <= max_jump * 0.75:
            fused_center = (
                flow_detection.center[0] * 0.72 + normalized_detector.center[0] * 0.28,
                flow_detection.center[1] * 0.72 + normalized_detector.center[1] * 0.28,
            )
            fused_confidence = max(flow_detection.confidence, normalized_detector.confidence)
            return _detection_at_center(flow_detection, fused_center, (flow_detection.bbox[2], flow_detection.bbox[3]), fused_confidence)

        return flow_detection

    def _reset_kalman(self, center: tuple[float, float]) -> None:
        self.kalman.statePost = np.array([[center[0]], [center[1]], [0.0], [0.0]], dtype=np.float32)


def _build_kalman_filter(config: DetectionConfig) -> cv2.KalmanFilter:
    kalman = cv2.KalmanFilter(4, 2)
    kalman.transitionMatrix = np.array(
        [[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]],
        dtype=np.float32,
    )
    kalman.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32)
    kalman.processNoiseCov = np.eye(4, dtype=np.float32) * np.float32(config.kalman_process_noise)
    kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * np.float32(config.kalman_measurement_noise)
    kalman.errorCovPost = np.eye(4, dtype=np.float32)
    return kalman


def _normalize_plate_detection(
    detection: BarbellDetection | None,
    frame: np.ndarray,
    config: DetectionConfig,
) -> BarbellDetection | None:
    if detection is None:
        return None

    x, y, width, height = detection.bbox
    bbox_center = (x + width / 2.0, y + height / 2.0)
    circle_center = _hough_plate_center(frame, detection.bbox, config)
    center = circle_center or bbox_center
    return BarbellDetection(center=center, bbox=detection.bbox, confidence=detection.confidence, label="plate")


def _hough_plate_center(
    frame: np.ndarray,
    bbox: tuple[int, int, int, int],
    config: DetectionConfig,
) -> tuple[float, float] | None:
    x, y, width, height = bbox
    if width < 24 or height < 24:
        return None

    pad = max(8, int(min(width, height) * 0.18))
    x1 = max(0, x - pad)
    y1 = max(0, y - pad)
    x2 = min(frame.shape[1], x + width + pad)
    y2 = min(frame.shape[0], y + height + pad)
    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return None

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)
    min_radius = max(8, int(min(width, height) * 0.28))
    max_radius = max(min_radius + 2, int(max(width, height) * 0.7))
    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=max(16, min(width, height) // 2),
        param1=80,
        param2=24,
        minRadius=min_radius,
        maxRadius=max_radius,
    )
    if circles is None:
        return None

    circles = np.round(circles[0]).astype(int)
    bbox_center = np.array([x + width / 2.0, y + height / 2.0])
    best = min(circles, key=lambda circle: float(np.linalg.norm(np.array([x1 + circle[0], y1 + circle[1]]) - bbox_center)))
    return (float(x1 + best[0]), float(y1 + best[1]))


def _feature_points(gray: np.ndarray, bbox: tuple[int, int, int, int], config: DetectionConfig) -> np.ndarray:
    x, y, width, height = bbox
    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(gray.shape[1], x + width)
    y2 = min(gray.shape[0], y + height)
    roi = gray[y1:y2, x1:x2]
    if roi.size == 0:
        return np.empty((0, 1, 2), dtype=np.float32)

    features = cv2.goodFeaturesToTrack(
        roi,
        maxCorners=max(12, config.optical_flow_feature_count),
        qualityLevel=0.01,
        minDistance=5,
        blockSize=5,
    )
    if features is None:
        return _grid_points(x1, y1, x2, y2, config)

    features[:, :, 0] += x1
    features[:, :, 1] += y1
    return features.astype(np.float32)


def _grid_points(x1: int, y1: int, x2: int, y2: int, config: DetectionConfig) -> np.ndarray:
    min_points = max(4, config.optical_flow_min_features)
    columns = max(2, int(np.ceil(np.sqrt(min_points))))
    rows = max(2, int(np.ceil(min_points / columns)))
    xs = np.linspace(x1 + 4, max(x1 + 4, x2 - 4), columns, dtype=np.float32)
    ys = np.linspace(y1 + 4, max(y1 + 4, y2 - 4), rows, dtype=np.float32)
    points = [[[float(x), float(y)]] for y in ys for x in xs]
    return np.array(points[:min_points], dtype=np.float32)


def _detection_at_center(
    detection: BarbellDetection,
    center: tuple[float, float],
    size: tuple[int, int],
    confidence: float | None = None,
) -> BarbellDetection:
    width, height = size
    x = int(round(center[0] - width / 2.0))
    y = int(round(center[1] - height / 2.0))
    return BarbellDetection(
        center=center,
        bbox=(x, y, width, height),
        confidence=detection.confidence if confidence is None else confidence,
        label="plate",
    )


def _clamp_bbox(bbox: tuple[int, int, int, int], frame_width: int, frame_height: int) -> tuple[int, int, int, int]:
    x, y, width, height = bbox
    x = max(0, min(frame_width - 1, x))
    y = max(0, min(frame_height - 1, y))
    return x, y, max(1, min(width, frame_width - x)), max(1, min(height, frame_height - y))


def _max_jump(frame_width: int, frame_height: int, config: DetectionConfig) -> float:
    return float(np.hypot(frame_width, frame_height) * config.hybrid_max_jump_ratio)


def _distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))
