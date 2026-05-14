"""Template-locked detector wrapper."""

from __future__ import annotations

import cv2
import numpy as np

from app.detection.base import BarbellDetection, Detector
from app.utils.config import DetectionConfig


class TemplateLockedDetector:
    """Keep tracking the same visual target after initial acquisition.

    The classical detector is useful for acquisition, but repeatedly choosing
    the best geometric candidate can jump to rack uprights, captions or the
    floor. This wrapper uses template matching between detector refreshes so
    the target behaves like a locked object.
    """

    def __init__(self, detector: Detector, config: DetectionConfig) -> None:
        self.detector = detector
        self.config = config
        self.template: np.ndarray | None = None
        self.previous_detection: BarbellDetection | None = None
        self.output_size: tuple[int, int] | None = None
        self.frame_index = 0

    def detect(self, frame: np.ndarray) -> BarbellDetection | None:
        self.frame_index += 1
        should_refresh = self.frame_index % max(1, self.config.detector_refresh_interval) == 1

        if self.template is not None and self.previous_detection is not None and not should_refresh:
            tracked = self._match_template(frame)
            if tracked is not None and tracked.confidence >= self.config.template_min_confidence:
                self.previous_detection = tracked
                self._update_template(frame, tracked)
                return tracked

        detection = self.detector.detect(frame)
        if detection is not None:
            self.previous_detection = detection
            self._update_template(frame, detection)
            return detection

        if self.template is None or self.previous_detection is None:
            return None
        tracked = self._match_template(frame)
        if tracked is not None and tracked.confidence >= self.config.template_min_confidence:
            self.previous_detection = tracked
            self._update_template(frame, tracked)
            return tracked
        return None

    def _match_template(self, frame: np.ndarray) -> BarbellDetection | None:
        if self.template is None or self.previous_detection is None:
            return None

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        x, y, width, height = self.previous_detection.bbox
        margin = int(max(frame.shape[:2]) * self.config.template_search_margin_ratio)
        sx1 = max(0, x - margin)
        sy1 = max(0, y - margin)
        sx2 = min(frame.shape[1], x + width + margin)
        sy2 = min(frame.shape[0], y + height + margin)
        search = gray[sy1:sy2, sx1:sx2]

        template_height, template_width = self.template.shape[:2]
        if search.shape[0] < template_height or search.shape[1] < template_width:
            return None

        search = cv2.equalizeHist(search)
        result = cv2.matchTemplate(search, self.template, cv2.TM_CCOEFF_NORMED)
        _, max_value, _, max_location = cv2.minMaxLoc(result)
        nx = sx1 + max_location[0]
        ny = sy1 + max_location[1]
        center = (nx + template_width / 2.0, ny + template_height / 2.0)
        bbox_width, bbox_height = self.output_size or (template_width, template_height)
        output_x = int(center[0] - bbox_width / 2.0)
        output_y = int(center[1] - bbox_height / 2.0)
        return BarbellDetection(
            center=center,
            bbox=(output_x, output_y, bbox_width, bbox_height),
            confidence=float(max_value),
            label=self.previous_detection.label,
        )

    def _update_template(self, frame: np.ndarray, detection: BarbellDetection) -> None:
        if detection.label == "plate":
            x, y, width, height = _plate_center_bbox(
                detection,
                self.config.plate_template_radius_ratio,
                frame.shape[1],
                frame.shape[0],
            )
        else:
            x, y, width, height = _expanded_bbox(detection.bbox, frame.shape[1], frame.shape[0])
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        patch = gray[y : y + height, x : x + width]
        if patch.size == 0 or patch.shape[0] < 8 or patch.shape[1] < 8:
            return
        self.template = cv2.equalizeHist(patch)
        self.output_size = (detection.bbox[2], detection.bbox[3])
        self.previous_detection = BarbellDetection(
            center=(x + width / 2.0, y + height / 2.0),
            bbox=detection.bbox,
            confidence=detection.confidence,
            label=detection.label,
        )


def _expanded_bbox(bbox: tuple[int, int, int, int], frame_width: int, frame_height: int) -> tuple[int, int, int, int]:
    x, y, width, height = bbox
    pad_x = max(6, int(width * 0.15))
    pad_y = max(6, int(height * 0.15))
    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_y)
    x2 = min(frame_width, x + width + pad_x)
    y2 = min(frame_height, y + height + pad_y)
    return x1, y1, max(1, x2 - x1), max(1, y2 - y1)


def _plate_center_bbox(
    detection: BarbellDetection,
    radius_ratio: float,
    frame_width: int,
    frame_height: int,
) -> tuple[int, int, int, int]:
    x, y, width, height = detection.bbox
    diameter = max(20, int(min(width, height) * radius_ratio))
    cx, cy = detection.center
    x1 = max(0, int(cx - diameter / 2))
    y1 = max(0, int(cy - diameter / 2))
    x2 = min(frame_width, x1 + diameter)
    y2 = min(frame_height, y1 + diameter)
    return x1, y1, max(1, x2 - x1), max(1, y2 - y1)
