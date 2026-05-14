"""OpenCV-based barbell detector for the V1 MVP.

This detector intentionally favors a practical, dependency-light baseline:
find bright, circular plate-like blobs and infer the barbell center from the
strongest candidate. Custom YOLO weights can replace it without changing the
rest of the pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import atan2, degrees

import cv2
import numpy as np

from app.detection.base import BarbellDetection
from app.utils.config import DetectionConfig


class ClassicalBarbellDetector:
    """Detect barbell plates using contour geometry and contrast."""

    def __init__(self, config: DetectionConfig) -> None:
        self.config = config
        self.previous_center: tuple[float, float] | None = None
        self.previous_detection: BarbellDetection | None = None
        self.frame_index = 0

    def detect(self, frame: np.ndarray) -> BarbellDetection | None:
        self.frame_index += 1
        detection_frame = _suppress_overlay_markup(frame) if self.config.suppress_overlay_markup else frame
        candidates: list[_Candidate] = []
        if self.config.target in {"auto", "bar"}:
            candidates.extend(self._line_candidates(detection_frame))
        if self.config.target in {"auto", "plate", "right_plate", "left_plate"}:
            candidates.extend(self._colored_plate_candidates(detection_frame))
            candidates.extend(self._plate_candidates(detection_frame))
            candidates.extend(self._contour_candidates(detection_frame))
        candidates = _filter_candidates_for_target(candidates, self.config.target, frame.shape[1])
        if self.previous_detection is not None and _target_accepts_label(
            self.config.target,
            self.previous_detection.label,
        ):
            candidates.append(_Candidate(detection=self.previous_detection, base_score=2.35))

        if not candidates:
            return None

        height, width = frame.shape[:2]
        diagonal = float(np.hypot(width, height))
        best = max(candidates, key=lambda candidate: self._score(candidate, diagonal, width))
        self.previous_center = best.detection.center
        self.previous_detection = best.detection
        return best.detection

    def _line_candidates(self, frame: np.ndarray) -> list["_Candidate"]:
        height, width = frame.shape[:2]
        y1 = int(height * self.config.roi_top_ratio)
        y2 = int(height * self.config.roi_bottom_ratio)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (7, 7), 0)
        roi_gray = gray[y1:y2]
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        if self.config.equalize_histogram:
            roi_gray = cv2.equalizeHist(roi_gray)

        edges = cv2.Canny(
            roi_gray,
            threshold1=self.config.canny_low,
            threshold2=self.config.canny_high,
        )

        min_length = int(width * self.config.min_bar_line_length_ratio)
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=45,
            minLineLength=min_length,
            maxLineGap=32,
        )
        if lines is None:
            return []

        candidates: list[_Candidate] = []
        segments: list[tuple[int, int, int, int]] = []
        for line in lines[:, 0]:
            x_start, local_y_start, x_end, local_y_end = [int(value) for value in line]
            y_start = local_y_start + y1
            y_end = local_y_end + y1
            if (y_start + y_end) / 2.0 > height * self.config.max_bar_line_y_ratio:
                continue
            dx = x_end - x_start
            dy = y_end - y_start
            length = float(np.hypot(dx, dy))
            if length < min_length:
                continue

            angle = abs(degrees(atan2(dy, dx)))
            angle = min(angle, abs(180.0 - angle))
            if angle > self.config.max_bar_angle_degrees:
                continue

            segments.append((x_start, y_start, x_end, y_end))
            center = ((x_start + x_end) / 2.0, (y_start + y_end) / 2.0)
            bar_height = max(8, int(height * 0.012))
            x = min(x_start, x_end)
            y = int(center[1] - bar_height / 2)
            bbox = (x, y, max(1, abs(dx)), bar_height)
            confidence = float(np.clip(length / width, 0.2, 0.98))
            saturation = _mean_saturation(hsv, bbox)
            if saturation > self.config.max_colored_line_saturation:
                continue
            neutrality_bonus = 1.0 - saturation
            candidates.append(
                _Candidate(
                    detection=BarbellDetection(center=center, bbox=bbox, confidence=confidence),
                    base_score=1.2 + length / width + neutrality_bonus - saturation,
                )
            )

        candidates.extend(
            _group_line_segments(
                segments,
                hsv,
                frame_width=width,
                frame_height=height,
                max_saturation=self.config.max_colored_line_saturation,
            )
        )
        return candidates

    def _colored_plate_candidates(self, frame: np.ndarray) -> list["_Candidate"]:
        height, width = frame.shape[:2]
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        saturation = hsv[:, :, 1]
        value = hsv[:, :, 2]
        mask = cv2.inRange(saturation, 58, 255)
        mask = cv2.bitwise_and(mask, cv2.inRange(value, 45, 255))

        y1, y2 = self._plate_search_bounds(height)
        roi_mask = np.zeros_like(mask)
        roi_mask[y1:y2, :] = mask[y1:y2, :]

        kernel = np.ones((5, 5), dtype=np.uint8)
        roi_mask = cv2.morphologyEx(roi_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        roi_mask = cv2.morphologyEx(roi_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        contours, _ = cv2.findContours(roi_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        frame_area = float(height * width)
        min_radius = width * self.config.min_plate_radius_ratio * 0.7
        max_radius = width * self.config.max_plate_radius_ratio
        candidates: list[_Candidate] = []
        for contour in contours:
            area = float(cv2.contourArea(contour))
            if area < max(80.0, self.config.min_plate_area * 0.45):
                continue
            if area > frame_area * self.config.max_plate_area_ratio:
                continue

            x, y, w, h = cv2.boundingRect(contour)
            if h <= 0 or w <= 0:
                continue
            aspect = w / h
            if not 0.28 <= aspect <= 3.2:
                continue

            (cx, cy), radius = cv2.minEnclosingCircle(contour)
            if radius < min_radius or radius > max_radius:
                continue
            if cy < height * self.config.min_plate_center_y_ratio:
                continue
            if cy > height * self.config.max_plate_center_y_ratio:
                continue

            fill_ratio = area / max(float(w * h), 1.0)
            if fill_ratio < 0.16:
                continue

            perimeter = cv2.arcLength(contour, True)
            circularity = 0.0 if perimeter <= 0 else float(4.0 * np.pi * area / (perimeter * perimeter))
            if circularity < 0.14 and min(w, h) < radius * 0.75:
                continue

            bbox = (x, y, w, h)
            mean_saturation = _mean_saturation(hsv, bbox)
            side_bonus = abs((cx / max(width, 1)) - 0.5) * 1.4
            size_score = min(1.0, area / max(frame_area * 0.018, 1.0))
            vertical_bonus = _plate_vertical_bonus(float(cy), height, self.config)
            lower_frame_penalty = _lower_frame_penalty(cy, height, self.config.lift_type)
            confidence = float(np.clip(0.28 + size_score * 0.35 + mean_saturation * 0.22, 0.15, 0.96))
            candidates.append(
                _Candidate(
                    detection=BarbellDetection(
                        center=(float(cx), float(cy)),
                        bbox=bbox,
                        confidence=confidence,
                        label="plate",
                    ),
                    base_score=2.65
                    + size_score * 0.65
                    + fill_ratio * 0.45
                    + min(circularity, 1.0) * 0.25
                    + mean_saturation * 0.45
                    + side_bonus
                    + vertical_bonus * 1.05
                    - lower_frame_penalty * 0.9,
                )
            )

        return candidates

    def _plate_candidates(self, frame: np.ndarray) -> list["_Candidate"]:
        if not self.config.enable_plate_circle_detection:
            return []
        interval = max(1, self.config.plate_detection_interval)
        if self.previous_detection is not None and self.frame_index % interval != 0:
            return []

        height, width = frame.shape[:2]
        min_radius = int(width * self.config.min_plate_radius_ratio)
        max_radius = int(width * self.config.max_plate_radius_ratio)
        if min_radius <= 0 or max_radius <= min_radius:
            return []

        search_y1, search_y2 = self._plate_search_bounds(height)
        search_frame = frame[search_y1:search_y2]
        if search_frame.size == 0:
            return []

        scale = 1.0
        if width > self.config.circle_detection_width:
            scale = self.config.circle_detection_width / float(width)
            resized_width = self.config.circle_detection_width
            resized_height = max(1, int(search_frame.shape[0] * scale))
            search_frame = cv2.resize(
                search_frame,
                (resized_width, resized_height),
                interpolation=cv2.INTER_AREA,
            )

        gray = cv2.cvtColor(search_frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.medianBlur(gray, 5)
        circles = cv2.HoughCircles(
            gray,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=max(30, int(width * scale * 0.12)),
            param1=80,
            param2=32,
            minRadius=max(8, int(min_radius * scale)),
            maxRadius=max(10, int(max_radius * scale)),
        )
        if circles is None:
            return []

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        candidates: list[_Candidate] = []
        full_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        for x_center, y_center, radius in np.round(circles[0]).astype(int):
            x_center = int(round(x_center / scale))
            y_center = int(round(y_center / scale)) + search_y1
            radius = int(round(radius / scale))
            if y_center < int(height * self.config.min_plate_center_y_ratio):
                continue
            if y_center > int(height * self.config.max_plate_center_y_ratio):
                continue

            x = max(0, x_center - radius)
            y = max(0, y_center - radius)
            box_size = int(radius * 2)
            bbox = (x, y, min(width - x, box_size), min(height - y, box_size))
            if bbox[2] <= 0 or bbox[3] <= 0:
                continue

            roi_gray = full_gray[y : y + bbox[3], x : x + bbox[2]]
            mean_intensity = float(np.mean(roi_gray) / 255.0)
            saturation = _mean_saturation(hsv, bbox)
            darkness = 1.0 - mean_intensity
            side_bonus = abs((x_center / max(width, 1)) - 0.5) * 1.55
            radius_score = min(1.0, radius / max(max_radius, 1))
            preferred_radius = max(1.0, width * self.config.preferred_plate_radius_ratio)
            radius_match = 1.0 - min(1.0, abs(radius - preferred_radius) / preferred_radius)
            vertical_bonus = _plate_vertical_bonus(float(y_center), height, self.config)
            lower_frame_penalty = _lower_frame_penalty(float(y_center), height, self.config.lift_type)
            confidence = float(np.clip(0.30 + radius_score * 0.35 + darkness * 0.25, 0.15, 0.98))
            base_score = (
                2.25
                + radius_match * 0.9
                + radius_score * 0.2
                + darkness * 0.35
                + side_bonus
                + vertical_bonus * 1.05
                - saturation * 0.25
                - lower_frame_penalty * 0.9
            )
            candidates.append(
                _Candidate(
                    detection=BarbellDetection(
                        center=(float(x_center), float(y_center)),
                        bbox=bbox,
                        confidence=confidence,
                        label="plate",
                    ),
                    base_score=base_score,
                )
            )

        return candidates

    def _plate_search_bounds(self, height: int) -> tuple[int, int]:
        y1 = int(height * self.config.min_plate_center_y_ratio)
        y2 = int(height * self.config.max_plate_center_y_ratio)
        if self.previous_center is None:
            return y1, y2

        _, previous_y = self.previous_center
        margin = int(height * self.config.temporal_search_margin_ratio)
        return max(y1, int(previous_y) - margin), min(y2, int(previous_y) + margin)

    def _contour_candidates(self, frame: np.ndarray) -> list["_Candidate"]:
        height, width = frame.shape[:2]
        y1 = int(height * self.config.roi_top_ratio)
        y2 = int(height * self.config.roi_bottom_ratio)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (7, 7), 0)

        if self.config.equalize_histogram:
            gray = cv2.equalizeHist(gray)

        edges = cv2.Canny(
            gray[y1:y2],
            threshold1=self.config.canny_low,
            threshold2=self.config.canny_high,
        )
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        candidates: list[_Candidate] = []
        frame_area = float(height * width)

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self.config.min_plate_area:
                continue
            if area > frame_area * self.config.max_plate_area_ratio:
                continue

            x, y, w, h = cv2.boundingRect(contour)
            y += y1
            aspect = w / max(h, 1)
            if not self.config.min_aspect_ratio <= aspect <= self.config.max_aspect_ratio:
                continue

            perimeter = cv2.arcLength(contour, True)
            if perimeter <= 0:
                continue
            circularity = 4.0 * np.pi * area / (perimeter * perimeter)
            if circularity < self.config.min_circularity:
                continue

            mask = np.zeros_like(gray)
            shifted = contour.copy()
            shifted[:, :, 1] += y1
            cv2.drawContours(mask, [shifted], -1, 255, thickness=-1)
            mean_intensity = float(cv2.mean(gray, mask=mask)[0])
            score = (area / frame_area) * 4.0 + circularity + mean_intensity / 255.0
            center = (x + w / 2.0, y + h / 2.0)
            confidence = float(np.clip(score / 2.5, 0.05, 0.99))

            candidates.append(
                _Candidate(
                    detection=BarbellDetection(
                        center=center,
                        bbox=(x, y, w, h),
                        confidence=confidence,
                        label="plate",
                    ),
                    base_score=score,
                )
            )
        return candidates

    def _score(self, candidate: "_Candidate", diagonal: float, frame_width: int) -> float:
        score = candidate.base_score + _target_bonus(
            candidate.detection,
            self.config.target,
            frame_width,
        )
        if self.previous_center is None:
            return score

        cx, cy = candidate.detection.center
        px, py = self.previous_center
        normalized_distance = np.hypot(cx - px, cy - py) / max(diagonal, 1.0)
        temporal_bonus = max(0.0, 1.0 - normalized_distance) * self.config.temporal_prior_weight
        return score + temporal_bonus


@dataclass(frozen=True)
class _Candidate:
    detection: BarbellDetection
    base_score: float


def _mean_saturation(hsv_frame: np.ndarray, bbox: tuple[int, int, int, int]) -> float:
    x, y, width, height = bbox
    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(hsv_frame.shape[1], x + width)
    y2 = min(hsv_frame.shape[0], y + height)
    if x2 <= x1 or y2 <= y1:
        return 1.0
    return float(np.mean(hsv_frame[y1:y2, x1:x2, 1]) / 255.0)


def _suppress_overlay_markup(frame: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    green_yellow = cv2.inRange(hsv, np.array([24, 70, 95]), np.array([100, 255, 255]))
    cyan = cv2.inRange(hsv, np.array([80, 55, 95]), np.array([105, 255, 255]))
    raw_mask = cv2.bitwise_or(green_yellow, cyan)
    if int(cv2.countNonZero(raw_mask)) == 0:
        return frame

    stroke_kernel = np.ones((5, 5), dtype=np.uint8)
    eroded = cv2.erode(raw_mask, stroke_kernel, iterations=1)
    thin_strokes = cv2.bitwise_and(raw_mask, cv2.bitwise_not(eroded))

    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(raw_mask, connectivity=8)
    mask = np.zeros_like(raw_mask)
    frame_area = float(frame.shape[0] * frame.shape[1])
    for component_id in range(1, component_count):
        x = int(stats[component_id, cv2.CC_STAT_LEFT])
        y = int(stats[component_id, cv2.CC_STAT_TOP])
        width = int(stats[component_id, cv2.CC_STAT_WIDTH])
        height = int(stats[component_id, cv2.CC_STAT_HEIGHT])
        area = int(stats[component_id, cv2.CC_STAT_AREA])
        fill_ratio = area / max(float(width * height), 1.0)
        relative_area = area / max(frame_area, 1.0)
        looks_like_markup = area < 220 or fill_ratio < 0.22 or relative_area < 0.0012
        if looks_like_markup:
            mask[labels == component_id] = 255

    mask = cv2.bitwise_or(mask, thin_strokes)
    if int(cv2.countNonZero(mask)) == 0:
        return frame

    kernel = np.ones((3, 3), dtype=np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=1)
    return cv2.inpaint(frame, mask, 3, cv2.INPAINT_TELEA)


def _target_bonus(detection: BarbellDetection, target: str, frame_width: int) -> float:
    if target == "bar":
        return 0.9 if detection.label == "barbell" else -0.35
    if target == "right_plate":
        if detection.label != "plate":
            return -0.45
        return 1.0 if detection.center[0] >= frame_width * 0.5 else -0.6
    if target == "left_plate":
        if detection.label != "plate":
            return -0.45
        return 1.0 if detection.center[0] < frame_width * 0.5 else -0.6
    if target == "plate":
        if detection.label != "plate":
            return -0.5
        side_distance = abs((detection.center[0] / max(frame_width, 1)) - 0.5)
        return 0.35 + side_distance * 1.4
    return 0.0


def _lower_frame_penalty(y_center: float, frame_height: int, lift_type: str) -> float:
    if lift_type.strip().lower() == "deadlift":
        return 0.0
    return max(0.0, (y_center / max(frame_height, 1) - 0.55) / 0.22)


def _plate_vertical_bonus(y_center: float, frame_height: int, config: DetectionConfig) -> float:
    if config.lift_type.strip().lower() == "deadlift":
        return min(1.0, y_center / max(frame_height * config.max_plate_center_y_ratio, 1))
    return 1.0 - min(1.0, y_center / max(frame_height * config.max_plate_center_y_ratio, 1))


def _target_accepts_label(target: str, label: str) -> bool:
    if target == "bar":
        return label == "barbell"
    if target in {"plate", "right_plate", "left_plate"}:
        return label == "plate"
    return True


def _filter_candidates_for_target(
    candidates: list[_Candidate],
    target: str,
    frame_width: int,
) -> list[_Candidate]:
    if target == "bar":
        return [candidate for candidate in candidates if candidate.detection.label == "barbell"]
    if target == "right_plate":
        return [
            candidate
            for candidate in candidates
            if candidate.detection.label == "plate" and candidate.detection.center[0] >= frame_width * 0.5
        ]
    if target == "left_plate":
        return [
            candidate
            for candidate in candidates
            if candidate.detection.label == "plate" and candidate.detection.center[0] < frame_width * 0.5
        ]
    if target == "plate":
        return [
            candidate
            for candidate in candidates
            if candidate.detection.label == "plate"
            and (
                candidate.detection.center[0] <= frame_width * 0.40
                or candidate.detection.center[0] >= frame_width * 0.60
            )
        ]
    return candidates


def _group_line_segments(
    segments: list[tuple[int, int, int, int]],
    hsv_frame: np.ndarray,
    frame_width: int,
    frame_height: int,
    max_saturation: float,
) -> list[_Candidate]:
    buckets: dict[int, list[tuple[int, int, int, int]]] = {}
    bucket_size = max(18, int(frame_height * 0.028))
    for segment in segments:
        _, y_start, _, y_end = segment
        y_center = int((y_start + y_end) / 2)
        buckets.setdefault(y_center // bucket_size, []).append(segment)

    candidates: list[_Candidate] = []
    bar_height = max(8, int(frame_height * 0.012))
    for grouped_segments in buckets.values():
        if len(grouped_segments) < 2:
            continue

        xs = [value for x_start, _, x_end, _ in grouped_segments for value in (x_start, x_end)]
        ys = [value for _, y_start, _, y_end in grouped_segments for value in (y_start, y_end)]
        x = max(0, min(xs))
        width = min(frame_width - x, max(xs) - x)
        if width < frame_width * 0.28:
            continue

        y_center = float(np.mean(ys))
        y = int(y_center - bar_height / 2)
        bbox = (x, y, max(1, width), bar_height)
        saturation = _mean_saturation(hsv_frame, bbox)
        if saturation > max_saturation:
            continue
        neutrality_bonus = 1.0 - saturation
        center = (float(np.mean(xs)), y_center)
        confidence = float(np.clip(width / frame_width, 0.25, 0.99))
        candidates.append(
            _Candidate(
                detection=BarbellDetection(center=center, bbox=bbox, confidence=confidence),
                base_score=1.75 + width / frame_width + neutrality_bonus - saturation,
            )
        )

    return candidates
