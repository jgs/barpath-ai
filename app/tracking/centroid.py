"""Centroid smoothing for bar path tracking."""

from __future__ import annotations

from dataclasses import dataclass, field

from app.detection.base import BarbellDetection
from app.utils.config import TrackingConfig


@dataclass
class BarPathTrack:
    """Tracked bar path state."""

    points: list[tuple[float, float]] = field(default_factory=list)
    filtered_points: list[tuple[float, float]] = field(default_factory=list)
    confidences: list[float] = field(default_factory=list)
    missed_frames: int = 0

    @property
    def latest(self) -> tuple[float, float] | None:
        return self.points[-1] if self.points else None


class CentroidTracker:
    """Smooth frame-by-frame detections into one continuous path."""

    def __init__(self, config: TrackingConfig) -> None:
        self.config = config
        self.track = BarPathTrack()

    def update(self, detection: BarbellDetection | None) -> BarPathTrack:
        if detection is None or detection.confidence < self.config.min_confidence:
            self.track.missed_frames += 1
            return self.track

        raw_x, raw_y = detection.center
        previous = self.track.latest

        if previous is not None:
            vertical_jump = abs(raw_y - previous[1])
            horizontal_jump = abs(raw_x - previous[0])
            if vertical_jump > self.config.max_jump_px:
                self.track.missed_frames += 1
                return self.track
            if horizontal_jump > self.config.max_horizontal_jump_px:
                raw_x = previous[0]

        if previous is None or self.track.missed_frames > self.config.max_missed_frames:
            ema_point = (raw_x, raw_y)
        else:
            alpha_y = self.config.smoothing_alpha
            alpha_x = self.config.horizontal_smoothing_alpha
            ema_point = (
                previous[0] * (1.0 - alpha_x) + raw_x * alpha_x,
                previous[1] * (1.0 - alpha_y) + raw_y * alpha_y,
            )

        smoothed = self._moving_average_point(ema_point)
        self.track.points.append(smoothed)
        self.track.filtered_points.append(ema_point)
        self.track.confidences.append(detection.confidence)
        self.track.missed_frames = 0
        return self.track

    def _moving_average_point(self, point: tuple[float, float]) -> tuple[float, float]:
        window = max(1, self.config.moving_average_window)
        recent_points = [*self.track.filtered_points, point]
        if len(recent_points) < window:
            return point

        window_points = recent_points[-window:]
        return (
            sum(p[0] for p in window_points) / window,
            sum(p[1] for p in window_points) / window,
        )
