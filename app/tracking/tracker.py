"""Backward-compatible tracker adapter."""

from __future__ import annotations

from typing import Optional

from app.detection.base import BarbellDetection
from app.tracking.centroid import CentroidTracker
from app.utils.config import Config, TrackingConfig

class BarbellTracker:
    """Legacy wrapper around CentroidTracker."""

    def __init__(self, config: Config):
        self.config = config
        self._tracker = CentroidTracker(TrackingConfig())
        self.previous_position: Optional[tuple[float, float]] = None
        self.track_history: list[tuple[float, float]] = []

    def track(self, detection: BarbellDetection | None) -> Optional[tuple[float, float]]:
        """Track a detection and return the latest smoothed center."""

        track = self._tracker.update(detection)
        self.previous_position = track.latest
        self.track_history = track.points[-self.config.max_track_length :]
        return self.previous_position
