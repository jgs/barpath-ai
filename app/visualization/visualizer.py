"""Backward-compatible visualization adapter."""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
from app.biomechanics.metrics import FrameMetrics
from app.detection.base import BarbellDetection
from app.utils.config import Config
from app.visualization.overlay import draw_analysis_overlay


class TrajectoryVisualizer:
    """Legacy wrapper around the current overlay renderer."""

    def __init__(self, config: Config):
        self.config = config
        self.trajectory_points: list[tuple[float, float]] = []

    def draw_trajectory(
        self,
        frame: np.ndarray,
        position: Optional[tuple[float, float]],
        analysis: dict[str, Any],
    ) -> np.ndarray:
        """Draw a bar path overlay using the modern renderer."""

        if position is not None:
            self.trajectory_points.append(position)
        self.trajectory_points = self.trajectory_points[-self.config.max_track_length :]

        detection = None
        if position is not None:
            x, y = int(position[0]), int(position[1])
            detection = BarbellDetection(center=position, bbox=(x - 8, y - 8, 16, 16), confidence=1.0)

        metrics = FrameMetrics(
            frame_index=len(self.trajectory_points),
            reps=int(analysis.get("rep_count", 0)),
            velocity_px_s=float(analysis.get("velocity", 0.0)),
            vertical_displacement_px=float(analysis.get("vertical_displacement", 0.0)),
            phase=str(analysis.get("phase", "LOCKOUT")),
            bar_height_px=float(analysis.get("bar_height", 0.0)),
            lockout_height_px=float(analysis.get("lockout_height", 0.0)),
            min_depth_px=float(analysis.get("min_depth", 0.0)),
            events=(),
        )
        return draw_analysis_overlay(frame, detection, self.trajectory_points, metrics)
