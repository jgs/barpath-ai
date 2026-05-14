"""Backward-compatible biomechanics analyzer."""

from __future__ import annotations

from typing import Any, Optional

from app.biomechanics.metrics import analyze_path
from app.utils.config import Config

class BiomechanicsAnalyzer:
    """Legacy stateful wrapper around analyze_path."""

    def __init__(self, config: Config):
        self.config = config
        self.positions: list[tuple[float, float]] = []
        self.rep_count: int = 0

    def analyze(self, position: Optional[tuple[float, float]]) -> dict[str, Any]:
        """Analyze the latest tracked position."""

        if position is None:
            return {}

        self.positions.append(position)
        metrics = analyze_path(
            self.positions,
            frame_index=len(self.positions) - 1,
            fps=30.0,
        )
        self.rep_count = metrics.reps
        return {
            "position": position,
            "velocity": metrics.velocity_px_s,
            "rep_count": self.rep_count,
            "vertical_displacement": metrics.vertical_displacement_px,
        }
