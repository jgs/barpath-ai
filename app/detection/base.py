"""Detection interfaces and shared types."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


@dataclass(frozen=True)
class BarbellDetection:
    """Single-frame barbell detection."""

    center: tuple[float, float]
    bbox: tuple[int, int, int, int]
    confidence: float
    label: str = "barbell"


class Detector(Protocol):
    """Frame-level detector contract."""

    def detect(self, frame: np.ndarray) -> BarbellDetection | None:
        """Return the best barbell detection for a frame."""
