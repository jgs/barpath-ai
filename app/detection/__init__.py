"""Barbell detection backends."""

from app.detection.base import BarbellDetection, Detector
from app.detection.classical import ClassicalBarbellDetector
from app.detection.hybrid import HybridPlateTracker
from app.detection.template import TemplateLockedDetector
from app.detection.yolo import YOLOBarbellDetector

__all__ = [
    "BarbellDetection",
    "ClassicalBarbellDetector",
    "Detector",
    "HybridPlateTracker",
    "TemplateLockedDetector",
    "YOLOBarbellDetector",
]
