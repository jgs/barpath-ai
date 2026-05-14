from app.detection.classical import ClassicalBarbellDetector
from app.detection.hybrid import HybridPlateTracker
from app.detection.template import TemplateLockedDetector
from app.pipeline import build_detector
from app.utils import AppConfig, DetectionConfig


def test_build_detector_uses_hybrid_tracker_for_plate_targets() -> None:
    detector = build_detector(AppConfig(detection=DetectionConfig(target="plate")))

    assert isinstance(detector, HybridPlateTracker)


def test_build_detector_keeps_template_lock_for_auto_target() -> None:
    detector = build_detector(AppConfig(detection=DetectionConfig(target="auto")))

    assert isinstance(detector, TemplateLockedDetector)
