from app.biomechanics.metrics import analyze_path
from app.detection.classical import ClassicalBarbellDetector
from app.tracking.centroid import CentroidTracker
from app.utils.config import AppConfig


def test_default_detector_initializes_without_model_weights() -> None:
    config = AppConfig()
    detector = ClassicalBarbellDetector(config.detection)

    assert detector.config == config.detection


def test_tracker_initializes_empty() -> None:
    tracker = CentroidTracker(AppConfig().tracking)

    assert tracker.track.latest is None
    assert tracker.track.points == []


def test_metrics_return_frame_summary() -> None:
    metrics = analyze_path([(0.0, 100.0), (0.0, 150.0)], frame_index=1, fps=30.0)

    assert metrics.frame_index == 1
    assert metrics.velocity_px_s > 0
