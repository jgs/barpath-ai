from app.detection.base import BarbellDetection
from app.tracking.centroid import CentroidTracker
from app.utils.config import TrackingConfig


def test_centroid_tracker_smooths_detection_points() -> None:
    tracker = CentroidTracker(TrackingConfig(smoothing_alpha=0.5, horizontal_smoothing_alpha=0.5))

    tracker.update(BarbellDetection(center=(0.0, 0.0), bbox=(0, 0, 10, 10), confidence=0.9))
    track = tracker.update(BarbellDetection(center=(10.0, 10.0), bbox=(5, 5, 10, 10), confidence=0.9))

    assert track.latest == (5.0, 5.0)


def test_centroid_tracker_rejects_large_single_frame_jump() -> None:
    tracker = CentroidTracker(
        TrackingConfig(smoothing_alpha=0.5, horizontal_smoothing_alpha=0.5, max_jump_px=50.0)
    )

    tracker.update(BarbellDetection(center=(100.0, 100.0), bbox=(95, 95, 10, 10), confidence=0.9))
    track = tracker.update(BarbellDetection(center=(500.0, 800.0), bbox=(495, 795, 10, 10), confidence=0.9))

    assert track.latest == (100.0, 100.0)
    assert track.missed_frames == 1


def test_centroid_tracker_accepts_side_switch_when_vertical_position_is_stable() -> None:
    tracker = CentroidTracker(
        TrackingConfig(
            smoothing_alpha=0.5,
            horizontal_smoothing_alpha=0.5,
            max_jump_px=50.0,
            max_horizontal_jump_px=120.0,
        )
    )

    tracker.update(BarbellDetection(center=(620.0, 410.0), bbox=(600, 390, 40, 40), confidence=0.9))
    track = tracker.update(BarbellDetection(center=(120.0, 420.0), bbox=(100, 400, 40, 40), confidence=0.9))

    assert track.latest == (620.0, 415.0)
    assert track.missed_frames == 0


def test_centroid_tracker_applies_moving_average_after_window_fills() -> None:
    tracker = CentroidTracker(
        TrackingConfig(
            smoothing_alpha=1.0,
            horizontal_smoothing_alpha=1.0,
            moving_average_window=3,
        )
    )

    tracker.update(BarbellDetection(center=(0.0, 100.0), bbox=(0, 95, 10, 10), confidence=0.9))
    tracker.update(BarbellDetection(center=(0.0, 110.0), bbox=(0, 105, 10, 10), confidence=0.9))
    track = tracker.update(BarbellDetection(center=(0.0, 120.0), bbox=(0, 115, 10, 10), confidence=0.9))

    assert track.latest == (0.0, 110.0)
