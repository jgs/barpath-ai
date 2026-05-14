from app.detection.base import BarbellDetection
from app.detection.template import _plate_center_bbox


def test_plate_center_bbox_uses_small_patch_around_plate_center() -> None:
    detection = BarbellDetection(center=(200.0, 150.0), bbox=(80, 30, 240, 240), confidence=0.8, label="plate")

    bbox = _plate_center_bbox(detection, radius_ratio=0.3, frame_width=400, frame_height=300)

    assert bbox == (164, 114, 72, 72)
