from app.detection.base import BarbellDetection
from app.detection.classical import _Candidate, _filter_candidates_for_target


def test_filter_candidates_keeps_only_requested_plate_side() -> None:
    candidates = [
        _Candidate(BarbellDetection((100.0, 200.0), (80, 180, 40, 40), 0.8, "plate"), 1.0),
        _Candidate(BarbellDetection((700.0, 210.0), (680, 190, 40, 40), 0.8, "plate"), 1.0),
        _Candidate(BarbellDetection((400.0, 220.0), (100, 210, 600, 10), 0.8, "barbell"), 1.0),
    ]

    right = _filter_candidates_for_target(candidates, "right_plate", frame_width=800)
    left = _filter_candidates_for_target(candidates, "left_plate", frame_width=800)

    assert [candidate.detection.center for candidate in right] == [(700.0, 210.0)]
    assert [candidate.detection.center for candidate in left] == [(100.0, 200.0)]


def test_filter_candidates_accepts_any_side_for_plate_target() -> None:
    candidates = [
        _Candidate(BarbellDetection((100.0, 200.0), (80, 180, 40, 40), 0.8, "plate"), 1.0),
        _Candidate(BarbellDetection((400.0, 205.0), (380, 185, 40, 40), 0.8, "plate"), 1.0),
        _Candidate(BarbellDetection((700.0, 210.0), (680, 190, 40, 40), 0.8, "plate"), 1.0),
        _Candidate(BarbellDetection((400.0, 220.0), (100, 210, 600, 10), 0.8, "barbell"), 1.0),
    ]

    filtered = _filter_candidates_for_target(candidates, "plate", frame_width=800)

    assert [candidate.detection.center for candidate in filtered] == [(100.0, 200.0), (700.0, 210.0)]
