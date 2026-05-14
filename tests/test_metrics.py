from app.biomechanics.metrics import analyze_path
from app.utils.config import BiomechanicsConfig


def test_analyze_path_counts_lockout_after_vertical_cycle() -> None:
    points = [(100.0, y) for y in [100, 101, 103, 112, 132, 160, 190, 220, 235, 238, 236, 220, 190, 155, 122, 105, 101, 100]]

    metrics = analyze_path(points, frame_index=len(points) - 1, fps=30.0, min_rep_displacement_px=60.0)

    assert metrics.reps >= 1
    assert metrics.velocity_px_s > 0
    assert metrics.vertical_displacement_px > 100
    assert metrics.phase in {"LOCKOUT", "DESCENT", "BOTTOM", "ASCENT"}


def test_analyze_path_reports_stable_velocity_from_recent_trend() -> None:
    noisy_points = [(100.0, y) for y in [300, 298, 295, 291, 286, 284, 279, 281, 276]]

    metrics = analyze_path(noisy_points, frame_index=8, fps=30.0)

    assert 20.0 < metrics.velocity_px_s < 120.0
    assert metrics.phase in {"ASCENT", "LOCKOUT", "BOTTOM"}


def test_analyze_path_ignores_micro_movements() -> None:
    points = [(100.0, y) for y in [100, 101, 99, 102, 100, 101, 100, 99, 101, 100, 102, 100]]

    metrics = analyze_path(points, frame_index=len(points) - 1, fps=30.0, min_rep_displacement_px=60.0)

    assert metrics.reps == 0
    assert metrics.phase == "LOCKOUT"


def test_analyze_path_requires_return_to_lockout_to_count_rep() -> None:
    points = [
        (100.0, y)
        for y in [100, 100, 101, 103, 108, 120, 140, 165, 195, 225, 250, 260, 262, 261, 255, 240, 220, 200, 180, 160, 145, 135]
    ]

    metrics = analyze_path(points, frame_index=len(points) - 1, fps=30.0, min_rep_displacement_px=60.0)

    assert metrics.reps == 0
    assert metrics.phase == "ASCENT"


def test_analyze_path_counts_rep_with_brief_top_pause() -> None:
    points = [
        (100.0, y)
        for y in [
            100,
            101,
            103,
            112,
            134,
            164,
            198,
            232,
            252,
            258,
            258,
            252,
            228,
            194,
            158,
            125,
            108,
            103,
            102,
            102,
        ]
    ]

    metrics = analyze_path(points, frame_index=len(points) - 1, fps=30.0, min_rep_displacement_px=60.0)

    assert metrics.reps == 1
    assert metrics.phase == "LOCKOUT"
    assert metrics.lockout_height_px > 0
    assert metrics.min_depth_px > 0


def test_analyze_path_counts_bench_press_with_smaller_range() -> None:
    points = [
        (100.0, y)
        for y in [150, 151, 152, 158, 170, 186, 202, 213, 216, 215, 205, 188, 170, 156, 151, 150]
    ]

    metrics = analyze_path(
        points,
        frame_index=len(points) - 1,
        fps=30.0,
        config=BiomechanicsConfig(lift_type="bench", min_rep_displacement_px=24.0, min_rep_depth_ratio=0.34),
    )

    assert metrics.reps == 1
    assert metrics.phase == "LOCKOUT"


def test_analyze_path_counts_deadlift_from_floor_to_lockout() -> None:
    points = [
        (100.0, y)
        for y in [430, 430, 428, 418, 398, 370, 338, 304, 276, 252, 240, 238, 238, 239]
    ]

    metrics = analyze_path(
        points,
        frame_index=len(points) - 1,
        fps=30.0,
        config=BiomechanicsConfig(lift_type="deadlift", min_rep_displacement_px=45.0),
    )

    assert metrics.reps == 1
    assert metrics.phase == "LOCKOUT"


def test_analyze_path_deadlift_does_not_count_tiny_floor_movement() -> None:
    points = [(100.0, y) for y in [430, 431, 429, 427, 425, 424, 426, 423, 425, 424]]

    metrics = analyze_path(
        points,
        frame_index=len(points) - 1,
        fps=30.0,
        config=BiomechanicsConfig(lift_type="deadlift", min_rep_displacement_px=45.0),
    )

    assert metrics.reps == 0
