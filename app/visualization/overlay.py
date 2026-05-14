"""Video overlay rendering."""

from __future__ import annotations

import cv2
import numpy as np

from app.biomechanics.metrics import FrameMetrics
from app.detection.base import BarbellDetection


ACCENT = (74, 222, 128)
CYAN = (34, 211, 238)
WHITE = (241, 245, 249)
PANEL = (15, 23, 42)
WARNING = (250, 204, 21)
MUTED = (148, 163, 184)


def draw_analysis_overlay(
    frame: np.ndarray,
    detection: BarbellDetection | None,
    path_points: list[tuple[float, float]],
    metrics: FrameMetrics,
    show_debug: bool = False,
    interpolate_path: bool = True,
) -> np.ndarray:
    """Render detection, trajectory and metrics onto a frame."""

    output = frame.copy()
    _draw_path(output, path_points, interpolate=interpolate_path)
    _draw_detection(output, detection)
    _draw_metrics_panel(output, metrics)
    if show_debug:
        _draw_debug_panel(output, metrics)
    return output


def _draw_path(frame: np.ndarray, points: list[tuple[float, float]], interpolate: bool) -> None:
    if len(points) < 2:
        return

    render_points = _interpolate_points(points) if interpolate else points
    for index in range(1, len(render_points)):
        start = tuple(int(v) for v in render_points[index - 1])
        end = tuple(int(v) for v in render_points[index])
        alpha = index / len(render_points)
        color = (
            int(CYAN[0] * (1.0 - alpha) + ACCENT[0] * alpha),
            int(CYAN[1] * (1.0 - alpha) + ACCENT[1] * alpha),
            int(CYAN[2] * (1.0 - alpha) + ACCENT[2] * alpha),
        )
        cv2.line(frame, start, end, color, thickness=2, lineType=cv2.LINE_AA)

    current = tuple(int(v) for v in render_points[-1])
    cv2.circle(frame, current, 4, ACCENT, thickness=-1, lineType=cv2.LINE_AA)
    cv2.circle(frame, current, 9, ACCENT, thickness=1, lineType=cv2.LINE_AA)


def _draw_detection(frame: np.ndarray, detection: BarbellDetection | None) -> None:
    if detection is None:
        cv2.putText(frame, "SEARCHING", (20, frame.shape[0] - 24), cv2.FONT_HERSHEY_SIMPLEX, 0.5, WARNING, 1)
        return

    x, y, w, h = detection.bbox
    center = tuple(int(v) for v in detection.center)
    if detection.label == "plate":
        radius = max(10, min(w, h) // 2)
        cv2.circle(frame, center, radius, ACCENT, thickness=1, lineType=cv2.LINE_AA)
    else:
        shaft_y = int(detection.center[1])
        cv2.line(frame, (x, shaft_y), (x + w, shaft_y), MUTED, thickness=1, lineType=cv2.LINE_AA)
    cv2.circle(frame, center, 4, ACCENT, thickness=-1, lineType=cv2.LINE_AA)


def _draw_metrics_panel(frame: np.ndarray, metrics: FrameMetrics) -> None:
    overlay = frame.copy()
    panel_width = 214
    panel_height = 96
    cv2.rectangle(overlay, (16, 16), (16 + panel_width, 16 + panel_height), PANEL, thickness=-1)
    cv2.addWeighted(overlay, 0.66, frame, 0.34, 0, frame)
    cv2.rectangle(frame, (16, 16), (16 + panel_width, 16 + panel_height), (51, 65, 85), thickness=1)

    rows = [
        ("PHASE", metrics.phase.upper()),
        ("REPS", str(metrics.reps)),
        ("VEL", f"{metrics.velocity_px_s:.0f} px/s"),
        ("ROM", f"{metrics.vertical_displacement_px:.0f} px"),
    ]
    for offset, (label, value) in enumerate(rows):
        y = 38 + offset * 20
        cv2.putText(frame, label, (28, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, MUTED, 1, cv2.LINE_AA)
        cv2.putText(frame, value, (88, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, WHITE if label == "PHASE" else ACCENT, 1, cv2.LINE_AA)


def _draw_debug_panel(frame: np.ndarray, metrics: FrameMetrics) -> None:
    overlay = frame.copy()
    panel_width = 250
    panel_height = 94
    x1 = frame.shape[1] - panel_width - 16
    y1 = 16
    x2 = frame.shape[1] - 16
    y2 = y1 + panel_height
    cv2.rectangle(overlay, (x1, y1), (x2, y2), PANEL, thickness=-1)
    cv2.addWeighted(overlay, 0.58, frame, 0.42, 0, frame)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (51, 65, 85), thickness=1)

    rows = [
        f"bar y     {metrics.bar_height_px:6.1f}",
        f"lockout   {metrics.lockout_height_px:6.1f}",
        f"min depth {metrics.min_depth_px:6.1f}",
        f"events    {len(metrics.events):6d}",
    ]
    for index, row in enumerate(rows):
        y = y1 + 22 + index * 18
        cv2.putText(frame, row, (x1 + 12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, MUTED, 1, cv2.LINE_AA)


def _interpolate_points(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    if len(points) < 3:
        return points

    interpolated: list[tuple[float, float]] = [points[0]]
    for start, end in zip(points, points[1:]):
        sx, sy = start
        ex, ey = end
        distance = float(np.hypot(ex - sx, ey - sy))
        steps = max(1, min(6, int(distance / 8.0)))
        for step in range(1, steps + 1):
            ratio = step / steps
            interpolated.append((sx + (ex - sx) * ratio, sy + (ey - sy) * ratio))
    return interpolated
