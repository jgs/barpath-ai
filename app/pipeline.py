"""End-to-end video analysis pipeline."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable
from uuid import uuid4

import cv2

from app.biomechanics import FrameMetrics, analyze_path
from app.detection import ClassicalBarbellDetector, Detector, HybridPlateTracker, TemplateLockedDetector, YOLOBarbellDetector
from app.tracking import CentroidTracker
from app.utils.config import AppConfig
from app.visualization import draw_analysis_overlay

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class AnalysisResult:
    """Summary returned after processing a video."""

    output_path: Path
    frames_processed: int
    fps: float
    final_metrics: FrameMetrics


ProgressCallback = Callable[[int, int], None]


def build_detector(config: AppConfig) -> Detector:
    """Create the configured detector backend."""

    if config.detection.backend == "yolo":
        detector: Detector = YOLOBarbellDetector(config.detection)
    else:
        detector = ClassicalBarbellDetector(config.detection)
    if config.detection.enable_hybrid_plate_tracking and config.detection.target in {"plate", "right_plate", "left_plate"}:
        return HybridPlateTracker(detector, config.detection)
    if config.detection.enable_template_lock and config.detection.target == "auto":
        return TemplateLockedDetector(detector, config.detection)
    return detector


def analyze_video(
    video_path: Path,
    config: AppConfig,
    progress_callback: ProgressCallback | None = None,
) -> AnalysisResult:
    """Analyze a lifting video and export an annotated MP4."""

    config.video.output_dir.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = float(capture.get(cv2.CAP_PROP_FPS) or 30.0)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    output_path = config.video.output_dir / f"barpath_{uuid4().hex[:10]}.mp4"
    frame_stride = max(1, config.video.frame_stride)
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if config.video.max_frames is not None:
        total_frames = min(total_frames, config.video.max_frames)
    output_fps = max(1.0, fps / frame_stride)

    fourcc = cv2.VideoWriter_fourcc(*config.video.codec)
    writer = cv2.VideoWriter(str(output_path), fourcc, output_fps, (width, height))
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"Could not create output video: {output_path}")

    detector = build_detector(config)
    tracker = CentroidTracker(config.tracking)
    final_metrics = FrameMetrics(0, 0, 0.0, 0.0, "LOCKOUT", 0.0, 0.0, config.biomechanics.min_rep_displacement_px, ())
    frames_processed = 0
    frames_seen = 0

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if config.video.max_frames is not None and frames_seen >= config.video.max_frames:
                break

            should_process = frames_seen % frame_stride == 0
            frames_seen += 1
            if not should_process:
                continue

            detection = detector.detect(frame)
            track = tracker.update(detection)
            final_metrics = analyze_path(
                track.points,
                frames_processed,
                output_fps,
                config=config.biomechanics,
            )
            annotated = draw_analysis_overlay(
                frame,
                detection,
                track.points,
                final_metrics,
                show_debug=config.visualization.show_debug,
                interpolate_path=config.visualization.path_interpolation,
            )
            writer.write(annotated)
            frames_processed += 1

            if frames_processed % 30 == 0:
                LOGGER.info("Processed %s frames", frames_processed)
                if progress_callback is not None:
                    progress_callback(frames_seen, total_frames)
    finally:
        capture.release()
        writer.release()

    return AnalysisResult(
        output_path=output_path,
        frames_processed=frames_processed,
        fps=fps,
        final_metrics=final_metrics,
    )
