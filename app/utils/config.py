"""Application configuration."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class DetectionConfig:
    """Detector configuration."""

    backend: str = "classical"
    target: str = "bar"
    lift_type: str = "squat"
    confidence_threshold: float = 0.35
    yolo_weights: str | None = None
    canny_low: int = 60
    canny_high: int = 150
    min_plate_area: int = 120
    max_plate_area_ratio: float = 0.12
    min_circularity: float = 0.35
    min_aspect_ratio: float = 0.45
    max_aspect_ratio: float = 2.2
    equalize_histogram: bool = True
    roi_top_ratio: float = 0.28
    roi_bottom_ratio: float = 0.94
    max_bar_line_y_ratio: float = 0.58
    min_bar_line_length_ratio: float = 0.18
    max_bar_angle_degrees: float = 8.0
    temporal_prior_weight: float = 0.55
    max_colored_line_saturation: float = 0.35
    enable_plate_circle_detection: bool = True
    min_plate_radius_ratio: float = 0.045
    max_plate_radius_ratio: float = 0.26
    preferred_plate_radius_ratio: float = 0.11
    max_plate_center_y_ratio: float = 0.72
    min_plate_center_y_ratio: float = 0.14
    circle_detection_width: int = 360
    plate_detection_interval: int = 1
    temporal_search_margin_ratio: float = 0.22
    suppress_overlay_markup: bool = True
    enable_template_lock: bool = True
    template_search_margin_ratio: float = 0.12
    template_min_confidence: float = 0.48
    detector_refresh_interval: int = 6
    plate_template_radius_ratio: float = 0.34
    enable_hybrid_plate_tracking: bool = True
    hybrid_detector_interval: int = 12
    optical_flow_feature_count: int = 80
    optical_flow_min_features: int = 8
    hybrid_max_jump_ratio: float = 0.16
    kalman_process_noise: float = 0.025
    kalman_measurement_noise: float = 0.85


@dataclass(frozen=True)
class TrackingConfig:
    """Tracker configuration."""

    smoothing_alpha: float = 0.35
    horizontal_smoothing_alpha: float = 0.08
    moving_average_window: int = 5
    max_missed_frames: int = 12
    max_jump_px: float = 140.0
    max_horizontal_jump_px: float = 260.0
    min_confidence: float = 0.12


@dataclass(frozen=True)
class BiomechanicsConfig:
    """Biomechanics and rep-counting configuration."""

    lift_type: str = "squat"
    min_rep_displacement_px: float = 40.0
    min_rep_depth_ratio: float = 0.42
    descent_debounce_frames: int = 3
    ascent_debounce_frames: int = 3
    bottom_stable_frames: int = 2
    rep_cooldown_frames: int = 8
    velocity_deadband_ratio: float = 0.025
    lockout_tolerance_ratio: float = 0.32
    debug_logging: bool = False


@dataclass(frozen=True)
class VisualizationConfig:
    """Overlay rendering configuration."""

    show_debug: bool = False
    path_interpolation: bool = True


@dataclass(frozen=True)
class VideoConfig:
    """Video processing configuration."""

    output_dir: Path = Path("assets/outputs")
    max_preview_width: int = 960
    codec: str = "mp4v"
    frame_stride: int = 1
    max_frames: int | None = None


@dataclass(frozen=True)
class AppConfig:
    """Top-level app configuration."""

    detection: DetectionConfig = field(default_factory=DetectionConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    biomechanics: BiomechanicsConfig = field(default_factory=BiomechanicsConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)
    video: VideoConfig = field(default_factory=VideoConfig)

    @classmethod
    def from_ui(
        cls,
        backend: str,
        yolo_weights: str | None,
        confidence_threshold: float,
        smoothing_alpha: float,
        frame_stride: int = 1,
        max_frames: int | None = None,
    ) -> "AppConfig":
        return cls(
            detection=DetectionConfig(
                backend=backend,
                yolo_weights=yolo_weights or None,
                confidence_threshold=confidence_threshold,
            ),
            tracking=TrackingConfig(smoothing_alpha=smoothing_alpha),
            video=VideoConfig(frame_stride=frame_stride, max_frames=max_frames),
        )


@dataclass(frozen=True)
class Config:
    """Compatibility config for early prototype modules.

    New code should use AppConfig and the focused nested config classes.
    """

    model_path: str = "assets/models/barbell-yolov8.pt"
    confidence_threshold: float = 0.35
    max_track_length: int = 180
    lift_type: str = "Squat"
