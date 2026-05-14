"""Biomechanical estimates derived from tracked bar path."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from app.utils.config import BiomechanicsConfig


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class RepEvent:
    """A detected repetition phase transition."""

    frame_index: int
    event_type: str
    y_position: float


@dataclass(frozen=True)
class FrameMetrics:
    """Metrics available at a specific frame."""

    frame_index: int
    reps: int
    velocity_px_s: float
    vertical_displacement_px: float
    phase: str
    bar_height_px: float
    lockout_height_px: float
    min_depth_px: float
    events: tuple[RepEvent, ...]


def analyze_path(
    points: list[tuple[float, float]],
    frame_index: int,
    fps: float,
    min_rep_displacement_px: float = 40.0,
    config: BiomechanicsConfig | None = None,
) -> FrameMetrics:
    """Estimate reps and vertical velocity from path points.

    OpenCV coordinates increase downward, so upward bar motion has negative
    delta-y. Velocity is reported as magnitude in pixels per second.
    """

    if len(points) < 2 or fps <= 0:
        empty_config = config or BiomechanicsConfig(min_rep_displacement_px=min_rep_displacement_px)
        return FrameMetrics(
            frame_index,
            reps=0,
            velocity_px_s=0.0,
            vertical_displacement_px=0.0,
            phase="LOCKOUT",
            bar_height_px=0.0,
            lockout_height_px=0.0,
            min_depth_px=empty_config.min_rep_displacement_px,
            events=(),
        )

    rep_config = config or BiomechanicsConfig(min_rep_displacement_px=min_rep_displacement_px)
    y = np.array([point[1] for point in points], dtype=np.float32)
    y_smooth = _moving_average(y, window=7)
    velocity = _stable_velocity(y_smooth, fps)
    displacement = float(y.max() - y.min())

    if _normalized_lift_type(rep_config.lift_type) == "deadlift":
        rep_state = _run_deadlift_state_machine(y_smooth, rep_config)
    else:
        rep_state = _run_top_down_state_machine(y_smooth, rep_config)

    return FrameMetrics(
        frame_index=frame_index,
        reps=rep_state.reps,
        velocity_px_s=velocity,
        vertical_displacement_px=displacement,
        phase=rep_state.phase,
        bar_height_px=float(y_smooth[-1]),
        lockout_height_px=rep_state.lockout_height,
        min_depth_px=rep_state.min_depth,
        events=tuple(rep_state.events),
    )


def _moving_average(values: np.ndarray, window: int) -> np.ndarray:
    if len(values) < window:
        return values
    kernel = np.ones(window, dtype=np.float32) / float(window)
    padded = np.pad(values, (window // 2, window // 2), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def _stable_velocity(y_positions: np.ndarray, fps: float, window: int = 8) -> float:
    if len(y_positions) < 2 or fps <= 0:
        return 0.0

    samples = min(window, len(y_positions) - 1)
    recent = y_positions[-(samples + 1) :]
    frame_offsets = np.arange(len(recent), dtype=np.float32)
    slope_px_per_frame = float(np.polyfit(frame_offsets, recent, deg=1)[0])
    return abs(slope_px_per_frame) * fps


@dataclass(frozen=True)
class _RepState:
    reps: int
    phase: str
    events: list[RepEvent]
    lockout_height: float
    min_depth: float


def _run_top_down_state_machine(y_positions: np.ndarray, config: BiomechanicsConfig) -> _RepState:
    if len(y_positions) < 8:
        lockout_height = float(y_positions[-1]) if len(y_positions) else 0.0
        return _RepState(
            reps=0,
            phase="LOCKOUT",
            events=[],
            lockout_height=lockout_height,
            min_depth=config.min_rep_displacement_px,
        )

    observed_range = float(y_positions.max() - y_positions.min())
    min_depth = max(config.min_rep_displacement_px * 0.75, observed_range * config.min_rep_depth_ratio)
    min_depth = max(24.0, min_depth)
    deadband = max(1.0, min_depth * config.velocity_deadband_ratio)
    descent_start_depth = max(7.0, min_depth * 0.16)
    lockout_tolerance = max(12.0, min_depth * config.lockout_tolerance_ratio)
    descent_frames = max(2, config.descent_debounce_frames)
    ascent_frames = max(2, config.ascent_debounce_frames)
    stable_frames_required = max(1, config.bottom_stable_frames)
    cooldown_frames = max(1, config.rep_cooldown_frames)

    velocities = np.diff(y_positions)
    lockout_height = float(np.median(y_positions[: min(8, len(y_positions))]))
    phase = "LOCKOUT"
    events: list[RepEvent] = []
    reps = 0
    down_count = 0
    up_count = 0
    stable_count = 0
    cooldown = 0
    rep_start_y = lockout_height
    bottom_y = lockout_height
    bottom_index: int | None = None

    for index, velocity in enumerate(velocities, start=1):
        current_y = float(y_positions[index])
        if velocity > deadband:
            down_count += 1
            up_count = 0
            stable_count = 0
        elif velocity < -deadband:
            up_count += 1
            down_count = 0
            stable_count = 0
        else:
            stable_count += 1
            down_count = 0
            up_count = 0

        if cooldown > 0:
            cooldown -= 1

        if phase == "LOCKOUT":
            if abs(current_y - lockout_height) <= lockout_tolerance and stable_count > 0:
                lockout_height = lockout_height * 0.96 + current_y * 0.04
            if cooldown == 0 and down_count >= descent_frames and current_y - lockout_height >= descent_start_depth:
                phase = "DESCENT"
                rep_start_y = lockout_height
                bottom_y = current_y
                bottom_index = index
                events.append(RepEvent(index, "descent", current_y))
                _log_transition(config, phase, index, current_y, "sustained downward movement")
            continue

        if phase == "DESCENT":
            if current_y > bottom_y:
                bottom_y = current_y
                bottom_index = index
            depth = bottom_y - rep_start_y
            if depth >= min_depth and stable_count >= stable_frames_required:
                phase = "BOTTOM"
                events.append(RepEvent(bottom_index or index, "bottom", bottom_y))
                _log_transition(config, phase, bottom_index or index, bottom_y, "stable bottom")
            elif depth >= min_depth and up_count >= ascent_frames:
                phase = "ASCENT"
                events.append(RepEvent(bottom_index or index, "bottom", bottom_y))
                _log_transition(config, phase, bottom_index or index, bottom_y, "upward reversal")
            elif current_y <= rep_start_y + lockout_tolerance and up_count >= ascent_frames:
                phase = "LOCKOUT"
                cooldown = cooldown_frames
            continue

        if phase == "BOTTOM":
            if current_y > bottom_y:
                bottom_y = current_y
                bottom_index = index
            if up_count >= ascent_frames:
                phase = "ASCENT"
                _log_transition(config, phase, index, current_y, "sustained upward movement")
            continue

        if phase == "ASCENT":
            depth = bottom_y - rep_start_y
            near_original_lockout = current_y <= rep_start_y + lockout_tolerance
            tiny_top_motion = abs(float(velocity)) <= deadband * 1.8
            stable_or_up = stable_count >= 1 or up_count >= 1 or tiny_top_motion
            if depth >= min_depth and near_original_lockout and stable_or_up and cooldown == 0:
                reps += 1
                phase = "LOCKOUT"
                lockout_height = rep_start_y * 0.85 + current_y * 0.15
                cooldown = cooldown_frames
                events.append(RepEvent(index, "lockout", current_y))
                _log_transition(config, phase, index, current_y, "returned near lockout")

    return _RepState(reps=reps, phase=phase, events=events, lockout_height=lockout_height, min_depth=min_depth)


def _run_deadlift_state_machine(y_positions: np.ndarray, config: BiomechanicsConfig) -> _RepState:
    if len(y_positions) < 8:
        start_height = float(y_positions[-1]) if len(y_positions) else 0.0
        return _RepState(
            reps=0,
            phase="BOTTOM",
            events=[],
            lockout_height=start_height,
            min_depth=config.min_rep_displacement_px,
        )

    observed_range = float(y_positions.max() - y_positions.min())
    min_depth = max(config.min_rep_displacement_px * 0.75, observed_range * config.min_rep_depth_ratio)
    min_depth = max(24.0, min_depth)
    deadband = max(1.0, min_depth * config.velocity_deadband_ratio)
    lockout_tolerance = max(12.0, min_depth * config.lockout_tolerance_ratio)
    ascent_frames = max(2, config.ascent_debounce_frames)
    descent_frames = max(2, config.descent_debounce_frames)
    cooldown_frames = max(1, config.rep_cooldown_frames)

    velocities = np.diff(y_positions)
    bottom_height = float(np.median(y_positions[: min(8, len(y_positions))]))
    lockout_height = bottom_height
    phase = "BOTTOM"
    events: list[RepEvent] = []
    reps = 0
    up_count = 0
    down_count = 0
    stable_count = 0
    cooldown = 0
    rep_start_y = bottom_height
    highest_y = bottom_height
    highest_index = 0

    for index, velocity in enumerate(velocities, start=1):
        current_y = float(y_positions[index])
        if velocity < -deadband:
            up_count += 1
            down_count = 0
            stable_count = 0
        elif velocity > deadband:
            down_count += 1
            up_count = 0
            stable_count = 0
        else:
            stable_count += 1
            up_count = 0
            down_count = 0

        if cooldown > 0:
            cooldown -= 1

        if phase == "BOTTOM":
            if current_y > bottom_height and stable_count > 0:
                bottom_height = bottom_height * 0.96 + current_y * 0.04
            if cooldown == 0 and up_count >= ascent_frames and bottom_height - current_y >= max(7.0, min_depth * 0.16):
                phase = "ASCENT"
                rep_start_y = bottom_height
                highest_y = current_y
                highest_index = index
                events.append(RepEvent(index, "ascent", current_y))
                _log_transition(config, phase, index, current_y, "deadlift pull started")
            continue

        if phase == "ASCENT":
            if current_y < highest_y:
                highest_y = current_y
                highest_index = index
            displacement = rep_start_y - highest_y
            near_top = current_y <= rep_start_y - min_depth + lockout_tolerance
            tiny_top_motion = abs(float(velocity)) <= deadband * 1.8
            if displacement >= min_depth and near_top and (stable_count >= 1 or up_count >= 1 or tiny_top_motion) and cooldown == 0:
                reps += 1
                phase = "LOCKOUT"
                lockout_height = highest_y
                cooldown = cooldown_frames
                events.append(RepEvent(highest_index, "lockout", highest_y))
                _log_transition(config, phase, highest_index, highest_y, "deadlift lockout reached")
            continue

        if phase == "LOCKOUT":
            if abs(current_y - lockout_height) <= lockout_tolerance and stable_count > 0:
                lockout_height = lockout_height * 0.96 + current_y * 0.04
            if down_count >= descent_frames and current_y - lockout_height >= max(7.0, min_depth * 0.16):
                phase = "DESCENT"
                events.append(RepEvent(index, "descent", current_y))
                _log_transition(config, phase, index, current_y, "deadlift lowering started")
            continue

        if phase == "DESCENT":
            if current_y >= rep_start_y - lockout_tolerance:
                phase = "BOTTOM"
                bottom_height = bottom_height * 0.8 + current_y * 0.2
                cooldown = cooldown_frames
                events.append(RepEvent(index, "bottom", current_y))
                _log_transition(config, phase, index, current_y, "bar returned to floor")

    return _RepState(reps=reps, phase=phase, events=events, lockout_height=lockout_height, min_depth=min_depth)


def _normalized_lift_type(lift_type: str) -> str:
    normalized = lift_type.strip().lower().replace(" ", "_")
    if normalized in {"bench", "bench_press", "press_banca"}:
        return "bench"
    if normalized in {"deadlift", "peso_muerto"}:
        return "deadlift"
    return "squat"


def _log_transition(
    config: BiomechanicsConfig,
    phase: str,
    frame_index: int,
    y_position: float,
    reason: str,
) -> None:
    if not config.debug_logging:
        return
    LOGGER.info("Rep phase -> %s at frame %s y=%.1f (%s)", phase, frame_index, y_position, reason)
