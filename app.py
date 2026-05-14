"""Streamlit interface for BarPath AI."""

from __future__ import annotations

import logging
from pathlib import Path

import streamlit as st

from app.pipeline import analyze_video
from app.utils import AppConfig, BiomechanicsConfig, DetectionConfig, TrackingConfig, VideoConfig, VisualizationConfig, configure_logging
from app.utils.video import persist_uploaded_video

configure_logging()
LOGGER = logging.getLogger(__name__)


st.set_page_config(
    page_title="BarPath AI",
    page_icon="assets/hero.svg",
    layout="wide",
)

st.markdown(
    """
    <style>
    :root {
      --bg: #020617;
      --panel: #0f172a;
      --line: #1e293b;
      --text: #e2e8f0;
      --muted: #94a3b8;
      --accent: #4ade80;
      --cyan: #22d3ee;
    }
    .stApp { background: radial-gradient(circle at top left, #0f172a 0, #020617 36rem); color: var(--text); }
    [data-testid="stHeader"] { background: rgba(2, 6, 23, 0); }
    [data-testid="stSidebar"] { background: #020617; border-right: 1px solid var(--line); }
    h1, h2, h3 { letter-spacing: 0; }
    .hero {
      padding: 2rem 0 1.25rem;
      border-bottom: 1px solid var(--line);
      margin-bottom: 1.25rem;
    }
    .hero h1 {
      font-size: clamp(2.4rem, 5vw, 5.2rem);
      line-height: 0.95;
      margin: 0;
      color: #f8fafc;
    }
    .hero p {
      max-width: 760px;
      color: var(--muted);
      font-size: 1.08rem;
      margin-top: 1rem;
    }
    .metric-card {
      border: 1px solid var(--line);
      background: rgba(15, 23, 42, 0.74);
      border-radius: 8px;
      padding: 1rem;
    }
    .metric-card span {
      color: var(--muted);
      font-size: 0.78rem;
      text-transform: uppercase;
      letter-spacing: 0.08em;
    }
    .metric-card strong {
      display: block;
      margin-top: 0.35rem;
      color: var(--accent);
      font-size: 1.65rem;
    }
    .stButton button, .stDownloadButton button {
      border-radius: 8px;
      border: 1px solid #334155;
      background: #111827;
      color: #f8fafc;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <section class="hero">
      <h1>BarPath AI</h1>
      <p>Computer vision biomechanics analysis for squat, bench press and deadlift footage.
      Track the barbell, render its trajectory, estimate velocity and export a coach-ready overlay.</p>
    </section>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.subheader("Analysis")
    lift_type = st.selectbox(
        "Lift type",
        options=["squat", "bench", "deadlift"],
        index=0,
        help="Adapts detection zones and rep-counting logic to the lift.",
    )
    backend = st.selectbox("Detector", options=["classical", "yolo"], index=0)
    target = st.selectbox(
        "Tracking target",
        options=["auto", "bar", "plate", "right_plate", "left_plate"],
        index=1,
        help="Use bar for clear shafts, plate for the strongest visible disc, or pick a side when one sleeve is always visible.",
    )
    weights = st.text_input("YOLO weights path", value="", disabled=backend != "yolo")
    confidence = st.slider("Confidence threshold", 0.05, 0.95, 0.35, 0.05)
    smoothing = st.slider("Tracking smoothing", 0.05, 0.95, 0.35, 0.05)
    show_debug = st.checkbox(
        "Debug overlay",
        value=False,
        help="Show state-machine internals on the exported video.",
    )
    mode = st.radio(
        "Run mode",
        options=["Fast preview", "Full export"],
        index=0,
        help="Fast preview samples frames so you can iterate quickly. Full export processes every frame.",
    )
    st.caption("Use classical mode for a zero-training baseline. Use YOLO mode with custom barbell weights.")

uploaded = st.file_uploader(
    "Upload a lifting video",
    type=["mp4", "mov", "avi", "mkv"],
    accept_multiple_files=False,
)

left, right = st.columns([0.92, 1.08], gap="large")

with left:
    st.subheader("Source")
    if uploaded is None:
        st.info("Upload a squat, bench press or deadlift clip to begin.")
    else:
        if uploaded.name.lower().startswith("barpath_"):
            st.warning(
                "This looks like an exported BarPath AI video. Use the original raw clip for best tracking; "
                "old overlays are already burned into exported videos."
            )
        st.video(uploaded)

with right:
    st.subheader("Analyzed Output")
    run = st.button("Analyze video", type="primary", disabled=uploaded is None, use_container_width=True)

    if uploaded is not None and run:
        lift_detection_overrides = {
            "squat": {"roi_top_ratio": 0.20, "roi_bottom_ratio": 0.96, "max_bar_line_y_ratio": 0.72},
            "bench": {"roi_top_ratio": 0.16, "roi_bottom_ratio": 0.88, "max_bar_line_y_ratio": 0.78},
            "deadlift": {
                "roi_top_ratio": 0.18,
                "roi_bottom_ratio": 0.98,
                "max_bar_line_y_ratio": 0.98,
                "min_plate_center_y_ratio": 0.10,
                "max_plate_center_y_ratio": 0.96,
                "preferred_plate_radius_ratio": 0.18,
            },
        }[lift_type]
        tracking_target = "plate" if lift_type == "deadlift" and target in {"auto", "bar"} else target
        lift_depth_defaults = {
            "squat": 40.0,
            "bench": 24.0,
            "deadlift": 45.0,
        }[lift_type]
        config = AppConfig(
            detection=DetectionConfig(
                backend=backend,
                target=tracking_target,
                lift_type=lift_type,
                yolo_weights=weights or None,
                confidence_threshold=confidence,
                **lift_detection_overrides,
            ),
            tracking=TrackingConfig(smoothing_alpha=smoothing),
            biomechanics=BiomechanicsConfig(
                lift_type=lift_type,
                min_rep_displacement_px=lift_depth_defaults,
                min_rep_depth_ratio=0.34 if lift_type == "bench" else 0.42,
                lockout_tolerance_ratio=0.38 if lift_type == "deadlift" else 0.32,
                debug_logging=show_debug,
            ),
            visualization=VisualizationConfig(show_debug=show_debug),
            video=VideoConfig(
                frame_stride=3 if mode == "Fast preview" else 1,
                max_frames=None,
            ),
        )
        source_path = persist_uploaded_video(uploaded)

        try:
            with st.status("Tracking bar path...", expanded=True) as status:
                progress = st.progress(0)

                def update_progress(done: int, total: int) -> None:
                    if total > 0:
                        progress.progress(min(1.0, done / total))

                st.write("Loading video frames")
                result = analyze_video(Path(source_path), config, progress_callback=update_progress)
                progress.progress(1.0)
                st.write(f"Processed {result.frames_processed} frames")
                status.update(label="Analysis complete", state="complete")

            c1, c2, c3 = st.columns(3)
            c1.markdown(f'<div class="metric-card"><span>Reps</span><strong>{result.final_metrics.reps}</strong></div>', unsafe_allow_html=True)
            c2.markdown(f'<div class="metric-card"><span>Velocity</span><strong>{result.final_metrics.velocity_px_s:.1f} px/s</strong></div>', unsafe_allow_html=True)
            c3.markdown(f'<div class="metric-card"><span>ROM</span><strong>{result.final_metrics.vertical_displacement_px:.0f} px</strong></div>', unsafe_allow_html=True)

            st.video(str(result.output_path))
            with result.output_path.open("rb") as analyzed_file:
                st.download_button(
                    "Download analyzed video",
                    data=analyzed_file,
                    file_name=result.output_path.name,
                    mime="video/mp4",
                    use_container_width=True,
                )
        except Exception as exc:
            LOGGER.exception("Video analysis failed")
            st.error(f"Analysis failed: {exc}")
