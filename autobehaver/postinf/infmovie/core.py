"""Run InfMovie overlays for inference CSVs in a source folder."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class InfMovieConfig:
    """Configuration for inference overlay movie generation."""

    inference_csv_dir: str
    movie_dir: str
    output_dir: Optional[str] = None

    num_workers: int = 16
    skip_first_n_frames: int = 5
    source_frame_offset: int | None = None

    # Optional timeline "barcode" visualization appended to the bottom of the output movie.
    # - none: do not draw
    # - pose: draw behavior (pose) timeline bar
    # - orient: draw orientation timeline bar
    # - both: draw behavior + orientation bars
    timeline: str = "none"

    # If True, draw predicted vs smoothed bars stacked for easy comparison.
    timeline_compare: bool = False

    # Which label source to use for the timeline bars in single mode.
    # - smoothed: Smoothed Pose/Orient Label (falls back to predicted if missing)
    # - predicted: Predicted Pose/Orient Label
    # Ignored when timeline_compare=True.
    timeline_labels: str = "predicted"

    # Height (pixels) for each timeline bar.
    timeline_height: int = 18

    # If True, render the right-side panel and timeline background in white with dark text.
    # Timeline bar colors are unchanged.
    white_mode: bool = False

    # Playback speed multiplier for the output movie.
    # Implemented by adjusting the output FPS, capped by max_output_fps.
    # If capping is required, frames are dropped to preserve the requested speed_factor.
    speed_factor: float = 1.0

    # Maximum FPS for the output movie file. Default is 40 (typical source FPS for this project).
    # If input_fps * speed_factor exceeds this value, output FPS is capped and frames are dropped.
    max_output_fps: float = 40.0

    # If True, overlay a timestamp (seconds) on the movie frame.
    show_timestamp: bool = False

    # Where to place the timestamp overlay on the movie frame.
    timestamp_position: str = "top_right"

    # Number of decimal places for the timestamp seconds.
    timestamp_decimals: int = 1


def run_infmovie(cfg: InfMovieConfig) -> None:
    """Generate overlay movies for all inference CSVs in a folder."""

    if not os.path.isdir(cfg.inference_csv_dir):
        raise FileNotFoundError(f"Inference CSV folder not found: {cfg.inference_csv_dir}")

    if not os.path.isdir(cfg.movie_dir):
        raise FileNotFoundError(f"Movie folder not found: {cfg.movie_dir}")

    output_dir = cfg.output_dir
    if not output_dir:
        output_dir = os.path.join(cfg.inference_csv_dir, "inference_movies")

    os.makedirs(output_dir, exist_ok=True)

    from .movies import generate_movies

    generate_movies(
        inference_csv_dir=cfg.inference_csv_dir,
        movie_dir=cfg.movie_dir,
        output_dir=output_dir,
        num_workers=int(cfg.num_workers),
        skip_first_n_frames=int(cfg.skip_first_n_frames),
        source_frame_offset=cfg.source_frame_offset,
        timeline=str(cfg.timeline),
        timeline_compare=bool(cfg.timeline_compare),
        timeline_labels=str(cfg.timeline_labels),
        timeline_height=int(cfg.timeline_height),
        white_mode=bool(cfg.white_mode),
        speed_factor=float(cfg.speed_factor),
        max_output_fps=float(cfg.max_output_fps),
        show_timestamp=bool(cfg.show_timestamp),
        timestamp_position=str(cfg.timestamp_position),
        timestamp_decimals=int(cfg.timestamp_decimals),
    )
