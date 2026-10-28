"""Define InfMovie clip selection, rendering, and optional registry settings."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class InfMoviePipelineConfig:
    """Typed runtime options for inference overlays.

    The command-line interface accepts these options directly and records the
    resolved values in each run log. Folder and registry discovery are supported.
    """

    # Paths
    # Folder containing inference CSVs when registry discovery is disabled.
    inference_csv_dir: str | None = None

    # Folder containing cropped movies when registry discovery is disabled.
    movie_dir: str | None = None

    # Root output folder (typically "7_infmovies")
    output_dir: str | None = None

    # Optional substring filter applied to recording_id / filenames (case-insensitive)
    file_filter: str | None = None

    # Mode
    # manual | interesting | qc | full
    mode: str = "manual"

    # Manual mode parameters
    recording_id: str | None = None
    start_sec: float = 0.0
    duration_sec: float = 20.0

    # Automated selection parameters (qc / interesting)
    num_clips: int = 10
    clip_duration_sec: float = 20.0
    clip_padding_sec: float = 0.0
    min_gap_sec: float = 20.0
    max_clips_per_recording: int = 1
    random_seed: int | None = 0

    # Which labels to use for selection: predicted | smoothed
    selection_label_source: str = "smoothed"

    # QC thresholds
    qc_uncertainty_threshold: float = 0.6
    qc_min_uncertainty_frames: int = 4
    qc_switch_window_sec: float = 5.0
    qc_switch_min_changes: int = 8
    qc_use_uncertainty: bool = True
    qc_use_switching: bool = True

    # Interesting thresholds
    interesting_min_unique_labels: int = 3
    interesting_min_changes: int = 2

    # Rendering
    num_workers: int = 8
    skip_first_n_frames: int = 5
    source_frame_offset: int | None = None

    timeline: str = "both"  # none|pose|orient|both
    timeline_compare: bool = False
    timeline_labels: str = "predicted"  # predicted|smoothed (single mode only)
    timeline_height: int = 36

    white_mode: bool = False

    speed_factor: float = 1.0
    max_output_fps: float = 40.0

    show_timestamp: bool = True
    timestamp_position: str = "top_right"
    timestamp_decimals: int = 1
    # movie|clip|rendered
    timestamp_reference: str = "movie"

    # If cv2 cannot read FPS, fall back to this.
    default_movie_fps: float = 40.0

    # Overwrite existing rendered movies.
    clobber: bool = False

    # Write plan CSV + run log
    write_plan_csv: bool = True
    write_run_log: bool = True

    # Project registry (optional)
    registry_enabled: bool = False
    registry_csv: str | None = None
    registry_lock_timeout_sec: int = 120
    registry_lock_stale_sec: int = 60 * 60

    def resolved_registry_csv(self) -> str | None:
        return self.registry_csv if self.registry_enabled else None
