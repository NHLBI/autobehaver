"""Define runtime options and output paths for Autocropper."""

from __future__ import annotations

from dataclasses import dataclass
import os
import re
from pathlib import Path


@dataclass(frozen=True)
class AutocropperConfig:
    """Store runtime options for Autocropper."""

    # Full-frame movies live here
    input_dir: str

    # Cropped chamber movies (and stage artifacts) live here (optional)
    output_dir: str | None = None

    # MovieChecker exclusions
    exclusions_csv: str | None = None
    require_exclusions: bool = True

    # Recording DB labeling (prefix cropped chamber movies with a chamber-specific label)
    recording_database_dir: str | None = None
    recording_database_csv: str | None = None
    skip_labeling: bool = False

    # Optional subset selection
    file_filter: str | None = None
    movie_list: str | None = None
    trial_number: int | None = None

    # Retry loop
    max_retries: int = 10

    # Geometry / acquisition settings
    approx_radius: int = 265
    chamber_columns: int = 3
    chamber_rows: int = 2
    crop_dim_2k: int = 600
    chamber_calibration_json: str | None = None

    # Detection settings
    frame_avg: int = 10
    down_factor: int = 2

    # Parallelism
    num_workers: int = 2
    max_workers: int = 6
    native_threads: int = 1
    ffmpeg_threads: int = 1

    # Monitoring / robustness
    heartbeat_interval: int = 10
    timeout: int = 600
    partial_cleanup_min_age_sec: int = 60 * 60

    # Output encoding
    crop_crf: int = 12

    # Logging
    log_level: str = "INFO"

    # Project registry (single CSV at project root)
    registry_enabled: bool = False
    registry_csv: str | None = None
    registry_lock_timeout_sec: int = 120
    registry_lock_stale_sec: int = 60 * 60

    # Path resolution
    def resolved_crop_output_dir(self) -> str:
        """Return the directory for cropped chamber movies."""
        if self.output_dir:
            return str(self.output_dir)
        return str(Path(self.input_dir) / "cropped_chambers")

    def resolved_run_dir(self) -> str:
        """Return the directory for logs and stage artifacts."""
        return str(self.output_dir or self.input_dir)

    def resolved_exclusions_csv(self) -> str:
        """Return the explicit exclusions path or the input-folder default."""
        if self.exclusions_csv:
            return str(Path(self.exclusions_csv).expanduser().resolve())
        return str((Path(self.input_dir).expanduser().resolve() / "exclusions.csv"))

    def resolved_recording_database_dir(self) -> str:
        """Return the metadata directory, defaulting to input_dir."""
        return str(self.recording_database_dir or self.input_dir)

    def resolved_recording_database_csv(self) -> str | None:
        """Return the explicit recording metadata path, if supplied."""
        return str(self.recording_database_csv) if self.recording_database_csv else None

    def resolved_registry_csv(self) -> str | None:
        """Return the explicit or inferred registry path, or None when disabled."""

        if not bool(self.registry_enabled):
            return None

        if self.registry_csv:
            return str(Path(self.registry_csv).expanduser().resolve())

        # Infer project root from the input/output folder structure.
        in_dir = Path(self.input_dir).expanduser().resolve()
        if self.output_dir:
            out_dir = Path(self.output_dir).expanduser().resolve()
            try:
                root = Path(os.path.commonpath([str(in_dir), str(out_dir)])).resolve()
            except Exception:
                root = in_dir.parent
        else:
            root = in_dir.parent if re.match(r"^\d+_", in_dir.name) else in_dir

        return str(root / "registry.csv")
