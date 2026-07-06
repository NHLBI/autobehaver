"""Configuration for autocropper."""

from __future__ import annotations

from dataclasses import dataclass
import os
import re
from pathlib import Path


def default_resource_path(filename: str) -> str:
    """Return an absolute path to a packaged resource file.

    The repo/zip layout is expected to include:
        autobehaver/resources/autocropper/<filename>
    """
    here = Path(__file__).resolve()
    autobehaver_dir = here.parents[2]
    p = autobehaver_dir / "resources" / "autocropper" / filename
    return str(p)


@dataclass(frozen=True)
class AutocropperConfig:
    """Runtime configuration for autocropper."""

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

    # Validation
    skip_validation: bool = False
    model_path: str | None = None
    norm_params_path: str | None = None

    # Logging
    log_level: str = "INFO"

    # ----------------------
    # Project registry (single CSV at project root)
    # ----------------------
    registry_enabled: bool = False
    registry_csv: str | None = None
    registry_lock_timeout_sec: int = 120
    registry_lock_stale_sec: int = 60 * 60

    # ----------------------
    # Resource resolution
    # ----------------------
    def resolved_model_path(self) -> str:
        return self.model_path or default_resource_path(
            "emptychamber_detector_v2_checkpoint_epoch_331.pt"
        )

    def resolved_norm_params_path(self) -> str:
        return self.norm_params_path or default_resource_path("normalization_params.json")

    # ----------------------
    # Path resolution
    # ----------------------
    def resolved_crop_output_dir(self) -> str:
        """Directory where cropped chamber movies are written."""
        if self.output_dir:
            return str(self.output_dir)
        return str(Path(self.input_dir) / "cropped_chambers")

    def resolved_run_dir(self) -> str:
        """Directory where logs/validation artifacts are written."""
        return str(self.output_dir or self.input_dir)

    def resolved_exclusions_csv(self) -> str:
        """Path to exclusions.csv (moviechecker output)."""
        if self.exclusions_csv:
            return str(self.exclusions_csv)
        return str(Path(self.input_dir) / "exclusions.csv")

    def resolved_recording_database_dir(self) -> str:
        """Directory that contains recording_database_*.csv files.

        If not provided, defaults to ``input_dir``.
        """
        return str(self.recording_database_dir or self.input_dir)

    def resolved_recording_database_csv(self) -> str | None:
        """Explicit path to a recording database CSV (optional override)."""
        return str(self.recording_database_csv) if self.recording_database_csv else None

    def resolved_registry_csv(self) -> str | None:
        """Path to the project-wide registry CSV (or None if disabled)."""

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
