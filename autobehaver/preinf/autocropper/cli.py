"""Command-line options for cropping chamber videos."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


from autobehaver.argparse_utils import add_bool_pair, parse_cli_args
from autobehaver.resource_monitor import add_monitor_arguments, monitored_cli, record_settings
from autobehaver.pipeline_layout import apply_main_dir_overrides

from .config import AutocropperConfig
from .core import run_autocropper


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the Autocropper command-line parser with runtime defaults."""
    defaults = AutocropperConfig(input_dir="")
    p = argparse.ArgumentParser(description='Crop standard six-chamber recordings with Autocropper')


    # Project registry (single CSV at project root)
    add_bool_pair(
        p,
        "registry_enabled",
        default=defaults.registry_enabled,
        help=(
            "If enabled, update the project-wide registry CSV after autocropper finishes. "
            "Default: disabled."
        ),
        dest="registry_enabled",
    )

    p.add_argument(
        "--registry_csv",
        type=str,
        default=defaults.registry_csv,
        help=(
            "Path to the project registry CSV. If omitted and registry_enabled is True, "
            "defaults to <project_root>/registry.csv (project_root inferred from input/output dirs)."
        ),
    )

    p.add_argument(
        "--registry_lock_timeout_sec",
        type=int,
        default=defaults.registry_lock_timeout_sec,
        help="Seconds to wait for registry lock before failing (default 120).",
    )

    p.add_argument(
        "--registry_lock_stale_sec",
        type=int,
        default=defaults.registry_lock_stale_sec,
        help="Legacy option retained for compatibility; locks release when their process exits.",
    )


    p.add_argument(
        "--main_dir",
        type=str,
        default=None,
        help=(
            "Optional project root containing the standardized numbered pipeline folders. "
            "For autocropper this fills input_dir=<main_dir>/1_full_movies and output_dir=<main_dir>/2_cropped_movies."
        ),
    )

    # Inputs / outputs
    p.add_argument(
        "--input_dir",
        "-i",
        type=str,
        default=None,
        help=(
            "Input directory containing full-frame MP4 files to process. "
            "Required unless --main_dir supplies the standard project folder."
        ),
    )

    p.add_argument(
        "--output_dir",
        type=str,
        default=defaults.output_dir,
        help=(
            "Directory to write cropped chamber movies and stage artifacts. "
            "If omitted, cropped chamber movies go under <input_dir>/cropped_chambers and stage artifacts stay under <input_dir>."
        ),
    )

    # Exclusions (moviechecker)
    p.add_argument(
        "--exclusions_csv",
        type=str,
        default=defaults.exclusions_csv,
        help=(
            'Path to exclusions.csv produced by MovieChecker. Defaults to <input_dir>/exclusions.csv.'
        ),
    )

    add_bool_pair(
        p,
        "require_exclusions",
        default=defaults.require_exclusions,
        help=(
            "Require exclusions.csv to exist before cropping. "
            "Disable to allow cropping when exclusions.csv is missing."
        ),
        dest="require_exclusions",
    )

    # Recording DB labeling (prefix cropped outputs)
    p.add_argument(
        "--recording_database_dir",
        type=str,
        default=defaults.recording_database_dir,
        help=(
            "Directory containing recording_database*.csv or crop_occupancy_*.csv metadata. "
            "If omitted, defaults to <input_dir>."
        ),
    )

    p.add_argument(
        "--recording_database_csv",
        type=str,
        default=defaults.recording_database_csv,
        help=(
            "Optional explicit path to a recording database or crop occupancy CSV instead of autodiscovery "
            "from --recording_database_dir."
        ),
    )

    add_bool_pair(
        p,
        "skip_labeling",
        default=defaults.skip_labeling,
        help=(
            "Skip recording-database labeling (do not prefix cropped outputs). "
            "Default: labeling enabled."
        ),
        dest="skip_labeling",
    )

    # Subset selection
    p.add_argument(
        "--file_filter",
        type=str,
        default=defaults.file_filter,
        help="Only process movies whose filename contains this substring (case-insensitive).",
    )

    p.add_argument(
        "--movie_list",
        type=str,
        default=defaults.movie_list,
        help=(
            'Optional selection file with one movie filename or path per line; relative movie paths resolve under --input_dir. Without it, process the folder or use --trial_number/--file_filter. Takes precedence over both selectors.'
        ),
    )

    p.add_argument(
        "--trial_number",
        "--trial",
        type=int,
        default=defaults.trial_number,
        help=(
            'Select the exact _trialN_ filename token; trial1 does not select trial10 or trial11. An explicit --movie_list takes precedence.'
        ),
    )

    # Geometry
    p.add_argument(
        "--approx_radius",
        "-a",
        type=int,
        default=defaults.approx_radius,
        help="Approximate radius of chambers in pixels",
    )

    p.add_argument(
        "--chamber_columns",
        "-c",
        type=int,
        default=defaults.chamber_columns,
        help="Number of chamber columns",
    )

    p.add_argument(
        "--chamber_rows",
        "-r",
        type=int,
        default=defaults.chamber_rows,
        help="Number of chamber rows",
    )

    p.add_argument(
        "--crop_dim_2k",
        "-d",
        type=int,
        default=defaults.crop_dim_2k,
        help="Crop dimension in pixels for the supported standard recordings",
    )

    p.add_argument('--chamber_calibration_json', type=str, default=defaults.chamber_calibration_json,
                   help='Optional measured chamber-center calibration for the supported standard recording sizes. The built-in six-chamber geometry is used otherwise.')
    # Detection settings
    p.add_argument(
        "--frame_avg",
        type=int,
        default=defaults.frame_avg,
        help="Number of frames to average for circle detection",
    )

    p.add_argument(
        "--down_factor",
        type=int,
        default=defaults.down_factor,
        help="Factor to downsample frames for circle detection",
    )

    # Parallelism
    p.add_argument(
        "--num_workers",
        type=int,
        default=defaults.num_workers,
        help="Number of parallel processes for movie processing",
    )

    p.add_argument(
        "--max_workers",
        type=int,
        default=defaults.max_workers,
        help='Maximum simultaneous chamber encoders per shared-source FFmpeg graph.',
    )

    p.add_argument(
        "--native_threads",
        type=int,
        default=defaults.native_threads,
        help=(
            'Native-library threads per Python process for OpenCV/OpenMP/BLAS (default: 1).'
        ),
    )

    p.add_argument(
        "--ffmpeg_threads",
        type=int,
        default=defaults.ffmpeg_threads,
        help="Threads per FFmpeg decoder/filter/encoder pool, not a subprocess-wide CPU cap (default: 1).",
    )

    # Monitoring / robustness
    p.add_argument(
        "--heartbeat_interval",
        type=int,
        default=defaults.heartbeat_interval,
        help="Heartbeat interval for monitoring processes in seconds",
    )

    p.add_argument(
        "--timeout",
        type=int,
        default=defaults.timeout,
        help="Timeout for cropping a single chamber in seconds",
    )

    p.add_argument(
        "--partial_cleanup_min_age_sec",
        type=int,
        default=defaults.partial_cleanup_min_age_sec,
        help=(
            "Only delete stale .partial crop files older than this many seconds. "
            "This prevents one swarm subjob from deleting another subjob's active ffmpeg temp file. "
            "Default: 3600."
        ),
    )

    p.add_argument(
        "--crop_crf",
        "--crf",
        type=int,
        default=defaults.crop_crf,
        help=(
            "CRF quality setting for cropped chamber MP4s (lower is higher quality, larger files). " 
            "Default: 12."
        ),
    )

    p.add_argument(
        "--max_retries",
        type=int,
        default=defaults.max_retries,
        help="Max retry attempts for incomplete movies (default: 10).",
    )

    # Logging
    p.add_argument(
        "--log_level",
        type=str,
        default=defaults.log_level,
        help="Logging level (DEBUG, INFO, WARNING, ERROR). Default: INFO.",
    )

    return add_monitor_arguments(p)


def _config_from_args(args: argparse.Namespace) -> AutocropperConfig:
    """Build Autocropper runtime options from parsed arguments."""
    return AutocropperConfig(
        input_dir=str(args.input_dir),
        output_dir=str(args.output_dir) if args.output_dir else None,
        exclusions_csv=str(args.exclusions_csv) if args.exclusions_csv else None,
        require_exclusions=bool(args.require_exclusions),
        recording_database_dir=str(args.recording_database_dir) if args.recording_database_dir else None,
        recording_database_csv=str(args.recording_database_csv) if args.recording_database_csv else None,
        skip_labeling=bool(args.skip_labeling),
        file_filter=str(args.file_filter) if args.file_filter else None,
        movie_list=str(args.movie_list) if args.movie_list else None,
        trial_number=int(args.trial_number) if args.trial_number is not None else None,
        max_retries=int(args.max_retries),
        approx_radius=int(args.approx_radius),
        chamber_columns=int(args.chamber_columns),
        chamber_rows=int(args.chamber_rows),
        crop_dim_2k=int(args.crop_dim_2k),
        chamber_calibration_json=args.chamber_calibration_json,
        frame_avg=int(args.frame_avg),
        down_factor=int(args.down_factor),
        num_workers=int(args.num_workers),
        max_workers=int(args.max_workers),
        native_threads=max(1, int(args.native_threads)),
        ffmpeg_threads=max(1, int(args.ffmpeg_threads)),
        heartbeat_interval=int(args.heartbeat_interval),
        timeout=int(args.timeout),
        partial_cleanup_min_age_sec=max(0, int(args.partial_cleanup_min_age_sec)),
        crop_crf=int(args.crop_crf),
        log_level=str(args.log_level),

        # Registry
        registry_enabled=bool(args.registry_enabled),
        registry_csv=str(args.registry_csv) if args.registry_csv else None,
        registry_lock_timeout_sec=int(args.registry_lock_timeout_sec),
        registry_lock_stale_sec=int(args.registry_lock_stale_sec),
    )


@monitored_cli("autocropper")
def main(argv: list[str] | None = None) -> None:
    """Parse runtime options and run Autocropper."""
    parser = build_arg_parser()
    argv_list = list(argv) if argv is not None else sys.argv[1:]

    args = parse_cli_args(parser, argv_list)
    record_settings(args)
    apply_main_dir_overrides(args, stage="autocropper")

    if not args.input_dir:
        raise SystemExit(
            "Missing --input_dir. Provide --input_dir explicitly, or use --main_dir for the standard project folders."
        )

    # Expand home-directory notation; relative paths retain current-directory semantics.
    if args.output_dir:
        args.output_dir = str(Path(args.output_dir).expanduser())
    if args.input_dir:
        args.input_dir = str(Path(args.input_dir).expanduser())
    if args.exclusions_csv:
        args.exclusions_csv = str(Path(args.exclusions_csv).expanduser())
    if args.movie_list:
        args.movie_list = str(Path(args.movie_list).expanduser())
    if args.recording_database_dir:
        args.recording_database_dir = str(Path(args.recording_database_dir).expanduser())
    if args.recording_database_csv:
        args.recording_database_csv = str(Path(args.recording_database_csv).expanduser())
    if args.registry_csv:
        args.registry_csv = str(Path(args.registry_csv).expanduser())

    cfg = _config_from_args(args)
    run_autocropper(cfg=cfg, args=args)


if __name__ == "__main__":
    main()
