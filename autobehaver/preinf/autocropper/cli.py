"""CLI for autocropper.

This stage is currently tailored to the lab's acquisition geometry.

"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Dict

import yaml

from autobehaver.argparse_utils import add_bool_pair
from autobehaver.pipeline_layout import apply_main_dir_overrides, config_main_dir_default

from .config import AutocropperConfig
from .core import run_autocropper


def _load_config_defaults(config_path: str) -> Dict[str, Any]:
    """Load defaults from a global YAML config file.

    Autocropper reads:
      - the ``autocropper:`` section (stage-specific)
      - the top-level ``registry:`` section (project-wide registry settings)

    Stage keys override registry defaults when they overlap.
    """
    cfg_path = Path(config_path).expanduser().resolve()
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")

    with cfg_path.open("r") as f:
        data = yaml.safe_load(f) or {}

    # Resolve relative paths relative to config file directory
    base_dir = cfg_path.parent

    resolved: Dict[str, Any] = {}

    global_main_dir = config_main_dir_default(data, base_dir)
    if global_main_dir:
        resolved["main_dir"] = global_main_dir

    # --- Project registry (global) ---
    reg = data.get("registry", {})
    if reg is None:
        reg = {}
    if isinstance(reg, dict):
        if "enabled" in reg:
            resolved["registry_enabled"] = bool(reg.get("enabled"))
        if "csv" in reg and isinstance(reg.get("csv"), str) and str(reg.get("csv")).strip() != "":
            p = Path(str(reg.get("csv")).strip())
            if not p.is_absolute():
                p = (base_dir / p).resolve()
            resolved["registry_csv"] = str(p)
        if "lock_timeout_sec" in reg and reg.get("lock_timeout_sec") is not None:
            try:
                resolved["registry_lock_timeout_sec"] = int(reg.get("lock_timeout_sec"))
            except Exception:
                pass
        if "lock_stale_sec" in reg and reg.get("lock_stale_sec") is not None:
            try:
                resolved["registry_lock_stale_sec"] = int(reg.get("lock_stale_sec"))
            except Exception:
                pass

    section = data.get("autocropper", {})
    if section is None:
        section = {}

    if not isinstance(section, dict):
        raise ValueError("Expected 'autocropper' section in config to be a mapping (dict)")

    path_keys = {
        "main_dir",
        "input_dir",
        "output_dir",
        "exclusions_csv",
        "recording_database_dir",
        "recording_database_csv",
        "model_path",
        "norm_params_path",
        "movie_list",
        "registry_csv",
    }

    # Stage section overrides registry defaults when present.
    for k, v in section.items():
        if k in path_keys and isinstance(v, str) and v.strip() != "":
            p = Path(v.strip())
            if not p.is_absolute():
                p = (base_dir / p).resolve()
            resolved[k] = str(p)
        else:
            resolved[k] = v

    return resolved


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Auto-crop fly chamber videos")

    # Global config
    p.add_argument(
        "--config",
        type=str,
        default=None,
        help=(
            "Path to global pipeline YAML config (autocropper reads the 'autocropper:' section). "
            "CLI args override config defaults."
        ),
    )

    # Project registry (single CSV at project root)
    add_bool_pair(
        p,
        "registry_enabled",
        default=False,
        help=(
            "If enabled, update the project-wide registry CSV after autocropper finishes. "
            "Default: disabled (commonly enabled via config template)."
        ),
        dest="registry_enabled",
    )

    p.add_argument(
        "--registry_csv",
        type=str,
        default=None,
        help=(
            "Path to the project registry CSV. If omitted and registry_enabled is True, "
            "defaults to <project_root>/registry.csv (project_root inferred from input/output dirs)."
        ),
    )

    p.add_argument(
        "--registry_lock_timeout_sec",
        type=int,
        default=120,
        help="Seconds to wait for registry lock before failing (default 120).",
    )

    p.add_argument(
        "--registry_lock_stale_sec",
        type=int,
        default=60 * 60,
        help="Consider registry lock stale after this many seconds (default 3600).",
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
            "Required unless provided via --config."
        ),
    )

    p.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help=(
            "Directory to write cropped chamber movies and stage artifacts. "
            "If omitted, cropped chamber movies go under <input_dir>/cropped_chambers and stage artifacts stay under <input_dir>."
        ),
    )

    # Exclusions (moviechecker)
    p.add_argument(
        "--exclusions_csv",
        type=str,
        default=None,
        help=(
            "Path to exclusions.csv produced by moviechecker. If omitted, defaults to <input_dir>/exclusions.csv."
        ),
    )

    add_bool_pair(
        p,
        "require_exclusions",
        default=True,
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
        default=None,
        help=(
            "Directory containing recording_database_*.csv backups created by the recording script. "
            "If omitted, defaults to <input_dir>."
        ),
    )

    p.add_argument(
        "--recording_database_csv",
        type=str,
        default=None,
        help=(
            "Optional explicit path to a recording database CSV to use instead of auto-selecting the latest one "
            "from --recording_database_dir."
        ),
    )

    add_bool_pair(
        p,
        "skip_labeling",
        default=False,
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
        default=None,
        help="Only process movies whose filename contains this substring (case-insensitive).",
    )

    p.add_argument(
        "--movie_list",
        type=str,
        default=None,
        help=(
            "Optional text file listing which movies to process (one filename or path per line). "
            "If provided, takes precedence over --file_filter."
        ),
    )

    p.add_argument(
        "--trial_number",
        "--trial",
        type=int,
        default=None,
        help=(
            "Only process movies whose filename has an exact _trialN_ token. "
            "This is safer than --file_filter trial1 because trial1 also matches trial10/trial11/etc."
        ),
    )

    # Geometry
    p.add_argument(
        "--approx_radius",
        "-a",
        type=int,
        default=265,
        help="Approximate radius of chambers in pixels",
    )

    p.add_argument(
        "--chamber_columns",
        "-c",
        type=int,
        default=3,
        help="Number of chamber columns",
    )

    p.add_argument(
        "--chamber_rows",
        "-r",
        type=int,
        default=2,
        help="Number of chamber rows",
    )

    p.add_argument(
        "--crop_dim_2k",
        "-d",
        type=int,
        default=600,
        help=(
            "Crop dimension for 2K resolution videos in pixels - if using 4k, "
            "please enter half the resolution"
        ),
    )

    # Validation flags
    add_bool_pair(
        p,
        "validate_chambers",
        default=True,
        help="Enable chamber validation using CNN to detect empty chambers.",
        dest="validate_chambers",
    )

    add_bool_pair(
        p,
        "skip_validation",
        default=False,
        help="Skip chamber validation and crop all detected chambers (except excluded ones).",
        dest="skip_validation",
    )

    p.add_argument(
        "--model_path",
        "-m",
        default=None,
        help=(
            "Path to the trained CNN model for chamber validation. "
            "If omitted, uses the packaged default model."
        ),
    )

    p.add_argument(
        "--norm_params_path",
        "-n",
        default=None,
        help=(
            "Path to the normalization parameters JSON file. "
            "If omitted, uses the packaged default JSON."
        ),
    )

    # Detection settings
    p.add_argument(
        "--frame_avg",
        type=int,
        default=10,
        help="Number of frames to average for circle detection",
    )

    p.add_argument(
        "--down_factor",
        type=int,
        default=2,
        help="Factor to downsample frames for circle detection",
    )

    # Parallelism
    p.add_argument(
        "--num_workers",
        type=int,
        default=2,
        help="Number of parallel processes for movie processing",
    )

    p.add_argument(
        "--max_workers",
        type=int,
        default=6,
        help="Max number of chamber-crop threads per movie worker",
    )

    p.add_argument(
        "--native_threads",
        type=int,
        default=1,
        help=(
            "Native-library threads per Python process for OpenCV/OpenMP/BLAS. "
            "Default: 1, which is safest for swarm/Slurm runs."
        ),
    )

    p.add_argument(
        "--ffmpeg_threads",
        type=int,
        default=1,
        help="ffmpeg/libx264 threads per chamber crop subprocess (default: 1).",
    )

    # Monitoring / robustness
    p.add_argument(
        "--heartbeat_interval",
        type=int,
        default=10,
        help="Heartbeat interval for monitoring processes in seconds",
    )

    p.add_argument(
        "--timeout",
        type=int,
        default=600,
        help="Timeout for cropping a single chamber in seconds",
    )

    p.add_argument(
        "--partial_cleanup_min_age_sec",
        type=int,
        default=60 * 60,
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
        default=12,
        help=(
            "CRF quality setting for cropped chamber MP4s (lower is higher quality, larger files). " 
            "Default: 12."
        ),
    )

    p.add_argument(
        "--max_retries",
        type=int,
        default=10,
        help="Max retry attempts for incomplete movies (default: 10).",
    )

    # Logging
    p.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        help="Logging level (DEBUG, INFO, WARNING, ERROR). Default: INFO.",
    )

    return p


def _config_from_args(args: argparse.Namespace) -> AutocropperConfig:
    # `skip_validation` is an override: it forces validation off.
    skip_validation = bool(getattr(args, "skip_validation", False)) or (not bool(args.validate_chambers))

    return AutocropperConfig(
        input_dir=str(args.input_dir),
        output_dir=str(args.output_dir) if args.output_dir else None,
        exclusions_csv=str(args.exclusions_csv) if args.exclusions_csv else None,
        require_exclusions=bool(getattr(args, "require_exclusions", True)),
        recording_database_dir=str(args.recording_database_dir) if args.recording_database_dir else None,
        recording_database_csv=str(args.recording_database_csv) if args.recording_database_csv else None,
        skip_labeling=bool(getattr(args, "skip_labeling", False)),
        file_filter=str(args.file_filter) if args.file_filter else None,
        movie_list=str(args.movie_list) if args.movie_list else None,
        trial_number=int(args.trial_number) if getattr(args, "trial_number", None) is not None else None,
        max_retries=int(getattr(args, "max_retries", 10)),
        approx_radius=int(args.approx_radius),
        chamber_columns=int(args.chamber_columns),
        chamber_rows=int(args.chamber_rows),
        crop_dim_2k=int(args.crop_dim_2k),
        frame_avg=int(args.frame_avg),
        down_factor=int(args.down_factor),
        num_workers=int(args.num_workers),
        max_workers=int(args.max_workers),
        native_threads=max(1, int(getattr(args, "native_threads", 1))),
        ffmpeg_threads=max(1, int(getattr(args, "ffmpeg_threads", 1))),
        heartbeat_interval=int(args.heartbeat_interval),
        timeout=int(args.timeout),
        partial_cleanup_min_age_sec=max(0, int(getattr(args, "partial_cleanup_min_age_sec", 3600))),
        crop_crf=int(getattr(args, "crop_crf", 12)),
        skip_validation=skip_validation,
        model_path=str(args.model_path) if args.model_path else None,
        norm_params_path=str(args.norm_params_path) if args.norm_params_path else None,
        log_level=str(args.log_level),

        # Registry
        registry_enabled=bool(getattr(args, "registry_enabled", False)),
        registry_csv=str(getattr(args, "registry_csv", "")) if getattr(args, "registry_csv", None) else None,
        registry_lock_timeout_sec=int(getattr(args, "registry_lock_timeout_sec", 120)),
        registry_lock_stale_sec=int(getattr(args, "registry_lock_stale_sec", 60 * 60)),
    )


def main(argv: list[str] | None = None) -> None:
    argv_list = list(argv) if argv is not None else sys.argv[1:]

    # Pre-parse config path (so we can apply defaults before full parse).
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", type=str, default=None)
    pre_args, _ = pre.parse_known_args(argv_list)

    parser = build_arg_parser()

    # Apply config defaults (if provided)
    if pre_args.config:
        defaults = _load_config_defaults(pre_args.config)
        # Only apply defaults for keys that correspond to known argparse destinations
        valid_dests = {a.dest for a in parser._actions}
        filtered = {k: v for k, v in defaults.items() if k in valid_dests}
        parser.set_defaults(**filtered)

    args = parser.parse_args(argv_list)
    apply_main_dir_overrides(args, stage="autocropper")

    if not getattr(args, "input_dir", None):
        raise SystemExit(
            "Missing --input_dir. Provide --input_dir explicitly, or use --config with an 'autocropper:' section that sets input_dir."
        )

    # If output_dir is provided and it's relative, resolve relative to CWD.
    # (Config loader already resolves relative paths relative to config file.)
    if getattr(args, "output_dir", None):
        args.output_dir = str(Path(args.output_dir).expanduser())
    if getattr(args, "input_dir", None):
        args.input_dir = str(Path(args.input_dir).expanduser())
    if getattr(args, "exclusions_csv", None):
        args.exclusions_csv = str(Path(args.exclusions_csv).expanduser())
    if getattr(args, "movie_list", None):
        args.movie_list = str(Path(args.movie_list).expanduser())
    if getattr(args, "recording_database_dir", None):
        args.recording_database_dir = str(Path(args.recording_database_dir).expanduser())
    if getattr(args, "recording_database_csv", None):
        args.recording_database_csv = str(Path(args.recording_database_csv).expanduser())
    if getattr(args, "registry_csv", None):
        args.registry_csv = str(Path(args.registry_csv).expanduser())

    cfg = _config_from_args(args)
    run_autocropper(cfg=cfg, args=args)


if __name__ == "__main__":
    main()
