"""Define command-line options for deriving KeyPre features from DeepLabCut CSVs."""

from __future__ import annotations

import argparse
import sys
from typing import Optional, Sequence


from autobehaver.argparse_utils import add_bool_pair, parse_cli_args
from autobehaver.resource_monitor import add_monitor_arguments, monitored_cli, record_settings
from autobehaver.pipeline_layout import apply_main_dir_overrides

from ..execution import add_memory_arguments
from .config import KeypreConfig
from .core import run_keypre


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the KeyPre command-line parser with runtime defaults."""
    defaults = KeypreConfig(input_dir="")
    parser = argparse.ArgumentParser(
        description="Process DLC CSV files to generate positional calculations (keypre)."
    )


    # Project registry (optional)
    add_bool_pair(
        parser,
        "registry_enabled",
        default=defaults.registry_enabled,
        help="Enable project registry integration (discover inputs + upsert outputs).",
        dest="registry_enabled",
    )
    parser.add_argument(
        "--registry_csv",
        type=str,
        default=defaults.registry_csv,
        help="Path to project registry CSV (usually at project root).",
    )
    parser.add_argument(
        "--registry_lock_timeout_sec",
        type=int,
        default=defaults.registry_lock_timeout_sec,
        help="Timeout (seconds) waiting for the registry writer lock.",
    )
    parser.add_argument(
        "--registry_lock_stale_sec",
        type=int,
        default=defaults.registry_lock_stale_sec,
        help="Legacy option retained for compatibility; locks release when their process exits.",
    )


    parser.add_argument(
        "--main_dir",
        type=str,
        default=None,
        help=(
            "Optional project root containing the standardized numbered pipeline folders. "
            "For keypre this fills input_dir=<main_dir>/3_dlc and output_dir=<main_dir>/4_keypoint_features."
        ),
    )

    parser.add_argument(
        "--input_dir",
        dest="input_dir",
        type=str,
        default=None,
        help=(
            "Directory containing DLC CSV files to process. If a 'DLC' subfolder exists, "
            "CSVs are read from there."
        ),
    )

    parser.add_argument(
        "--output_dir",
        dest="output_dir",
        type=str,
        default=defaults.output_dir,
        help=(
            "Output directory for processed CSV files. "
            "Default: <input>/positional_calcs (or <input>/raw_positional_calcs when --raw_only)."
        ),
    )

    parser.add_argument("--file_list", help="Explicit batch list of logical chamber recording stems")
    parser.add_argument(
        "--file_filter",
        type=str,
        default=defaults.file_filter,
        help="Only process input CSVs whose filename contains this substring (case-insensitive).",
    )

    parser.add_argument(
        "--num_workers",
        type=int,
        default=defaults.num_workers,
        help="Maximum parallel files; 0 = automatic within the CPU/RAM budget (default).",
    )

    # Overwrite controls
    add_bool_pair(
        parser,
        "clobber",
        default=defaults.clobber,
        help="Overwrite existing outputs.",
        dest="clobber",
    )

    parser.add_argument(
        "--movie_dim",
        type=int,
        default=defaults.movie_dim,
        help="Movie frame dimension in pixels (assumes square frames). Default: 600.",
    )

    add_bool_pair(
        parser,
        "raw_only",
        default=defaults.raw_only,
        help=(
            "Export only raw DLC keypoints (x/y/likelihood per body part; excludes Front/Back) "
            "and skip all feature engineering/corrections."
        ),
        dest="raw_only",
    )

    add_bool_pair(
        parser,
        "dry_run",
        default=defaults.dry_run,
        help="Print discovered input files and planned output directory, then exit.",
        dest="dry_run",
    )

    parser.add_argument(
        "--log_level",
        type=str,
        default=defaults.log_level,
        help="Logging level (e.g. DEBUG, INFO, WARNING, ERROR). Default: INFO.",
    )

    return add_monitor_arguments(add_memory_arguments(parser))


def _config_from_args(args: argparse.Namespace) -> KeypreConfig:
    """Build KeyPre runtime options from parsed arguments."""
    return KeypreConfig(
        input_dir=str(args.input_dir),
        output_dir=str(args.output_dir) if args.output_dir else None,
        file_filter=str(args.file_filter) if args.file_filter else None,
        file_list=args.file_list,
        num_workers=int(args.num_workers),
        memory_gb=args.memory_gb,
        clobber=bool(args.clobber),
        movie_dim=int(args.movie_dim),
        dry_run=bool(args.dry_run),
        log_level=str(args.log_level),
        raw_only=bool(args.raw_only),
        registry_enabled=bool(args.registry_enabled),
        registry_csv=str(args.registry_csv) if args.registry_csv else None,
        registry_lock_timeout_sec=int(args.registry_lock_timeout_sec),
        registry_lock_stale_sec=int(args.registry_lock_stale_sec),
    )


@monitored_cli("keypre")
def main(argv: Optional[Sequence[str]] = None) -> None:
    """Parse runtime options and run KeyPre."""
    parser = build_arg_parser()
    argv_list = list(argv) if argv is not None else sys.argv[1:]

    args = parse_cli_args(parser, argv_list)
    record_settings(args)
    apply_main_dir_overrides(args, stage="keypre")

    if not args.input_dir:
        raise SystemExit(
            "Missing --input_dir. Provide --input_dir explicitly, or use --main_dir for the standard project folders."
        )

    cfg = _config_from_args(args)
    run_keypre(cfg=cfg, args=args)
