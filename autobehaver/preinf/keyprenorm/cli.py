"""Command-line options for normalizing keypoint feature CSV files."""

from __future__ import annotations

import argparse
import sys
from typing import Optional, Sequence


from autobehaver.argparse_utils import add_bool_pair, parse_cli_args
from autobehaver.resource_monitor import add_monitor_arguments, monitored_cli, record_settings
from autobehaver.pipeline_layout import apply_main_dir_overrides

from ..execution import add_memory_arguments
from .config import KeypreNormConfig
from .core import run_keyprenorm


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the KeyPreNorm command-line parser with runtime defaults."""
    defaults = KeypreNormConfig(input_dir="")
    p = argparse.ArgumentParser(description="Normalize positional_calcs CSV files and write H5 outputs")


    p.add_argument(
        "--main_dir",
        type=str,
        default=None,
        help=(
            "Optional project root containing the standardized numbered pipeline folders. "
            "For keyprenorm this fills input_dir=<main_dir>/4_keypoint_features and output_dir=<main_dir>/5_normalized_features."
        ),
    )

    p.add_argument(
        "--input_dir",
        type=str,
        default=None,
        help="Directory containing positional_calcs CSV files",
    )

    p.add_argument(
        "--output_dir",
        type=str,
        default=defaults.output_dir,
        help=(
            "Optional stage output root directory. When provided, normalized H5 files and the default norm_params.csv are written directly into this folder. "
            "If omitted, outputs are written under <input_dir>/normalization_h5 and <input_dir>/norm_params/."
        ),
    )

    # Input discovery
    p.add_argument(
        "--pattern",
        type=str,
        default=defaults.pattern,
        help="Glob pattern for positional_calcs CSV files (default *_positional_calcs.csv)",
    )

    p.add_argument("--file_list", help="Explicit batch list of logical chamber recording stems")
    p.add_argument(
        "--file_filter",
        type=str,
        default=defaults.file_filter,
        help="Optional substring filter applied to filenames (case-insensitive).",
    )

    p.add_argument(
        "--labels_dir",
        type=str,
        default=defaults.labels_dir,
        help="Optional directory of labels to copy into output folder structure.",
    )

    p.add_argument(
        "--norm_params",
        type=str,
        default=defaults.norm_params,
        help='Path to normalization parameters to load; otherwise calculate them with --calculate_norm.',
    )

    add_bool_pair(
        p,
        "calculate_norm",
        default=defaults.calculate_norm,
        help="Calculate normalization params from the inputs (and write them).",
        dest="calculate_norm",
    )

    add_bool_pair(
        p,
        "norm_only",
        default=defaults.norm_only,
        help="Only compute and write norm params (do not write per-trial H5 files).",
        dest="norm_only",
    )

    # Overwrite controls
    add_bool_pair(
        p,
        "clobber",
        default=defaults.clobber,
        help='Overwrite existing normalized outputs (default: disabled).',
        dest="clobber",
    )

    p.add_argument(
        "--num_workers",
        type=int,
        default=defaults.num_workers,
        help="Maximum parallel files; 0 = automatic within the CPU/RAM budget (default).",
    )

    p.add_argument(
        "--edge_skip",
        type=int,
        default=defaults.edge_skip,
        help="Preprocessing edge rows to omit at each end (default 5); original movie frames are retained in metadata.",
    )

    p.add_argument(
        "--batch_size",
        type=int,
        default=defaults.batch_size,
        help='CSV files per statistics-worker batch when calculating parameters (default: 10).',
    )

    p.add_argument(
        "--log_level",
        type=str,
        default=defaults.log_level,
        help="Logging level (default INFO)",
    )

    p.add_argument(
        "--log_file",
        type=str,
        default=defaults.log_file,
        help="Optional log file path",
    )

    # Registry
    add_bool_pair(
        p,
        "registry_enabled",
        default=defaults.registry_enabled,
        help="Enable project registry integration (discover inputs + upsert outputs).",
        dest="registry_enabled",
    )

    p.add_argument(
        "--registry_csv",
        type=str,
        default=defaults.registry_csv,
        help="Path to the project registry CSV (usually at project root).",
    )

    p.add_argument(
        "--registry_lock_timeout_sec",
        type=int,
        default=defaults.registry_lock_timeout_sec,
        help="Seconds to wait for registry lock before failing (default 120)",
    )

    p.add_argument(
        "--registry_lock_stale_sec",
        type=int,
        default=defaults.registry_lock_stale_sec,
        help="Legacy option retained for compatibility; locks release when their process exits.",
    )

    return add_monitor_arguments(add_memory_arguments(p))


def _config_from_args(args: argparse.Namespace) -> KeypreNormConfig:
    """Build KeyPreNorm runtime options from parsed arguments."""
    clobber = bool(args.clobber)

    return KeypreNormConfig(
        input_dir=str(args.input_dir),
        output_dir=str(args.output_dir) if args.output_dir else None,
        pattern=str(args.pattern),
        file_filter=str(args.file_filter) if args.file_filter else None,
        file_list=args.file_list,
        labels_dir=str(args.labels_dir) if args.labels_dir else None,
        norm_params=str(args.norm_params) if args.norm_params else None,
        calculate_norm=bool(args.calculate_norm),
        norm_only=bool(args.norm_only),
        clobber=clobber,
        num_workers=int(args.num_workers),
        memory_gb=args.memory_gb,
        edge_skip=int(args.edge_skip),
        batch_size=int(args.batch_size),
        log_level=str(args.log_level),
        log_file=str(args.log_file) if args.log_file else None,
        registry_enabled=bool(args.registry_enabled),
        registry_csv=str(args.registry_csv) if args.registry_csv else None,
        registry_lock_timeout_sec=int(args.registry_lock_timeout_sec),
        registry_lock_stale_sec=int(args.registry_lock_stale_sec),
    )


@monitored_cli("keyprenorm")
def main(argv: Optional[Sequence[str]] = None) -> None:
    """Parse runtime options and run KeyPreNorm."""
    parser = build_arg_parser()
    argv_list = list(argv) if argv is not None else sys.argv[1:]

    args = parse_cli_args(parser, argv_list)
    record_settings(args)
    apply_main_dir_overrides(args, stage="keyprenorm")

    if not args.input_dir:
        raise SystemExit(
            "Missing --input_dir. Provide --input_dir explicitly, or use --main_dir for the standard project folders."
        )

    cfg = _config_from_args(args)
    run_keyprenorm(cfg=cfg, args=args)
