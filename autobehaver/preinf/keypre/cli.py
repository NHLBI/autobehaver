"""CLI for keypre.

The global config file is expected to contain a top-level ``keypre:`` section.
CLI arguments override config defaults.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import yaml

from autobehaver.argparse_utils import add_bool_pair
from autobehaver.pipeline_layout import apply_main_dir_overrides, config_main_dir_default

from .config import KeypreConfig
from .core import run_keypre


def _load_config_defaults(config_path: str) -> Dict[str, Any]:
    """Load defaults from a global YAML config file.

    keypre reads:
      - the ``keypre:`` section (stage-specific)
      - the top-level ``registry:`` section (project-wide registry settings)

    Stage keys override registry defaults when they overlap.
    """
    cfg_path = Path(config_path).expanduser().resolve()
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")

    with cfg_path.open("r") as f:
        data = yaml.safe_load(f) or {}

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

    section = data.get("keypre", {})
    if section is None:
        section = {}
    if not isinstance(section, dict):
        raise ValueError("Expected 'keypre' section in config to be a mapping (dict)")

    path_keys = {
        "main_dir",
        "input_dir",
        "output_dir",
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
    parser = argparse.ArgumentParser(
        description="Process DLC CSV files to generate positional calculations (keypre)."
    )

    # Global pipeline config
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to global pipeline YAML config (reads the 'keypre:' section).",
    )

    # Project registry (optional)
    add_bool_pair(
        parser,
        "registry_enabled",
        default=False,
        help="Enable project registry integration (discover inputs + upsert outputs).",
        dest="registry_enabled",
    )
    parser.add_argument(
        "--registry_csv",
        type=str,
        default=None,
        help="Path to project registry CSV (usually at project root).",
    )
    parser.add_argument(
        "--registry_lock_timeout_sec",
        type=int,
        default=120,
        help="Timeout (seconds) waiting for the registry directory lock.",
    )
    parser.add_argument(
        "--registry_lock_stale_sec",
        type=int,
        default=60 * 60,
        help="Consider the registry lock stale after this many seconds.",
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
        default=None,
        help=(
            "Output directory for processed CSV files. "
            "Default: <input>/positional_calcs (or <input>/raw_positional_calcs when --raw_only)."
        ),
    )

    parser.add_argument(
        "--file_filter",
        type=str,
        default=None,
        help="Only process input CSVs whose filename contains this substring (case-insensitive).",
    )

    parser.add_argument(
        "--num_workers",
        type=int,
        default=15,
        help="Number of worker processes to use for parallel processing.",
    )

    # Overwrite controls
    add_bool_pair(
        parser,
        "clobber",
        default=False,
        help="Overwrite existing outputs.",
        dest="clobber",
    )

    parser.add_argument(
        "--movie_dim",
        type=int,
        default=600,
        help="Movie frame dimension in pixels (assumes square frames). Default: 600.",
    )

    add_bool_pair(
        parser,
        "raw_only",
        default=False,
        help=(
            "Export only raw DLC keypoints (x/y/likelihood per body part; excludes Front/Back) "
            "and skip all feature engineering/corrections."
        ),
        dest="raw_only",
    )

    add_bool_pair(
        parser,
        "dry_run",
        default=False,
        help="Print discovered input files and planned output directory, then exit.",
        dest="dry_run",
    )

    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        help="Logging level (e.g. DEBUG, INFO, WARNING, ERROR). Default: INFO.",
    )

    return parser


def _config_from_args(args: argparse.Namespace) -> KeypreConfig:
    return KeypreConfig(
        input_dir=str(args.input_dir),
        output_dir=str(args.output_dir) if args.output_dir else None,
        file_filter=str(args.file_filter) if args.file_filter else None,
        num_workers=int(args.num_workers),
        clobber=bool(args.clobber),
        movie_dim=int(args.movie_dim),
        dry_run=bool(args.dry_run),
        log_level=str(args.log_level),
        raw_only=bool(getattr(args, "raw_only", False)),
        registry_enabled=bool(getattr(args, "registry_enabled", False)),
        registry_csv=str(args.registry_csv) if getattr(args, "registry_csv", None) else None,
        registry_lock_timeout_sec=int(getattr(args, "registry_lock_timeout_sec", 120)),
        registry_lock_stale_sec=int(getattr(args, "registry_lock_stale_sec", 60 * 60)),
    )


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = build_arg_parser()

    argv_list = list(argv) if argv is not None else sys.argv[1:]

    pre = parser.parse_known_args(argv_list)[0]

    if pre.config:
        defaults = _load_config_defaults(pre.config)

        mapped: Dict[str, Any] = {}
        for k in ("input_dir", "output_dir"):
            if k in defaults:
                mapped[k] = defaults[k]

        passthrough = {
            "registry_enabled",
            "registry_csv",
            "registry_lock_timeout_sec",
            "registry_lock_stale_sec",
            "num_workers",
            "movie_dim",
            "raw_only",
            "dry_run",
            "log_level",
            "file_filter",
            "clobber",
        }
        for k in passthrough:
            if k in defaults:
                mapped[k] = defaults[k]

        parser.set_defaults(**mapped)

    args = parser.parse_args(argv_list)
    apply_main_dir_overrides(args, stage="keypre")

    if not args.input_dir:
        raise SystemExit(
            "Missing --input_dir. Provide --input_dir explicitly, or use --config with a 'keypre:' section that sets input_dir."
        )

    cfg = _config_from_args(args)
    run_keypre(cfg=cfg, args=args)
