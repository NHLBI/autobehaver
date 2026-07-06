"""CLI for the keyprenorm stage.

Keyprenorm takes positional_calcs CSV output from keypre and produces normalized
HDF5 files plus normalization parameters.

The global config file is expected to contain a top-level ``keyprenorm:``
section. CLI arguments override config defaults.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import yaml

from autobehaver.argparse_utils import add_bool_pair
from autobehaver.pipeline_layout import apply_main_dir_overrides, config_main_dir_default

from .config import KeypreNormConfig
from .core import run_keyprenorm


def _load_config_defaults(config_path: str) -> Dict[str, Any]:
    """Load defaults from a global YAML config file.

    keyprenorm reads:
      - the ``keyprenorm:`` section (stage-specific)
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

    section = data.get("keyprenorm", {})
    if section is None:
        section = {}
    if not isinstance(section, dict):
        raise ValueError("Expected 'keyprenorm' section in config to be a mapping (dict)")

    path_keys = {
        "main_dir",
        "input_dir",
        "output_dir",
        "labels_dir",
        "norm_params",
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
    p = argparse.ArgumentParser(description="Normalize positional_calcs CSV files and write H5 outputs")

    # Global pipeline config
    p.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to global pipeline YAML config (reads the 'keyprenorm:' section).",
    )


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
        default=None,
        help=(
            "Optional stage output root directory. When provided, normalized H5 files and the default norm_params.csv are written directly into this folder. "
            "If omitted, outputs are written under <input_dir>/normalization_h5 and <input_dir>/norm_params/."
        ),
    )

    # Input discovery
    p.add_argument(
        "--pattern",
        type=str,
        default="*_positional_calcs.csv",
        help="Glob pattern for positional_calcs CSV files (default *_positional_calcs.csv)",
    )

    p.add_argument(
        "--file_filter",
        type=str,
        default=None,
        help="Optional substring filter applied to filenames (case-insensitive).",
    )

    p.add_argument(
        "--labels_dir",
        type=str,
        default=None,
        help="Optional directory of labels to copy into output folder structure.",
    )

    p.add_argument(
        "--norm_params",
        type=str,
        default=None,
        help="Optional path to norm params to load (otherwise computed when --calculate_norm).",
    )

    add_bool_pair(
        p,
        "calculate_norm",
        default=False,
        help="Calculate normalization params from the inputs (and write them).",
        dest="calculate_norm",
    )

    add_bool_pair(
        p,
        "norm_only",
        default=False,
        help="Only compute and write norm params (do not write per-trial H5 files).",
        dest="norm_only",
    )

    # Overwrite controls
    add_bool_pair(
        p,
        "clobber",
        default=False,
        help="Overwrite existing outputs (default: disabled / safe).",
        dest="clobber",
    )

    p.add_argument(
        "--num_workers",
        type=int,
        default=14,
        help="Number of parallel workers (default 14)",
    )

    p.add_argument(
        "--edge_skip",
        type=int,
        default=5,
        help="Frames to skip at start/end (default 5)",
    )

    p.add_argument(
        "--max_rows",
        type=int,
        default=64750,
        help="Max rows per trial (0 for variable-length mode). Default 64750.",
    )

    add_bool_pair(
        p,
        "require_full_length",
        default=False,
        help="Require exactly max_rows rows for fixed-length mode.",
        dest="require_full_length",
    )

    add_bool_pair(
        p,
        "size_check",
        default=False,
        help="Run optional size checking utilities.",
        dest="size_check",
    )

    p.add_argument(
        "--batch_size",
        type=int,
        default=10,
        help="Batch size for processing (default 10)",
    )

    p.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        help="Logging level (default INFO)",
    )

    p.add_argument(
        "--log_file",
        type=str,
        default=None,
        help="Optional log file path",
    )

    # Registry
    add_bool_pair(
        p,
        "registry_enabled",
        default=False,
        help="Enable project registry integration (discover inputs + upsert outputs).",
        dest="registry_enabled",
    )

    p.add_argument(
        "--registry_csv",
        type=str,
        default=None,
        help="Path to the project registry CSV (usually at project root).",
    )

    p.add_argument(
        "--registry_lock_timeout_sec",
        type=int,
        default=120,
        help="Seconds to wait for registry lock before failing (default 120)",
    )

    p.add_argument(
        "--registry_lock_stale_sec",
        type=int,
        default=60 * 60,
        help="Consider registry lock stale after this many seconds (default 3600)",
    )

    return p


def _config_from_args(args: argparse.Namespace) -> KeypreNormConfig:
    clobber = bool(getattr(args, "clobber", False))

    return KeypreNormConfig(
        input_dir=str(args.input_dir),
        output_dir=str(args.output_dir) if args.output_dir else None,
        pattern=str(args.pattern),
        file_filter=str(args.file_filter) if args.file_filter else None,
        labels_dir=str(args.labels_dir) if args.labels_dir else None,
        norm_params=str(args.norm_params) if args.norm_params else None,
        calculate_norm=bool(getattr(args, "calculate_norm", False)),
        norm_only=bool(getattr(args, "norm_only", False)),
        clobber=clobber,
        num_workers=int(args.num_workers),
        edge_skip=int(args.edge_skip),
        max_rows=int(args.max_rows),
        require_full_length=bool(getattr(args, "require_full_length", False)),
        size_check=bool(getattr(args, "size_check", False)),
        batch_size=int(args.batch_size),
        log_level=str(args.log_level),
        log_file=str(args.log_file) if args.log_file else None,
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
        for k in ("input_dir", "output_dir", "labels_dir", "norm_params"):
            if k in defaults:
                mapped[k] = defaults[k]

        passthrough = {
            "registry_enabled",
            "registry_csv",
            "registry_lock_timeout_sec",
            "registry_lock_stale_sec",
            "pattern",
            "file_filter",
            "calculate_norm",
            "norm_only",
            "num_workers",
            "edge_skip",
            "max_rows",
            "require_full_length",
            "size_check",
            "batch_size",
            "log_level",
            "log_file",
            "clobber",
        }
        for k in passthrough:
            if k in defaults:
                mapped[k] = defaults[k]

        parser.set_defaults(**mapped)

    args = parser.parse_args(argv_list)
    apply_main_dir_overrides(args, stage="keyprenorm")

    if not args.input_dir:
        raise SystemExit(
            "Missing --input_dir. Provide --input_dir explicitly, or use --config with a 'keyprenorm:' section that sets input_dir."
        )

    cfg = _config_from_args(args)
    run_keyprenorm(cfg=cfg, args=args)
