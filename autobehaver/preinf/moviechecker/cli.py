"""CLI for the manual movie checker.

The movie checker is designed to be run locally (with a display) and writes an
``exclusions.csv`` file in the full-movies folder.
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

from .core import MovieCheckerApp


def _load_config_defaults(config_path: str) -> Dict[str, Any]:
    """Load moviechecker defaults from the global pipeline YAML config."""

    cfg_path = Path(config_path).expanduser().resolve()
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")

    with cfg_path.open("r") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError("Pipeline config must be a YAML mapping (dict) at the top level")

    base_dir = cfg_path.parent
    resolved: Dict[str, Any] = {}

    global_main_dir = config_main_dir_default(data, base_dir)
    if global_main_dir:
        resolved["main_dir"] = global_main_dir

    section = data.get("moviechecker", {})
    if section is None:
        section = {}
    if not isinstance(section, dict):
        raise ValueError("Expected 'moviechecker' section in config to be a mapping (dict)")

    path_keys = {"main_dir", "input_dir", "exclusions_csv"}
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
    p = argparse.ArgumentParser(description="Review full-frame movies and mark chamber exclusions")

    p.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to global pipeline YAML config (reads the 'moviechecker:' section).",
    )

    p.add_argument(
        "--input_dir",
        "-i",
        required=False,
        help="Directory containing full-frame movies (flat folder; no subfolders expected).",
    )

    p.add_argument(
        "--main_dir",
        type=str,
        default=None,
        help=(
            "Optional project root containing the standardized numbered pipeline folders. "
            "For moviechecker this fills input_dir=<main_dir>/1_full_movies and exclusions_csv=<main_dir>/1_full_movies/exclusions.csv."
        ),
    )

    p.add_argument(
        "--exclusions_csv",
        default=None,
        help=(
            "Path to exclusions.csv. Default: <input_dir>/exclusions.csv (recommended). "
            "The file is intentionally CSV so you can manually edit it if needed."
        ),
    )

    add_bool_pair(
        p,
        "review_all",
        default=False,
        help=(
            "Do not skip movies already listed in exclusions.csv. "
            "By default, the tool skips already-reviewed movie pairs."
        ),
    )

    p.add_argument(
        "--playback_stride",
        type=int,
        default=1,
        help=(
            "Show one frame and skip the next N-1 frames during playback. "
            "Use 1 for every frame (default). This can make review feel much faster on 40 fps movies."
        ),
    )

    return p


def main(argv: list[str] | None = None) -> None:
    argv_list = list(argv) if argv is not None else sys.argv[1:]

    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", type=str, default=None)
    pre_args, _ = pre.parse_known_args(argv_list)

    parser = build_arg_parser()

    if pre_args.config:
        defaults = _load_config_defaults(pre_args.config)
        valid_dests = {a.dest for a in parser._actions}
        parser.set_defaults(**{k: v for k, v in defaults.items() if k in valid_dests})

    args = parser.parse_args(argv_list)
    apply_main_dir_overrides(args, stage="moviechecker")

    if not args.input_dir:
        raise SystemExit(
            "Missing --input_dir. Provide --input_dir explicitly, or use --main_dir to target the standardized project layout."
        )

    input_dir = os.path.abspath(str(args.input_dir))
    exclusions_csv = str(args.exclusions_csv) if args.exclusions_csv else os.path.join(input_dir, "exclusions.csv")

    app = MovieCheckerApp(
        input_dir=input_dir,
        exclusions_csv_path=os.path.abspath(exclusions_csv),
        review_all=bool(args.review_all),
        playback_stride=max(1, int(args.playback_stride)),
    )
    app.run()


if __name__ == "__main__":
    main()
