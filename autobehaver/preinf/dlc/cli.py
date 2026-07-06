"""CLI for the DLC inference wrapper stage.

The global config file is expected to contain a top-level ``dlc:`` section.
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

from .config import DEFAULT_DLC_CONFIG_PATH, DLCInferencerConfig
from .core import run_dlc


def _load_config_defaults(config_path: str) -> Dict[str, Any]:
    """Load defaults from a global YAML config file.

    DLC reads:
      - the ``dlc:`` section (stage-specific)
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

    section = data.get("dlc", {})
    if section is None:
        section = {}
    if not isinstance(section, dict):
        raise ValueError("Expected 'dlc' section in config to be a mapping (dict)")

    path_keys = {
        "main_dir",
        "input_dir",
        "output_dir",
        "dlc_config",
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
    p = argparse.ArgumentParser(
        description="Run DeepLabCut inference on cropped videos and optionally create labeled videos.",
    )

    p.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to global pipeline YAML config (reads the 'dlc:' section).",
    )

    p.add_argument(
        "--dlc_config",
        type=str,
        default=None,
        help="Path to the DLC project config.yaml file (trained model).",
    )

    p.add_argument(
        "--main_dir",
        type=str,
        default=None,
        help=(
            "Optional project root containing the standardized numbered pipeline folders. "
            "For DLC this fills input_dir=<main_dir>/2_cropped_movies and output_dir=<main_dir>/3_dlc."
        ),
    )

    # Inputs / outputs
    p.add_argument(
        "--input_dir",
        type=str,
        default=None,
        help="Directory containing the videos to analyze.",
    )

    p.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help=(
            "Directory to write DLC outputs (.csv/.h5/_meta.pickle). "
            "If omitted, outputs go next to each video (DLC default)."
        ),
    )

    # Video selection
    p.add_argument(
        "--videos",
        nargs="*",
        default=None,
        help=(
            "Optional explicit list of video files (paths or basenames in --input_dir). "
            "If provided, only these are processed."
        ),
    )

    p.add_argument(
        "--video_glob",
        type=str,
        default=None,
        help="Glob pattern to select videos in --input_dir (e.g., '*.mp4' or 'session*_cam1.mp4').",
    )

    add_bool_pair(
        p,
        "recursive",
        default=False,
        help="Search recursively under --input_dir when using --video_glob (or default pattern).",
        dest="recursive",
    )

    p.add_argument(
        "--videotype",
        type=str,
        default=".mp4",
        help="Video file extension / type (default: .mp4).",
    )

    # Pipeline convenience
    p.add_argument(
        "--file_filter",
        type=str,
        default=None,
        help="Only process videos whose filename contains this substring (case-insensitive).",
    )

    add_bool_pair(
        p,
        "clobber",
        default=False,
        help=(
            "Overwrite existing DLC outputs in --output_dir (reprocess videos). "
            "Default: disabled (safe)."
        ),
        dest="clobber",
    )

    p.add_argument(
        "--dlc_tag_to_remove",
        type=str,
        default=None,
        help=(
            "Optional: remove this exact substring from DLC CSV output filenames in --output_dir after inference. "
            "(This is the long model-specific DLC tag, e.g. 'DLC_resnet50_...shuffle1_1000000')."
        ),
        dest="dlc_tag_to_remove",
    )


    add_bool_pair(
        p,
        "keep_dlc_tag",
        default=False,
        help=(
            "Keep the DLC tag in output CSV filenames even if dlc_tag_to_remove is set. "
            "Default: strip the tag when dlc_tag_to_remove is provided."
        ),
        dest="keep_dlc_tag",
    )

    # Inference options
    p.add_argument(
        "--shuffle",
        type=int,
        default=1,
        help="Shuffle index of the trained model to use (must match training).",
    )

    p.add_argument(
        "--max_retries",
        type=int,
        default=10,
        help="Maximum number of retry attempts for failed videos.",
    )

    # Movie options
    add_bool_pair(
        p,
        "make_movie",
        default=False,
        help="Create DLC labeled videos after inference (source frames + keypoint dots).",
        dest="make_movie",
    )

    add_bool_pair(
        p,
        "skip_analysis",
        default=False,
        help=(
            "Skip inference and ONLY try to create labeled videos "
            "(expects DLC .h5/.csv/_meta.pickle outputs already exist)."
        ),
        dest="skip_analysis",
    )

    p.add_argument(
        "--dotsize",
        type=int,
        default=24,
        help="Dot size for keypoints in labeled videos (default 24; increase for presentations).",
    )

    p.add_argument(
        "--pcutoff",
        type=float,
        default=0.6,
        help="Likelihood cutoff for showing points in labeled videos (lower shows more points).",
    )

    p.add_argument(
        "--colormap",
        type=str,
        default="rainbow",
        help="Matplotlib colormap name for bodypart colors (default: rainbow).",
    )

    p.add_argument(
        "--exclude_bodyparts",
        nargs="*",
        default=["Front", "Back"],
        help="Bodyparts to exclude from display (default: Front Back).",
    )

    p.add_argument(
        "--color_by",
        type=str,
        default="bodypart",
        choices=["bodypart", "individual"],
        help="Color points by bodypart (single animal) or by individual (multi-animal).",
    )

    p.add_argument(
        "--trailpoints",
        type=int,
        default=0,
        help="Number of previous frames to show as a trail (default 0).",
    )

    add_bool_pair(
        p,
        "fastmode",
        default=True,
        help="Use fast OpenCV rendering mode.",
        dest="fastmode",
    )

    p.add_argument(
        "--codec",
        type=str,
        default="mp4v",
        help="Codec for output video (default mp4v).",
    )

    add_bool_pair(
        p,
        "overwrite",
        default=True,
        help="Overwrite existing labeled videos if supported by your DLC version.",
        dest="overwrite",
    )

    # Project registry (single CSV at project root)
    add_bool_pair(
        p,
        "registry_enabled",
        default=False,
        help=(
            "Use/update the project-wide registry CSV. When enabled, DLC selects videos from the registry "
            "and upserts dlc_csv_relpath."
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

    return p


def main(argv: Optional[Sequence[str]] = None) -> None:
    # Two-pass parse so we can apply pipeline-config defaults.
    parser = build_arg_parser()

    argv_list = list(argv) if argv is not None else sys.argv[1:]

    pre = parser.parse_known_args(argv_list)[0]

    if pre.config:
        defaults = _load_config_defaults(pre.config)

        mapped: Dict[str, Any] = {}
        for k in ("input_dir", "output_dir", "dlc_config"):
            if k in defaults:
                mapped[k] = defaults[k]

        passthrough = {
            "shuffle",
            "videotype",
            "recursive",
            "video_glob",
            "max_retries",
            "make_movie",
            "skip_analysis",
            "dotsize",
            "pcutoff",
            "colormap",
            "exclude_bodyparts",
            "color_by",
            "trailpoints",
            "fastmode",
            "codec",
            "overwrite",
            "file_filter",
            "clobber",
            "dlc_tag_to_remove",
            "keep_dlc_tag",
            "registry_enabled",
            "registry_csv",
            "registry_lock_timeout_sec",
            "registry_lock_stale_sec",
        }
        for k in passthrough:
            if k in defaults:
                mapped[k] = defaults[k]

        parser.set_defaults(**mapped)

    args = parser.parse_args(argv_list)
    apply_main_dir_overrides(args, stage="dlc")

    if not args.input_dir:
        raise SystemExit(
            "Missing --input_dir. Provide --input_dir explicitly, or use --config with a 'dlc:' section that sets input_dir."
        )

    if not getattr(args, "dlc_config", None):
        args.dlc_config = DEFAULT_DLC_CONFIG_PATH

    if not getattr(args, "dlc_config", None):
        raise SystemExit(
            "Missing --dlc_config. Provide --dlc_config explicitly, use --config with a 'dlc:' section, "
            "or include the packaged DLC resource config."
        )

    dlc_config_path = Path(str(args.dlc_config)).expanduser()
    if not dlc_config_path.exists():
        raise SystemExit(f"DLC config.yaml not found: {dlc_config_path}")

    clobber = bool(getattr(args, "clobber", False))

    cfg = DLCInferencerConfig(
        input_dir=str(args.input_dir),
        dlc_config=str(dlc_config_path.resolve()),
        videos=args.videos,
        video_glob=args.video_glob,
        recursive=bool(args.recursive),
        videotype=args.videotype,
        file_filter=str(args.file_filter) if args.file_filter else None,
        shuffle=int(args.shuffle),
        max_retries=int(args.max_retries),
        output_dir=args.output_dir,
        clobber=clobber,
        dlc_tag_to_remove=getattr(args, "dlc_tag_to_remove", None),
        keep_dlc_tag=bool(getattr(args, "keep_dlc_tag", False)),
        make_movie=bool(args.make_movie),
        skip_analysis=bool(args.skip_analysis),
        dotsize=int(args.dotsize),
        pcutoff=float(args.pcutoff),
        colormap=str(args.colormap),
        exclude_bodyparts=list(args.exclude_bodyparts) if args.exclude_bodyparts is not None else [],
        color_by=str(args.color_by),
        trailpoints=int(args.trailpoints),
        fastmode=bool(args.fastmode),
        codec=str(args.codec),
        overwrite=bool(args.overwrite),

        # Project registry
        registry_enabled=bool(getattr(args, "registry_enabled", False)),
        registry_csv=str(getattr(args, "registry_csv", "")) if getattr(args, "registry_csv", None) else None,
        registry_lock_timeout_sec=int(getattr(args, "registry_lock_timeout_sec", 120)),
        registry_lock_stale_sec=int(getattr(args, "registry_lock_stale_sec", 60 * 60)),
    )

    raise SystemExit(run_dlc(cfg))


if __name__ == "__main__":
    main()
