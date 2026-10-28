"""Command-line options for DeepLabCut inference."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Sequence


from autobehaver.argparse_utils import add_bool_pair, parse_cli_args
from autobehaver.resource_monitor import add_monitor_arguments, monitored_cli, record_settings
from autobehaver.pipeline_layout import apply_main_dir_overrides

from .config import DEFAULT_DLC_CONFIG_PATH, DLCInferencerConfig
from .core import run_dlc


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the DeepLabCut command-line parser with runtime defaults."""
    defaults = DLCInferencerConfig(input_dir="", dlc_config=DEFAULT_DLC_CONFIG_PATH)
    p = argparse.ArgumentParser(
        description="Run DeepLabCut inference on cropped videos and optionally create labeled videos.",
    )
    p.add_argument('--scratch_dir', default=defaults.scratch_dir,
                   help='Temporary working directory; otherwise use configured local job scratch.')


    p.add_argument(
        "--dlc_config",
        type=str,
        default=None,
        help="Path to the DLC project config.yaml file (trained model).",
    )
    p.add_argument(
        "--dlc_python",
        type=str,
        default=defaults.dlc_python,
        help=(
            "Optional Python executable or cluster wrapper for the isolated DeepLabCut backend "
            "(for example python-DLC). Run this CLI with the supported main Python environment."
        ),
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
        default=defaults.output_dir,
        help=(
            "Directory to write DLC outputs (.csv/.h5/_meta.pickle). "
            "If omitted, outputs go next to each video (DLC default)."
        ),
    )

    # Video selection
    p.add_argument(
        "--videos",
        nargs="*",
        default=defaults.videos,
        help=(
            "Optional explicit list of video files (paths or basenames in --input_dir). "
            "If provided, only these are processed."
        ),
    )

    p.add_argument(
        "--video_glob",
        type=str,
        default=defaults.video_glob,
        help="Glob pattern to select videos in --input_dir (e.g., '*.mp4' or 'session*_cam1.mp4').",
    )

    add_bool_pair(
        p,
        "recursive",
        default=defaults.recursive,
        help="Search recursively under --input_dir when using --video_glob (or default pattern).",
        dest="recursive",
    )

    p.add_argument(
        "--videotype",
        type=str,
        default=defaults.videotype,
        help="Video file extension / type (default: .mp4).",
    )

    # Pipeline convenience
    p.add_argument("--file_list", help="Explicit batch list of logical chamber recording stems")
    p.add_argument(
        "--file_filter",
        type=str,
        default=defaults.file_filter,
        help="Only process videos whose filename contains this substring (case-insensitive).",
    )

    add_bool_pair(
        p,
        "clobber",
        default=defaults.clobber,
        help=(
            'Reprocess selected videos and replace validated DeepLabCut outputs (default: disabled).'
        ),
        dest="clobber",
    )

    p.add_argument(
        "--dlc_tag_to_remove",
        type=str,
        default=defaults.dlc_tag_to_remove,
        help=(
            "Optional: remove this exact substring from DLC CSV output filenames in --output_dir after inference. "
            "(This is the long model-specific DLC tag, e.g. 'DLC_resnet50_...shuffle1_1000000')."
        ),
        dest="dlc_tag_to_remove",
    )


    add_bool_pair(
        p,
        "keep_dlc_tag",
        default=defaults.keep_dlc_tag,
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
        default=defaults.shuffle,
        help="Shuffle index of the trained model to use (must match training).",
    )

    p.add_argument(
        "--max_retries",
        type=int,
        default=defaults.max_retries,
        help='Maximum total inference attempts per video, including the initial batch.',
    )

    # Movie options
    add_bool_pair(
        p,
        "make_movie",
        default=defaults.make_movie,
        help="Create DLC labeled videos after inference (source frames + keypoint dots).",
        dest="make_movie",
    )

    add_bool_pair(
        p,
        "skip_analysis",
        default=defaults.skip_analysis,
        help=(
            'Skip inference and use verified existing DeepLabCut outputs; combine with --make_movie to create labeled videos.'
        ),
        dest="skip_analysis",
    )

    p.add_argument(
        "--dotsize",
        type=int,
        default=defaults.dotsize,
        help='Dot size for keypoints in labeled videos (default: 24).',
    )

    p.add_argument(
        "--pcutoff",
        type=float,
        default=defaults.pcutoff,
        help="Likelihood cutoff for showing points in labeled videos (lower shows more points).",
    )

    p.add_argument(
        "--colormap",
        type=str,
        default=defaults.colormap,
        help="Matplotlib colormap name for bodypart colors (default: rainbow).",
    )

    p.add_argument(
        "--exclude_bodyparts",
        nargs="*",
        default=defaults.exclude_bodyparts,
        help="Bodyparts to exclude from display (default: Front Back).",
    )

    p.add_argument(
        "--color_by",
        type=str,
        default=defaults.color_by,
        choices=["bodypart", "individual"],
        help="Color points by bodypart (single animal) or by individual (multi-animal).",
    )

    p.add_argument(
        "--trailpoints",
        type=int,
        default=defaults.trailpoints,
        help="Number of previous frames to show as a trail (default 0).",
    )

    add_bool_pair(
        p,
        "fastmode",
        default=defaults.fastmode,
        help="Use fast OpenCV rendering mode.",
        dest="fastmode",
    )

    p.add_argument(
        "--codec",
        type=str,
        default=defaults.codec,
        help="Codec for output video (default mp4v).",
    )

    add_bool_pair(
        p,
        "overwrite",
        default=defaults.overwrite,
        help="Overwrite existing labeled videos if supported by your DLC version.",
        dest="overwrite",
    )

    # Project registry (single CSV at project root)
    add_bool_pair(
        p,
        "registry_enabled",
        default=defaults.registry_enabled,
        help=(
            "Use/update the project-wide registry CSV. When enabled, DLC selects videos from the registry "
            "and upserts dlc_csv_relpath."
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

    return add_monitor_arguments(p)


@monitored_cli("dlc")
def main(argv: Optional[Sequence[str]] = None) -> None:
    """Parse runtime options and run DeepLabCut."""
    parser = build_arg_parser()
    argv_list = list(argv) if argv is not None else sys.argv[1:]

    args = parse_cli_args(parser, argv_list)
    record_settings(args)
    apply_main_dir_overrides(args, stage="dlc")

    if not args.input_dir:
        raise SystemExit(
            "Missing --input_dir. Provide --input_dir explicitly, or use --main_dir for the standard project folders."
        )

    if not args.dlc_config:
        args.dlc_config = DEFAULT_DLC_CONFIG_PATH

    if not args.dlc_config:
        raise SystemExit(
            "Missing --dlc_config. Provide --dlc_config explicitly "
            "or include the packaged DLC resource config."
        )

    dlc_config_path = Path(str(args.dlc_config)).expanduser()
    if not dlc_config_path.exists():
        raise SystemExit(f"DLC config.yaml not found: {dlc_config_path}")

    clobber = bool(args.clobber)

    cfg = DLCInferencerConfig(
        input_dir=str(args.input_dir),
        dlc_config=str(dlc_config_path.resolve()),
        videos=args.videos,
        video_glob=args.video_glob,
        recursive=bool(args.recursive),
        videotype=args.videotype,
        file_filter=str(args.file_filter) if args.file_filter else None,
        file_list=args.file_list,
        shuffle=int(args.shuffle),
        max_retries=int(args.max_retries),
        dlc_python=args.dlc_python,
        output_dir=args.output_dir,
        scratch_dir=args.scratch_dir,
        clobber=clobber,
        dlc_tag_to_remove=args.dlc_tag_to_remove,
        keep_dlc_tag=bool(args.keep_dlc_tag),
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
        registry_enabled=bool(args.registry_enabled),
        registry_csv=str(args.registry_csv) if args.registry_csv else None,
        registry_lock_timeout_sec=int(args.registry_lock_timeout_sec),
        registry_lock_stale_sec=int(args.registry_lock_stale_sec),
    )

    raise SystemExit(run_dlc(cfg))


if __name__ == "__main__":
    main()
