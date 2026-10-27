"""Review full-frame movies and save manual chamber exclusions."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


from autobehaver.argparse_utils import add_bool_pair, parse_cli_args
from autobehaver.resource_monitor import add_monitor_arguments, monitored_cli, record_settings
from autobehaver.pipeline_layout import apply_main_dir_overrides
from autobehaver.preinf.autocropper.recording_db import (
    find_latest_recording_database_csv,
    load_recording_database_index,
)

from .core import MovieCheckerApp


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the MovieChecker command-line parser with runtime defaults."""
    p = argparse.ArgumentParser(description="Review full-frame movies and mark chamber exclusions")


    p.add_argument(
        "--input_dir",
        "-i",
        required=False,
        help='Directory containing full-frame movies; subfolders are not scanned.',
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
            'Path to the review-state CSV (default: <input_dir>/exclusions.csv).'
        ),
    )

    add_bool_pair(
        p,
        "review_all",
        default=False,
        help=(
            "Do not skip movies already listed in exclusions.csv. "
            "By default, the tool skips already-reviewed movie pairs. "
            "Cameras excluded entirely by recording metadata are always skipped."
        ),
    )

    p.add_argument(
        "--playback_stride",
        type=int,
        default=1,
        help=(
            'Display one frame and skip the next N-1 during playback; use 1 to display every frame (default).'
        ),
    )

    p.add_argument(
        "--recording_database_dir", default=None,
        help="Directory containing crop_occupancy_*.csv or recording_database*.csv. "
             "Defaults to <project>/rec_data/trials when present.",
    )
    p.add_argument(
        "--recording_database_csv", default=None,
        help="Explicit recording metadata CSV; overrides --recording_database_dir.",
    )

    return add_monitor_arguments(p)


@monitored_cli("moviechecker")
def main(argv: list[str] | None = None) -> None:
    """Parse runtime options and run MovieChecker."""
    parser = build_arg_parser()
    argv_list = list(argv) if argv is not None else sys.argv[1:]

    args = parse_cli_args(parser, argv_list)
    record_settings(args)
    apply_main_dir_overrides(args, stage="moviechecker")

    if not args.input_dir:
        raise SystemExit(
            "Missing --input_dir. Provide --input_dir explicitly, or use --main_dir to target the standardized project layout."
        )

    input_dir = os.path.abspath(str(args.input_dir))
    exclusions_csv = str(args.exclusions_csv) if args.exclusions_csv else os.path.join(input_dir, "exclusions.csv")

    recording_index = None
    source = None
    if args.recording_database_csv:
        source = Path(args.recording_database_csv).expanduser().resolve()
    elif args.recording_database_dir:
        source = find_latest_recording_database_csv(Path(args.recording_database_dir))
    else:
        project = Path(args.main_dir) if args.main_dir else Path(input_dir).parent
        candidate = project / "rec_data" / "trials"
        if candidate.exists():
            source = candidate
    if source is not None:
        recording_index = load_recording_database_index(source)
        print(f"Recording occupancy: {source}", flush=True)
    else:
        print("No recording occupancy selected; reviewing with manual exclusions only. "
              "Use --recording_database_dir to load recording records.", flush=True)

    app = MovieCheckerApp(
        input_dir=input_dir,
        exclusions_csv_path=os.path.abspath(exclusions_csv),
        review_all=bool(args.review_all),
        playback_stride=max(1, int(args.playback_stride)),
        recording_index=recording_index,
    )
    app.run()


if __name__ == "__main__":
    main()
