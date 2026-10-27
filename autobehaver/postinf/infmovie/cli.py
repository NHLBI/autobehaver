"""Command-line inference overlays with direct folder or registry discovery."""

import argparse
from dataclasses import fields


from autobehaver.argparse_utils import add_bool_pair, parse_cli_args
from autobehaver.resource_monitor import add_monitor_arguments, monitored_cli, record_settings
from autobehaver.pipeline_layout import apply_main_dir_overrides


from .config import InfMoviePipelineConfig


def build_parser() -> argparse.ArgumentParser:
    """Build the InfMovie clip-selection and rendering argument parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Generate inference overlay movies from Autobehaver inference CSVs.\n\n"
            "Use --main_dir for standard folders, explicit folders, or "
            "--registry_enabled --registry_csv for registry discovery."
        )
    )

    # Input discovery and clip selection
    parser.add_argument(
        "--main_dir",
        type=str,
        default=None,
        help=(
            'Project root for standard InfMovie input and output folders; explicit paths take precedence.'
        ),
    )

    parser.add_argument(
        "--mode",
        type=str,
        default=None,
        choices=["manual", "qc", "interesting", "full"],
        help="Clip selection mode: manual|qc|interesting|full",
    )

    parser.add_argument("--recording_id", type=str, default=None, help='Manual mode: recording identifier from the discovered folder or registry pairs.')
    parser.add_argument("--start_sec", type=float, default=None, help="Manual mode: clip start time (sec)")
    parser.add_argument("--duration_sec", type=float, default=None, help="Manual mode: clip duration (sec)")

    parser.add_argument("--num_clips", type=int, default=None, help="QC/interesting: number of clips to select")
    parser.add_argument("--clip_duration_sec", type=float, default=None, help="QC/interesting: clip duration (sec)")
    parser.add_argument("--clip_padding_sec", type=float, default=None, help="QC: padding around events (sec)")
    parser.add_argument(
        "--min_gap_sec",
        type=float,
        default=None,
        help="QC/interesting: minimum gap between clips per recording (sec)",
    )
    parser.add_argument("--max_clips_per_recording", type=int, default=None, help="QC/interesting: maximum clips per recording")

    parser.add_argument(
        "--selection_label_source",
        type=str,
        default=None,
        choices=["predicted", "smoothed"],
        help="Which labels to use for selection (QC/interesting).",
    )

    parser.add_argument(
        "--qc_uncertainty_threshold",
        type=float,
        default=None,
        help="QC: confidence threshold (lower => more uncertain)",
    )
    parser.add_argument("--qc_min_uncertainty_frames", type=int, default=None, help="QC: minimum consecutive uncertain frames")
    parser.add_argument("--qc_switch_window_sec", type=float, default=None, help="QC: window (sec) for rapid switching")
    parser.add_argument("--qc_switch_min_changes", type=int, default=None, help="QC: minimum label changes in window")

    add_bool_pair(parser, "qc_use_uncertainty", default=None, help="QC: include uncertainty-based clips")
    add_bool_pair(parser, "qc_use_switching", default=None, help="QC: include rapid-switching clips")

    parser.add_argument("--interesting_min_unique_labels", type=int, default=None, help="Interesting: minimum unique labels per clip")
    parser.add_argument("--interesting_min_changes", type=int, default=None, help="Interesting: minimum label changes per clip")

    parser.add_argument(
        "--file_filter",
        type=str,
        default=None,
        help="Optional substring filter applied to discovered pairs (case-insensitive).",
    )

    parser.add_argument(
        "--random_seed",
        type=int,
        default=None,
        help="Random seed for clip selection.",
    )

    # Rendering
    parser.add_argument("--num_workers", type=int, default=None, help="Number of parallel workers")
    parser.add_argument("--source_frame_offset", type=int, default=None, help='Legacy original-frame offset, when provenance is absent: 0 for untrimmed inputs or 5 when five leading rows were removed.')
    parser.add_argument("--skip_first_n_frames", type=int, default=None, help="Skip first N frames before rendering")

    parser.add_argument(
        "--timeline",
        type=str,
        default=None,
        choices=["none", "pose", "orient", "both"],
        help="Timeline overlay: none|pose|orient|both",
    )
    add_bool_pair(parser, "timeline_compare", default=None, help="Timeline compare: show predicted + smoothed")
    parser.add_argument(
        "--timeline_labels",
        type=str,
        default=None,
        choices=["predicted", "smoothed"],
        help="Timeline label source (single mode only)",
    )
    parser.add_argument("--timeline_height", type=int, default=None, help="Timeline bar height (px)")

    add_bool_pair(parser, "white_mode", default=None, help="White background + dark text")

    parser.add_argument("--speed_factor", type=float, default=None, help="Playback speed factor (e.g. 2.0)")
    parser.add_argument("--max_output_fps", type=float, default=None, help="Cap output FPS (default 40)")

    add_bool_pair(parser, "show_timestamp", default=None, help="Overlay timestamp in seconds")
    parser.add_argument(
        "--timestamp_position",
        type=str,
        default=None,
        choices=["top_left", "top_right", "bottom_left", "bottom_right"],
        help="Timestamp position",
    )
    parser.add_argument(
        "--timestamp_reference",
        type=str,
        default=None,
        choices=["movie", "clip", "rendered"],
        help="Timestamp reference frame: movie|clip|rendered",
    )
    parser.add_argument("--timestamp_decimals", type=int, default=None, help="Timestamp decimals")

    add_bool_pair(parser, "clobber", default=None, help="Overwrite rendering if output already exists")
    parser.add_argument("--default_movie_fps", type=float, help="Fallback FPS when the movie has no readable FPS.")
    add_bool_pair(parser, "write_plan_csv", default=None, help="Save the selected-clip plan CSV")
    add_bool_pair(parser, "write_run_log", default=None, help="Save run settings and rendered-output log")

    add_bool_pair(parser, "registry_enabled", default=None, help="Discover recording pairs from --registry_csv")
    parser.add_argument("--registry_csv", help="Project recording registry CSV used when registry discovery is enabled.")
    parser.add_argument("--registry_lock_timeout_sec", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--registry_lock_stale_sec", type=int, help=argparse.SUPPRESS)

    # Direct folder workflow
    parser.add_argument(
        "--inference_csv_dir",
        type=str,
        default=None,
        help="Directory containing inference CSV files",
    )
    parser.add_argument(
        "--movie_dir",
        type=str,
        default=None,
        help="Directory containing input movie files",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save output movies",
    )


    runtime_defaults = InfMoviePipelineConfig()
    parser.set_defaults(**{field.name: getattr(runtime_defaults, field.name)
                           for field in fields(runtime_defaults)})
    return add_monitor_arguments(parser)


@monitored_cli("infmovie")
def main(argv=None):
    """Resolve InfMovie inputs and run the selected overlay plan."""
    parser = build_parser()
    args = parse_cli_args(parser, argv)
    apply_main_dir_overrides(args, stage="infmovie")

    if not args.output_dir:
        parser.error("Provide --output_dir or --main_dir.")
    if args.registry_enabled:
        if not args.registry_csv:
            parser.error("--registry_enabled requires --registry_csv.")
    elif not args.inference_csv_dir or not args.movie_dir:
        parser.error("Provide --inference_csv_dir and --movie_dir, or --main_dir.")

    cfg = InfMoviePipelineConfig(**{field.name: getattr(args, field.name)
                                   for field in fields(InfMoviePipelineConfig)})
    record_settings(cfg)
    from .pipeline import run_infmovie_pipeline
    run_infmovie_pipeline(cfg)


if __name__ == "__main__":
    main()
