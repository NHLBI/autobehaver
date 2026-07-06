import argparse
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from autobehaver.argparse_utils import add_bool_pair, normalize_argv
from autobehaver.pipeline_layout import apply_main_dir_overrides, config_main_dir_default


def _load_config_defaults(config_path: str) -> Dict[str, Any]:
    """Load the infmovie section from the global pipeline YAML config."""

    cfg_path = Path(config_path).expanduser().resolve()
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")

    with cfg_path.open("r") as f:
        data = yaml.safe_load(f) or {}

    if not isinstance(data, dict):
        raise ValueError("Pipeline config must be a YAML mapping (dict) at the top level")

    section = data.get("infmovie", {})
    if section is None:
        section = {}
    if not isinstance(section, dict):
        raise ValueError("Expected 'infmovie' section in config to be a mapping (dict)")

    base_dir = cfg_path.parent

    resolved: Dict[str, Any] = {}

    global_main_dir = config_main_dir_default(data, base_dir)
    if global_main_dir:
        resolved["main_dir"] = global_main_dir

    # ------------------------------------------------------------------
    # Global registry settings (project-wide)
    # ------------------------------------------------------------------
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

    def _resolve_path(v: str) -> str:
        p = Path(str(v).strip())
        if not p.is_absolute():
            p = (base_dir / p).resolve()
        return str(p)

    v = section.get("main_dir")
    if isinstance(v, str) and v.strip() != "":
        resolved["main_dir"] = _resolve_path(v)

    # ------------------------------------------------------------------
    # infmovie section
    # ------------------------------------------------------------------
    if "inference_csv_dir" in section and isinstance(section.get("inference_csv_dir"), str) and section.get("inference_csv_dir").strip() != "":
        resolved["inference_csv_dir"] = _resolve_path(section.get("inference_csv_dir"))

    if "movie_dir" in section and isinstance(section.get("movie_dir"), str) and section.get("movie_dir").strip() != "":
        resolved["movie_dir"] = _resolve_path(section.get("movie_dir"))

    if "output_dir" in section and isinstance(section.get("output_dir"), str) and section.get("output_dir").strip() != "":
        resolved["output_dir"] = _resolve_path(section.get("output_dir"))

    if section.get("file_filter") is not None:
        file_filter = str(section.get("file_filter")).strip()
        resolved["file_filter"] = file_filter if file_filter != "" else None

    # Mode
    if "mode" in section and section.get("mode") is not None:
        resolved["mode"] = str(section.get("mode")).strip()

    # Manual mode fields
    for k in ("recording_id",):
        if k in section and section.get(k) is not None:
            resolved[k] = str(section.get(k)).strip()

    for k in ("start_sec", "duration_sec"):
        if k in section and section.get(k) is not None:
            try:
                resolved[k] = float(section.get(k))
            except Exception:
                pass

    # Automated selection
    for k in ("num_clips", "max_clips_per_recording", "random_seed"):
        if k in section and section.get(k) is not None:
            try:
                resolved[k] = int(section.get(k))
            except Exception:
                pass

    for k in ("clip_duration_sec", "clip_padding_sec", "min_gap_sec"):
        if k in section and section.get(k) is not None:
            try:
                resolved[k] = float(section.get(k))
            except Exception:
                pass

    if "selection_label_source" in section and section.get("selection_label_source") is not None:
        resolved["selection_label_source"] = str(section.get("selection_label_source")).strip()

    # QC thresholds
    for k in (
        "qc_uncertainty_threshold",
        "qc_switch_window_sec",
    ):
        if k in section and section.get(k) is not None:
            try:
                resolved[k] = float(section.get(k))
            except Exception:
                pass

    for k in (
        "qc_min_uncertainty_frames",
        "qc_switch_min_changes",
    ):
        if k in section and section.get(k) is not None:
            try:
                resolved[k] = int(section.get(k))
            except Exception:
                pass

    for k in ("qc_use_uncertainty", "qc_use_switching"):
        if k in section and section.get(k) is not None:
            resolved[k] = bool(section.get(k))

    # Interesting thresholds
    for k in ("interesting_min_unique_labels", "interesting_min_changes"):
        if k in section and section.get(k) is not None:
            try:
                resolved[k] = int(section.get(k))
            except Exception:
                pass

    # Rendering
    for k in ("num_workers", "skip_first_n_frames", "timeline_height", "timestamp_decimals"):
        if k in section and section.get(k) is not None:
            try:
                resolved[k] = int(section.get(k))
            except Exception:
                pass

    for k in ("speed_factor", "max_output_fps", "default_movie_fps"):
        if k in section and section.get(k) is not None:
            try:
                resolved[k] = float(section.get(k))
            except Exception:
                pass

    for k in (
        "timeline",
        "timeline_labels",
        "timestamp_position",
        "timestamp_reference",
    ):
        if k in section and section.get(k) is not None:
            resolved[k] = str(section.get(k)).strip()

    if "clobber" in section and section.get("clobber") is not None:
        resolved["clobber"] = bool(section.get("clobber"))

    for k in (
        "timeline_compare",
        "white_mode",
        "show_timestamp",
        "write_plan_csv",
        "write_run_log",
    ):
        if k in section and section.get(k) is not None:
            resolved[k] = bool(section.get(k))

    return resolved


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate inference overlay movies from Autobehaver inference CSVs.\n\n"
            "Two workflows are supported:\n"
            "  1) Pipeline config + registry: --config config.yaml\n"
            "  2) Direct folder matching: --inference_csv_dir ... --movie_dir ..."
        )
    )

    # ------------------------------------------------------------------
    # Pipeline/config workflow
    # ------------------------------------------------------------------
    parser.add_argument(
        "--main_dir",
        type=str,
        default=None,
        help=(
            "Optional project root containing the standardized numbered pipeline folders. "
            "For infmovie this fills inference_csv_dir=<main_dir>/6_posepred_inference, "
            "movie_dir=<main_dir>/2_cropped_movies, and output_dir=<main_dir>/7_infmovies."
        ),
    )

    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to the global pipeline YAML config (recommended).",
    )

    parser.add_argument(
        "--mode",
        type=str,
        default=None,
        choices=["manual", "qc", "interesting", "full"],
        help="infmovie mode (overrides config): manual|qc|interesting|full",
    )

    parser.add_argument("--recording_id", type=str, default=None, help="Manual mode: registry recording_id")
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
        help="Optional substring filter applied to registry pairs (case-insensitive).",
    )

    parser.add_argument(
        "--random_seed",
        type=int,
        default=None,
        help="Random seed for clip selection (overrides config).",
    )

    # Rendering overrides
    parser.add_argument("--num_workers", type=int, default=None, help="Number of parallel workers")
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

    # ------------------------------------------------------------------
    # Direct folder workflow
    # ------------------------------------------------------------------
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


    return parser


def main(argv=None):
    parser = build_parser()

    argv_list = list(argv) if argv is not None else sys.argv[1:]
    argv_list = normalize_argv(argv_list, parser=parser)

    args = parser.parse_args(argv_list)
    apply_main_dir_overrides(args, stage="infmovie")

    # ------------------------------------------------------------------
    # Pipeline/config workflow
    # ------------------------------------------------------------------
    if args.config:
        defaults = _load_config_defaults(args.config)
        if (not getattr(args, "main_dir", None)) and isinstance(defaults.get("main_dir"), str) and str(defaults.get("main_dir")).strip() != "":
            args.main_dir = str(defaults.get("main_dir"))
            apply_main_dir_overrides(args, stage="infmovie")

        # CLI overrides (only those explicitly provided)
        override_map = {
            "inference_csv_dir": args.inference_csv_dir,
            "movie_dir": args.movie_dir,
            "output_dir": args.output_dir,
            "mode": args.mode,
            "recording_id": args.recording_id,
            "start_sec": args.start_sec,
            "duration_sec": args.duration_sec,
            "num_clips": args.num_clips,
            "clip_duration_sec": args.clip_duration_sec,
            "clip_padding_sec": args.clip_padding_sec,
            "min_gap_sec": args.min_gap_sec,
            "max_clips_per_recording": args.max_clips_per_recording,
            "selection_label_source": args.selection_label_source,
            "qc_uncertainty_threshold": args.qc_uncertainty_threshold,
            "qc_min_uncertainty_frames": args.qc_min_uncertainty_frames,
            "qc_switch_window_sec": args.qc_switch_window_sec,
            "qc_switch_min_changes": args.qc_switch_min_changes,
            "qc_use_uncertainty": args.qc_use_uncertainty,
            "qc_use_switching": args.qc_use_switching,
            "interesting_min_unique_labels": args.interesting_min_unique_labels,
            "interesting_min_changes": args.interesting_min_changes,
            "file_filter": args.file_filter,
            "random_seed": args.random_seed,
            "num_workers": args.num_workers,
            "skip_first_n_frames": args.skip_first_n_frames,
            "timeline": args.timeline,
            "timeline_compare": args.timeline_compare,
            "timeline_labels": args.timeline_labels,
            "timeline_height": args.timeline_height,
            "white_mode": args.white_mode,
            "speed_factor": args.speed_factor,
            "max_output_fps": args.max_output_fps,
            "show_timestamp": args.show_timestamp,
            "timestamp_position": args.timestamp_position,
            "timestamp_reference": args.timestamp_reference,
            "timestamp_decimals": args.timestamp_decimals,
        }

        for k, v in override_map.items():
            if v is not None:
                defaults[k] = v

        if getattr(args, "clobber", None) is not None:
            defaults["clobber"] = bool(args.clobber)

        from .config import InfMoviePipelineConfig
        from .pipeline import run_infmovie_pipeline

        # Filter to dataclass fields.
        allowed = set(InfMoviePipelineConfig.__dataclass_fields__.keys())
        cfg_kwargs = {k: v for k, v in defaults.items() if k in allowed}

        cfg = InfMoviePipelineConfig(**cfg_kwargs)

        run_infmovie_pipeline(cfg)
        return

    # ------------------------------------------------------------------
    # Direct folder workflow
    # ------------------------------------------------------------------
    if not args.inference_csv_dir or not args.movie_dir or not args.output_dir:
        parser.error(
            "Either provide --config for the pipeline workflow, or provide "
            "--inference_csv_dir --movie_dir --output_dir"
        )

    from .core import InfMovieConfig, run_infmovie

    skip_first = args.skip_first_n_frames if args.skip_first_n_frames is not None else 5

    config = InfMovieConfig(
        inference_csv_dir=args.inference_csv_dir,
        movie_dir=args.movie_dir,
        output_dir=args.output_dir,
        num_workers=8,
        skip_first_n_frames=skip_first,
        timeline="both",
        timeline_compare=False,
        timeline_labels="predicted",
        timeline_height=18,
        white_mode=False,
        speed_factor=1.0,
        max_output_fps=40.0,
        show_timestamp=False,
        timestamp_position="top_right",
        timestamp_decimals=1,
    )

    run_infmovie(config)


if __name__ == "__main__":
    main()
