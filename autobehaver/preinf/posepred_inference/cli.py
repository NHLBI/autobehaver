"""Command-line options for PosePred checkpoint inference."""

from __future__ import annotations

import argparse
import sys
from typing import Optional, Sequence


from autobehaver.argparse_utils import add_bool_pair, parse_cli_args
from autobehaver.resource_monitor import add_monitor_arguments, monitored_cli, record_settings
from autobehaver.pipeline_layout import apply_main_dir_overrides

from .config import PosePredInferenceConfig, default_models
from .core import run_posepred_inference


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the PosePred command-line parser with runtime defaults."""
    defaults = PosePredInferenceConfig(input_dir="")
    p = argparse.ArgumentParser(
        description='Run PosePred checkpoint inference on normalized feature HDF5 files.',
    )


    # Project registry (optional)
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
        help="Path to project registry CSV (usually at project root).",
    )
    p.add_argument(
        "--registry_lock_timeout_sec",
        type=int,
        default=defaults.registry_lock_timeout_sec,
        help="Timeout (seconds) waiting for the registry writer lock.",
    )
    p.add_argument(
        "--registry_lock_stale_sec",
        type=int,
        default=defaults.registry_lock_stale_sec,
        help="Legacy option retained for compatibility; locks release when their process exits.",
    )


    p.add_argument(
        "--main_dir",
        type=str,
        default=None,
        help=(
            "Optional project root containing the standardized numbered pipeline folders. "
            "For posepred_inference this fills input_dir=<main_dir>/5_normalized_features and output_dir=<main_dir>/6_posepred_inference."
        ),
    )

    p.add_argument(
        "--input_dir",
        dest="input_dir",
        type=str,
        default=None,
        help="Directory containing normalized feature HDF5 files (e.g., 5_normalized_features).",
    )

    p.add_argument(
        "--output_dir",
        dest="output_dir",
        type=str,
        default=defaults.output_dir,
        help="Directory to write inference outputs (e.g., 6_posepred_inference).",
    )

    p.add_argument(
        "--inference_model",
        nargs="+",
        default=None,
        help="Checkpoint paths; defaults to the frozen final nh16/32/64 epoch-301 ensemble.",
    )

    # Selection / incremental processing
    p.add_argument("--file_list", help="Explicit batch list of logical chamber recording stems")
    p.add_argument(
        "--file_filter",
        type=str,
        default=defaults.file_filter,
        help="Only process HDF5 files whose filename contains this substring (case-insensitive).",
    )

    add_bool_pair(
        p,
        "clobber",
        default=defaults.clobber,
        help='Overwrite existing inference outputs (default: disabled).',
        dest="clobber",
    )

    add_bool_pair(
        p,
        "archive_existing",
        default=defaults.archive_existing,
        help='Archive existing outputs and details for selected recordings after input/checkpoint validation. Requires --clobber; disabled by default.',
        dest="archive_existing",
    )

    add_bool_pair(
        p,
        "flat_output",
        default=defaults.flat_output,
        help=(
            "Write inference CSVs directly into output_dir (and details into output_dir/inference_details). "
            "If disabled, write outputs under output_dir/inferences/."
        ),
        dest="flat_output",
    )

    add_bool_pair(
        p,
        "write_run_log",
        default=defaults.write_run_log,
        help="Write a small JSON run log to the output directory.",
        dest="write_run_log",
    )

    # Inference parameters
    p.add_argument("--num_workers", type=int, default=defaults.num_workers, help="Loader worker ceiling (-1 automatic; 0 synchronous).")
    p.add_argument("--batch", type=int, default=defaults.batch, help="Inference batch ceiling (0 = automatic RAM/VRAM/throughput calibration).")
    p.add_argument("--inf_average", type=int, default=defaults.inf_average, help="Forward-pass averaging count (default: 10).")

    add_bool_pair(
        p,
        "use_tta",
        default=defaults.use_tta,
        help="Enable test-time augmentation (noise injection).",
        dest="use_tta",
    )
    p.add_argument("--tta_noise_level", type=float, default=defaults.tta_noise_level, help='Test-time augmentation noise scale (default: 0.01).')
    p.add_argument("--tta_seed", type=int, default=defaults.tta_seed, help='Reproducible per-recording TTA seed (default: 42).')

    p.add_argument("--min_pose_bout", type=int, default=defaults.min_pose_bout, help='Minimum pose bout length in frames for smoothing (default: 5).')
    p.add_argument("--min_orient_bout", type=int, default=defaults.min_orient_bout, help='Minimum orientation bout length in frames for smoothing (default: 5).')

    p.add_argument(
        "--sequence_length",
        type=int,
        default=defaults.sequence_length,
        help="Fallback sequence length if checkpoint does not specify it (default: 41).",
    )

    from autobehaver.inference_options import add_inference_runtime_arguments
    return add_monitor_arguments(add_inference_runtime_arguments(p))


@monitored_cli("posepred_inference")
def main(argv: Optional[Sequence[str]] = None) -> None:
    """Parse runtime options and run PosePred."""
    parser = build_arg_parser()
    argv_list = list(argv) if argv is not None else sys.argv[1:]

    args = parse_cli_args(parser, argv_list)
    record_settings(args)
    apply_main_dir_overrides(args, stage="posepred_inference")

    if not args.input_dir:
        raise SystemExit(
            "Missing --input_dir. Provide --input_dir explicitly, or use --main_dir for the standard project folders."
        )

    if not args.inference_model:
        args.inference_model = default_models()

    clobber = bool(args.clobber)

    cfg = PosePredInferenceConfig(
        input_dir=str(args.input_dir),
        output_dir=str(args.output_dir) if args.output_dir else None,
        inference_model=[str(x) for x in (args.inference_model or [])],
        num_workers=int(args.num_workers),
        batch=int(args.batch),
        memory_gb=args.memory_gb, noise_workers=args.noise_workers, vram_fraction=args.vram_fraction,
        inf_average=int(args.inf_average),
        use_tta=bool(args.use_tta),
        tta_noise_level=float(args.tta_noise_level),
        tta_seed=int(args.tta_seed),
        min_pose_bout=int(args.min_pose_bout),
        min_orient_bout=int(args.min_orient_bout),
        sequence_length=int(args.sequence_length),
        file_filter=str(args.file_filter) if args.file_filter else None,
        file_list=args.file_list,
        clobber=clobber,
        archive_existing=bool(args.archive_existing),
        flat_output=bool(args.flat_output),
        write_run_log=bool(args.write_run_log),
        registry_enabled=bool(args.registry_enabled),
        registry_csv=str(args.registry_csv) if args.registry_csv else None,
        registry_lock_timeout_sec=int(args.registry_lock_timeout_sec),
        registry_lock_stale_sec=int(args.registry_lock_stale_sec),
    )

    run_posepred_inference(cfg)


if __name__ == "__main__":
    main()
