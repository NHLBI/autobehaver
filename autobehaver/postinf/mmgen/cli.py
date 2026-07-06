"""Command-line interface for mmgen (master matrix generation)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, Sequence

import yaml

from autobehaver.argparse_utils import add_bool_pair, normalize_argv
from autobehaver.pipeline_layout import apply_main_dir_overrides, config_main_dir_default


def _load_config_defaults(config_path: str) -> Dict[str, Any]:
    """Load mmgen defaults from the global pipeline YAML config.

    The config is expected to contain a top-level `mmgen:` section.
    All relative paths are resolved relative to the config file's directory.
    """

    cfg_path = Path(config_path).expanduser().resolve()
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")

    with cfg_path.open("r") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError("Pipeline config must be a YAML mapping (dict) at the top level")

    section = data.get("mmgen", {})
    if section is None:
        section = {}
    if not isinstance(section, dict):
        raise ValueError("Expected 'mmgen' section in config to be a mapping (dict)")

    base_dir = cfg_path.parent

    def _resolve_path(v: str) -> str:
        p = Path(str(v).strip())
        if not p.is_absolute():
            p = (base_dir / p).resolve()
        return str(p)

    defaults: Dict[str, Any] = {}

    global_main_dir = config_main_dir_default(data, base_dir)
    if global_main_dir:
        defaults["main_dir"] = global_main_dir

    v = section.get("main_dir")
    if isinstance(v, str) and v.strip() != "":
        defaults["main_dir"] = _resolve_path(v)

    # I/O paths
    if isinstance(section.get("inference_csv_dir"), str) and section.get("inference_csv_dir").strip() != "":
        defaults["inference_csv_dir"] = _resolve_path(section.get("inference_csv_dir"))
    if isinstance(section.get("poscalc_dir"), str) and section.get("poscalc_dir").strip() != "":
        defaults["poscalc_dir"] = _resolve_path(section.get("poscalc_dir"))
    if isinstance(section.get("output_dir"), str) and section.get("output_dir").strip() != "":
        defaults["output_dir"] = _resolve_path(section.get("output_dir"))

    # Core params
    for k in ("inf_tag", "min_bout", "edge_skip", "num_workers", "input_batch_size", "random_seed"):
        if k in section and section.get(k) is not None:
            defaults[k] = section.get(k)

    # Feature toggles / analysis scope
    for k in (
        "make_measurements",
        "calculate_occupancy",
        "calculate_postural_features",
        "generate_quiver_plots",
        "run_transition_statistics",
        "make_detailed_transition_outputs",
        "filter_features_by_bout_count",
    ):
        if k in section and section.get(k) is not None:
            defaults[k] = bool(section.get(k))
    for k in ("min_bout_count_for_features", "min_bout_count_for_transitions"):
        if k in section and section.get(k) is not None:
            defaults[k] = section.get(k)

    # Robustness / filtering
    for k in ("resume", "finalize_only", "process_only", "append_new_data", "plan_only"):
        if k in section and section.get(k) is not None:
            defaults[k] = bool(section.get(k))
    if "file_filter" in section and section.get("file_filter") is not None:
        fv = str(section.get("file_filter")).strip()
        defaults["file_filter"] = fv if fv != "" else None

    # Calibration overrides
    for k in ("chamber_diameter_px", "mm_per_px", "video_fps"):
        if k in section and section.get(k) is not None:
            defaults[k] = section.get(k)

    # Temporary-file policy
    if "keep_temp_files" in section and section.get("keep_temp_files") is not None:
        defaults["keep_temp_files"] = bool(section.get("keep_temp_files"))

    # Output toggles
    for k in ("master_matrix", "full_outputs"):
        if k in section and section.get(k) is not None:
            defaults[k] = bool(section.get(k))
    # Optional deterministic sharding
    for k in ("num_shards", "shard_index", "shard_by", "merge_shards", "matrix_first_shards", "finalize_shard_matrix", "merge_matrix_parts", "append_batch_id"):
        if k in section and section.get(k) is not None:
            if k in {"merge_shards", "matrix_first_shards", "finalize_shard_matrix", "merge_matrix_parts"}:
                defaults[k] = bool(section.get(k))
            else:
                defaults[k] = section.get(k)

    return defaults


def build_parser(defaults: Dict[str, Any] | None = None) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Autobehaver post-inference: master matrix generation (mmgen).\n\n"
            "Inputs: Autobehaver inference CSVs + positional_calcs CSVs.\n"
            "Outputs: Master feature matrix + summary artifacts in output_dir."
        )
    )

    # Global pipeline config (optional)
    p.add_argument("--config", help="Global pipeline YAML config file")

    p.add_argument(
        "--main_dir",
        default=(defaults or {}).get("main_dir"),
        required=False,
        help=(
            "Optional project root containing the standardized numbered pipeline folders. "
            "For mmgen this fills inference_csv_dir=<main_dir>/6_posepred_inference, "
            "poscalc_dir=<main_dir>/4_keypoint_features, and output_dir=<main_dir>/8_master_matrix."
        ),
    )

    # I/O (canonical)
    p.add_argument("--inference_csv_dir", required=False, help="Directory containing Autobehaver inference CSVs.")
    p.add_argument("--poscalc_dir", required=False, help="Directory containing positional_calcs CSVs.")
    p.add_argument("--output_dir", required=False, help="Directory to write mmgen outputs.")

    # Matching / parsing
    p.add_argument("--inf_tag", default="_normalized_inference", help="Suffix tag used to match inference to poscalc files.")

    # Bout params
    p.add_argument("--min_bout", type=int, default=5, help="Minimum bout length in frames.")
    p.add_argument("--edge_skip", type=int, default=5, help="Frames to skip at start/end for some computations.")

    # Parallelism / seeds
    p.add_argument("--num_workers", type=int, default=16, help="Multiprocessing workers.")
    p.add_argument("--random_seed", type=int, default=42, help="Random seed (affects sampling/subsetting steps).")

    # Feature toggles (default ON, provide --no_* to disable)
    p.add_argument("--no_measurements", dest="make_measurements", action="store_false", help="Disable measurement extraction.")
    p.add_argument("--no_occupancy", dest="calculate_occupancy", action="store_false", help="Disable occupancy/spatial analyses.")
    p.add_argument("--no_postural", dest="calculate_postural_features", action="store_false", help="Disable postural feature extraction.")
    p.add_argument("--no_quiver", dest="generate_quiver_plots", action="store_false", help="Disable orientation quiver plots.")
    p.add_argument("--no_transitions", dest="run_transition_statistics", action="store_false", help="Disable transition analysis/statistics.")
    p.add_argument("--no_transition_details", dest="make_detailed_transition_outputs", action="store_false", help="Disable detailed transition outputs.")

    p.set_defaults(
        make_measurements=True,
        calculate_occupancy=True,
        calculate_postural_features=True,
        generate_quiver_plots=True,
        run_transition_statistics=True,
        make_detailed_transition_outputs=True,
    )

    # Bout-count filtering
    p.add_argument(
        "--no_bout_count_filter",
        dest="filter_features_by_bout_count",
        action="store_false",
        help="Disable bout-count filtering guardrail (keep features even if bout counts are low).",
    )
    p.add_argument("--min_bout_count_for_features", type=int, default=6, help="Min bout count to keep bout-derived features.")
    p.add_argument("--min_bout_count_for_transitions", type=int, default=6, help="Min 'from' count to keep transition probability features.")
    p.set_defaults(filter_features_by_bout_count=True)

    # Temp / batching
    add_bool_pair(
        p,
        "keep_temp_files",
        default=True,
        help="Keep temporary on-disk HDF5 tables after successful run.",
        dest="keep_temp_files",
    )
    p.add_argument("--input_batch_size", type=int, default=20, help="How many inference CSVs to process per aggregation batch.")

    # Output emission controls
    add_bool_pair(
        p,
        "master_matrix",
        default=True,
        help="Emit MASTER_FEATURE_MATRIX.csv and its units manifest during finalization.",
        dest="master_matrix",
    )
    add_bool_pair(
        p,
        "full_outputs",
        default=True,
        help="Emit the numbered root-level mmgen outputs (1_, 2_, 3_, ... 12_) during finalization.",
        dest="full_outputs",
    )
    # Robustness / scheduling
    add_bool_pair(
        p,
        "resume",
        default=False,
        help="Resume from an existing temp_store.h5 by skipping committed files.",
        dest="resume",
    )
    add_bool_pair(
        p,
        "finalize_only",
        default=False,
        help="Only run final analysis + matrix generation from an existing temp_store.h5.",
        dest="finalize_only",
    )
    add_bool_pair(
        p,
        "process_only",
        default=False,
        help="Only do per-file processing (build/extend temp_store.h5) and exit.",
        dest="process_only",
    )
    add_bool_pair(
        p,
        "append_new_data",
        default=False,
        help=(
            "Incrementally update an existing MASTER_FEATURE_MATRIX.csv by detecting newly added source files, "
            "processing only the affected rows into a temporary append batch, and then replacing/appending rows in the canonical matrix."
        ),
        dest="append_new_data",
    )
    add_bool_pair(
        p,
        "plan_only",
        default=False,
        help=(
            "For append_new_data, create a frozen append batch plan (selected files, affected rows, shard settings) "
            "without processing it yet."
        ),
        dest="plan_only",
    )
    p.add_argument(
        "--append_batch_id",
        default=None,
        help=(
            "Append batch identifier created by --append_new_data --plan_only. "
            "Required for sharded append processing and shard merge/finalize steps."
        ),
    )
    p.add_argument(
        "--file_filter",
        default=None,
        help="Optional substring filter applied to inference CSV filenames (e.g., trial3)",
    )

    # ------------------------------------------------------------------
    # Deterministic sharding (optional)
    # ------------------------------------------------------------------
    p.add_argument(
        "--num_shards",
        type=int,
        default=1,
        help=(
            "Enable deterministic sharding across multiple jobs. "
            "Each shard writes to <output_dir>/_mmgen_shards/shard_XXX_of_YYY/temp_store.h5. "
            "Use --merge_shards after all shards complete. "
            "(For merge, you may omit --num_shards and it will auto-discover from shard folder names.)"
        ),
    )
    p.add_argument(
        "--shard_index",
        type=int,
        default=0,
        help="Shard index in [0, num_shards-1] (0-based).",
    )
    p.add_argument(
        "--shard_by",
        default="hash",
        choices=["hash", "trial", "rpi", "row_identity"],
        help="Sharding strategy: hash (default), bucket by trial/rpi number, or row_identity.",
    )

    add_bool_pair(
        p,
        "merge_shards",
        default=False,
        help=(
            "Merge shard temp stores into <output_dir>/temp_store.h5 and run finalization. "
            "If --num_shards is left at the default (1), mmgen will auto-discover the shard count from "
            "directories like shard_000_of_010 under <output_dir>/_mmgen_shards."
        ),
        dest="merge_shards",
    )

    add_bool_pair(
        p,
        "matrix_first_shards",
        default=False,
        help=(
            "Matrix-first sharded mode: process one shard and then immediately finalize that shard into a shard-local "
            "MASTER_FEATURE_MATRIX.csv. Requires --num_shards > 1 and --shard_by row_identity. After all shards finish, "
            "run --merge_matrix_parts to concatenate shard matrices."
        ),
        dest="matrix_first_shards",
    )
    add_bool_pair(
        p,
        "finalize_shard_matrix",
        default=False,
        help=(
            "Run shard-local finalization only, using an existing shard temp_store.h5 to create that shard's "
            "MASTER_FEATURE_MATRIX.csv. Requires --num_shards > 1 and --shard_by row_identity."
        ),
        dest="finalize_shard_matrix",
    )
    add_bool_pair(
        p,
        "merge_matrix_parts",
        default=False,
        help=(
            "Merge shard-local MASTER_FEATURE_MATRIX.csv files into <output_dir>/MASTER_FEATURE_MATRIX.csv. "
            "This is the final step for matrix-first sharded mode."
        ),
        dest="merge_matrix_parts",
    )

    # Calibration overrides
    p.add_argument(
        "--chamber_diameter_px",
        type=int,
        default=530,
        help="Arena diameter in pixels (or set to 0 to disable fixed diameter).",
    )
    p.add_argument("--mm_per_px", type=float, default=0.0169896744624207, help="Real-world calibration: mm per pixel.")
    p.add_argument("--video_fps", type=float, default=40.0, help="Video FPS used for time conversions.")

    # Apply config-derived defaults last.
    if defaults:
        p.set_defaults(**defaults)

    return p


def main(argv: Sequence[str] | None = None) -> None:
    # Two-stage parse so --config can provide defaults.
    tmp_parser = argparse.ArgumentParser(add_help=False)
    tmp_parser.add_argument("--config")
    if argv is None:
        tmp_ns, _ = tmp_parser.parse_known_args()
        argv_list = sys.argv[1:]
    else:
        argv_list = list(argv)
        tmp_ns, _ = tmp_parser.parse_known_args(argv_list)

    cfg_path = getattr(tmp_ns, "config", None)

    defaults: Dict[str, Any] = {}
    if cfg_path:
        defaults = _load_config_defaults(cfg_path)

    parser = build_parser(defaults=defaults)
    argv_list = normalize_argv(argv_list, parser=parser)
    args = parser.parse_args(argv_list)
    apply_main_dir_overrides(args, stage="mmgen")

    if any(bool(getattr(args, name, False)) for name in ["matrix_first_shards", "finalize_shard_matrix"]):
        args.master_matrix = True
        args.full_outputs = False

    if not args.inference_csv_dir or not args.poscalc_dir or not args.output_dir:
        raise SystemExit(
            "mmgen requires --inference_csv_dir, --poscalc_dir, and --output_dir "
            "(or provide them via --config mmgen section)"
        )

    if args.finalize_only and args.process_only:
        raise SystemExit("--finalize_only and --process_only are mutually exclusive")

    if bool(args.append_new_data) and bool(args.finalize_only):
        raise SystemExit("--append_new_data cannot be combined with --finalize_only")
    if bool(args.append_new_data) and any(bool(getattr(args, name, False)) for name in ["matrix_first_shards", "finalize_shard_matrix", "merge_matrix_parts"]):
        raise SystemExit("--append_new_data cannot be combined with matrix-first shard stages")

    if bool(getattr(args, "merge_shards", False)) and bool(args.process_only):
        raise SystemExit("--merge_shards cannot be combined with --process_only")

    matrix_stage_flags = [
        bool(getattr(args, "matrix_first_shards", False)),
        bool(getattr(args, "finalize_shard_matrix", False)),
        bool(getattr(args, "merge_matrix_parts", False)),
    ]
    if sum(1 for v in matrix_stage_flags if v) > 1:
        raise SystemExit("Use at most one of --matrix_first_shards, --finalize_shard_matrix, or --merge_matrix_parts")
    if bool(getattr(args, "matrix_first_shards", False)) and bool(args.process_only):
        raise SystemExit("--matrix_first_shards cannot be combined with --process_only; it already performs shard processing before shard-local finalization")
    if bool(getattr(args, "matrix_first_shards", False)) and bool(args.finalize_only):
        raise SystemExit("--matrix_first_shards cannot be combined with --finalize_only")
    if bool(getattr(args, "finalize_shard_matrix", False)) and (bool(args.process_only) or bool(args.finalize_only)):
        raise SystemExit("--finalize_shard_matrix is its own stage and cannot be combined with --process_only or --finalize_only")
    if bool(getattr(args, "merge_matrix_parts", False)) and (bool(args.process_only) or bool(args.finalize_only) or bool(getattr(args, "merge_shards", False))):
        raise SystemExit("--merge_matrix_parts cannot be combined with --process_only, --finalize_only, or --merge_shards")
    if (bool(getattr(args, "matrix_first_shards", False)) or bool(getattr(args, "finalize_shard_matrix", False))) and int(getattr(args, "num_shards", 1)) <= 1:
        raise SystemExit("matrix-first shard-local stages require --num_shards > 1")
    if (bool(getattr(args, "matrix_first_shards", False)) or bool(getattr(args, "finalize_shard_matrix", False))) and str(getattr(args, "shard_by", "hash")) != "row_identity":
        raise SystemExit("matrix-first shard-local stages require --shard_by row_identity")

    if bool(args.append_new_data) and bool(args.plan_only) and (
        bool(args.process_only) or bool(args.resume) or bool(getattr(args, "merge_shards", False))
    ):
        raise SystemExit("--append_new_data --plan_only cannot be combined with --process_only, --resume, or --merge_shards")

    if bool(args.append_new_data) and bool(args.resume) and not bool(args.process_only):
        raise SystemExit("--append_new_data only supports --resume together with --process_only (for rerunning an append shard)")

    if bool(args.append_new_data) and bool(args.process_only) and not getattr(args, "append_batch_id", None):
        raise SystemExit("append_new_data shard processing requires --append_batch_id")

    if bool(args.append_new_data) and bool(getattr(args, "merge_shards", False)) and not getattr(args, "append_batch_id", None):
        raise SystemExit("append_new_data shard merge/finalize requires --append_batch_id")

    if bool(args.append_new_data) and int(getattr(args, "num_shards", 1)) > 1 and str(getattr(args, "shard_by", "hash")) != "row_identity":
        raise SystemExit("append_new_data sharded mode requires --shard_by row_identity")

    if bool(args.append_new_data) and int(getattr(args, "num_shards", 1)) > 1 and not (
        bool(args.plan_only) or bool(args.process_only) or bool(getattr(args, "merge_shards", False))
    ):
        raise SystemExit(
            "append_new_data with num_shards > 1 uses a planned workflow: first run --plan_only, "
            "then per-shard --process_only jobs with --append_batch_id, then a final --merge_shards step"
        )

    if (not bool(args.process_only)) and (not bool(args.master_matrix)) and (not bool(args.full_outputs)):
        raise SystemExit(
            "mmgen has nothing to emit: use at least one of --master_matrix or --full_outputs, "
            "or add --process_only if you only want to build/extend the temp store"
        )

    chamber_diameter = args.chamber_diameter_px
    if chamber_diameter == 0:
        chamber_diameter = None

    from .core import MMGenConfig, run_mmgen

    cfg = MMGenConfig(
        inference_csv_dir=args.inference_csv_dir,
        poscalc_dir=args.poscalc_dir,
        output_dir=args.output_dir,
        inf_tag=args.inf_tag,
        min_bout=args.min_bout,
        edge_skip=args.edge_skip,
        num_workers=args.num_workers,
        random_seed=args.random_seed,
        make_measurements=args.make_measurements,
        calculate_occupancy=args.calculate_occupancy,
        calculate_postural_features=args.calculate_postural_features,
        generate_quiver_plots=args.generate_quiver_plots,
        run_transition_statistics=args.run_transition_statistics,
        make_detailed_transition_outputs=args.make_detailed_transition_outputs,
        filter_features_by_bout_count=args.filter_features_by_bout_count,
        min_bout_count_for_features=args.min_bout_count_for_features,
        min_bout_count_for_transitions=args.min_bout_count_for_transitions,
        keep_temp_files=args.keep_temp_files,
        emit_master_matrix=bool(args.master_matrix),
        emit_full_outputs=bool(args.full_outputs),
        input_batch_size=args.input_batch_size,
        chamber_diameter_px=chamber_diameter,
        mm_per_px=args.mm_per_px,
        video_fps=args.video_fps,
        resume=bool(args.resume),
        finalize_only=bool(args.finalize_only),
        process_only=bool(args.process_only),
        file_filter=args.file_filter,
        num_shards=int(getattr(args, "num_shards", 1)),
        shard_index=int(getattr(args, "shard_index", 0)),
        shard_by=str(getattr(args, "shard_by", "hash")),
        merge_shards=bool(getattr(args, "merge_shards", False)),
        matrix_first_shards=bool(getattr(args, "matrix_first_shards", False)),
        finalize_shard_matrix=bool(getattr(args, "finalize_shard_matrix", False)),
        merge_matrix_parts=bool(getattr(args, "merge_matrix_parts", False)),
        append_new_data=bool(getattr(args, "append_new_data", False)),
        plan_only=bool(getattr(args, "plan_only", False)),
        append_batch_id=getattr(args, "append_batch_id", None),
    )

    run_mmgen(cfg)


if __name__ == "__main__":
    main()
