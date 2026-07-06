"""Core implementation of keyprenorm.

Key behaviors:
  - Canonical schema mismatch => file skipped.
  - Normalized outputs are compressed HDF5 with dataset 'data' and attr 'columns'.
  - When cfg.output_dir is provided, normalized H5 files and norm_params.csv are
    written directly into that stage root.
  - When cfg.output_dir is omitted, outputs are written under
    <input_dir>/normalization_h5 and <input_dir>/norm_params/.
  - Optional labels CSVs can be written as per-column datasets in an HDF5 file.
  - A per-run JSON manifest is written to the output folder.
"""

from __future__ import annotations

import argparse
import glob
import logging
import os
from datetime import datetime
from multiprocessing import Pool
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

from . import __stage_version__
from .config import KeypreNormConfig
from .inspect_outputs import inspect_label_h5, inspect_normalized_h5
from .labels import label_paths_for_csv
from .logging_utils import setup_logging_queue
from .manifest import atomic_write_json, build_run_manifest
from .normparams import calculate_normalization_params
from .normalize import init_normalize_worker, normalize_one_csv
from .schema import (
    filter_files_by_schema,
    find_processing_log,
    read_csv_header_columns,
    resolve_glob_pattern,
    summarize_bad_schema,
)
from .sizecheck import (
    estimate_rows_from_size_worker,
    get_accurate_bytes_per_row,
    init_sizecheck_worker,
)

logger = logging.getLogger("keyprenorm")


def run_keyprenorm(*, cfg: KeypreNormConfig, args: argparse.Namespace) -> None:
    """Run keyprenorm with a validated configuration."""

    base_input_folder = cfg.input_dir
    input_folder = base_input_folder

    num_workers = int(cfg.num_workers)
    edge_skip = int(cfg.edge_skip)
    max_rows = int(cfg.max_rows)
    batch_size = int(cfg.batch_size)

    if num_workers < 1:
        raise ValueError("--num_workers must be >= 1")

    if max_rows < 0:
        raise ValueError("--max_rows must be >= 0")
    if batch_size <= 0:
        raise ValueError("--batch_size must be >= 1")

    require_full_length = bool(cfg.require_full_length)
    if require_full_length and max_rows == 0:
        # Variable-length mode: strict full-length doesn't apply
        require_full_length = False

    # Output roots
    explicit_output_dir = bool(getattr(cfg, "output_dir", None))
    base_output_root = cfg.output_dir if explicit_output_dir else base_input_folder
    if base_output_root is None:
        base_output_root = base_input_folder

    # When output_dir is provided, use a flat stage-root layout.
    if explicit_output_dir:
        base_output_folder = str(base_output_root)
        os.makedirs(base_output_folder, exist_ok=True)
    else:
        base_output_folder = os.path.join(base_output_root, "normalization_h5")
        os.makedirs(base_output_folder, exist_ok=True)

    # Resolve norm params path
    if cfg.norm_params:
        norm_params_path = cfg.norm_params
        norm_params_dir = os.path.dirname(norm_params_path) or "."
        os.makedirs(norm_params_dir, exist_ok=True)
    else:
        if explicit_output_dir:
            norm_params_dir = str(base_output_root)
            os.makedirs(norm_params_dir, exist_ok=True)
            norm_params_path = os.path.join(norm_params_dir, "norm_params.csv")
        else:
            norm_params_dir = os.path.join(base_output_root, "norm_params")
            os.makedirs(norm_params_dir, exist_ok=True)
            norm_params_path = os.path.join(norm_params_dir, "norm_params.csv")

    # Logging setup (multiprocessing-safe)
    log_queue, listener = setup_logging_queue(log_level=cfg.log_level, log_file=cfg.log_file)

    start_time = datetime.now()
    manifest_path: Optional[str] = None
    manifest: Optional[Dict[str, object]] = None

    # Registry integration (optional)
    used_registry = False
    registry_csv_path: Optional[str] = None
    registry_project_root: Optional[Path] = None

    # Track all discovered inputs so registry updates can include existing outputs.
    all_csv_files: list[str] = []

    try:
        # -------------------------
        # Discover inputs
        # -------------------------
        # Priority:
        #   1) registry (if enabled)
        #   2) filesystem scan of input_folder
        logger.info("Using input folder: %s", input_folder)

        csv_glob = resolve_glob_pattern(input_folder, cfg.pattern)

        if bool(getattr(cfg, "registry_enabled", False)) and getattr(cfg, "registry_csv", None):
            registry_csv_path = str(getattr(cfg, "registry_csv"))
            reg_path = Path(registry_csv_path).expanduser().resolve()
            if not reg_path.exists():
                raise FileNotFoundError(
                    f"Registry enabled but CSV not found: {reg_path}. "
                    "Run autocropper + dlc + keypre first (to create registry entries), or disable registry for this run."
                )

            registry_project_root = reg_path.parent

            try:
                from autobehaver.registry import read_registry
            except Exception as e:
                raise RuntimeError(
                    "Failed to import registry helpers (pandas required). "
                    "Install pandas in this environment or disable registry."
                ) from e

            df = read_registry(reg_path)

            if not df.empty and "keypre_csv_relpath" in df.columns:
                def _is_active(val: object) -> bool:
                    if val is None:
                        return True
                    try:
                        if pd.isna(val):
                            return True
                    except Exception:
                        pass
                    if isinstance(val, bool):
                        return bool(val)
                    s = str(val).strip().lower()
                    if s in {"0", "false", "no", "n", "off"}:
                        return False
                    if s in {"1", "true", "yes", "y", "on"}:
                        return True
                    return True

                try:
                    df2 = df[df["keypre_csv_relpath"].notna()].copy()
                except Exception:
                    df2 = df.copy()

                if "active" in df2.columns:
                    try:
                        df2 = df2[df2["active"].apply(_is_active)]
                    except Exception:
                        pass

                rels = [
                    str(x).replace("\\", "/")
                    for x in df2["keypre_csv_relpath"].tolist()
                    if str(x).strip() != ""
                ]

                files: list[str] = []
                for rel in rels:
                    pth = (registry_project_root / rel).resolve()
                    if not pth.exists():
                        bn = Path(rel).name
                        # Common local/HPC mirror fallbacks
                        cands = [
                            Path(input_folder) / bn,
                            Path(input_folder) / "positional_calcs" / bn,
                            Path(input_folder) / "raw_positional_calcs" / bn,
                        ]
                        for c in cands:
                            if c.exists():
                                pth = c.resolve()
                                break
                    if not pth.exists():
                        continue
                    if not str(pth).lower().endswith(".csv"):
                        continue
                    files.append(str(pth))

                all_csv_files = sorted(set(files))

            used_registry = True
            logger.info("Using registry for keyprenorm inputs: %s (%d CSV files)", reg_path, len(all_csv_files))
        else:
            all_csv_files = sorted(glob.glob(csv_glob))

        # Optional substring filter (pipeline convenience)
        if getattr(cfg, "file_filter", None):
            needle = str(cfg.file_filter).lower()
            before = len(all_csv_files)
            all_csv_files = [f for f in all_csv_files if needle in os.path.basename(f).lower()]
            logger.info("Applied file_filter='%s': %d/%d CSVs selected.", cfg.file_filter, len(all_csv_files), before)

        if not all_csv_files:
            if getattr(cfg, "file_filter", None):
                if used_registry:
                    raise FileNotFoundError(
                        f"No registry keypre CSV inputs matched file_filter='{cfg.file_filter}'. "
                        "Disable file_filter, or disable registry for this run."
                    )
                raise FileNotFoundError(
                    f"No CSV files matched pattern '{cfg.pattern}' with file_filter='{cfg.file_filter}' in: {input_folder}"
                )
            if used_registry:
                raise FileNotFoundError(
                    "No keypre CSV files discovered from registry. "
                    "Check that the registry has keypre_csv_relpath entries (run the keypre stage), "
                    "or disable registry for this run."
                )
            raise FileNotFoundError(f"No CSV files found in: {input_folder}")
        logger.info("Found %d CSV files total.", len(all_csv_files))
        logger.info("Output folder: %s", base_output_folder)
        logger.info("Norm params path: %s", norm_params_path)

        # Build manifest early (status=running)
        manifest_path, manifest = build_run_manifest(
            out_dir=base_output_folder,
            args=args,
            input_folder=input_folder,
            csv_glob=csv_glob,
            norm_params_path=norm_params_path,
            start_time=start_time,
            all_csv_files=all_csv_files,
            stage_version=__stage_version__,
            script_path=None,
        )

        # Labels folders
        labels_input_folder = cfg.labels_dir
        labels_output_folder: Optional[str] = None
        if labels_input_folder:
            if os.path.exists(labels_input_folder):
                labels_output_folder = os.path.join(base_output_folder, "labels")
                os.makedirs(labels_output_folder, exist_ok=True)
                logger.info("Labels output folder: %s", labels_output_folder)
            else:
                logger.warning("Provided labels directory does not exist: %s", labels_input_folder)
                labels_input_folder = None

        # Optional size check (only sensible in fixed-length mode)
        size_check_info: Dict[str, object] = {"enabled": bool(cfg.size_check), "ran": False}
        if cfg.size_check:
            if max_rows == 0:
                logger.info("--size_check skipped because --max_rows is 0 (variable-length mode).")
            else:
                accurate_bpr = get_accurate_bytes_per_row(all_csv_files)
                size_check_info["ran"] = True
                size_check_info["bytes_per_row"] = accurate_bpr

                with Pool(processes=num_workers, initializer=init_sizecheck_worker, initargs=(accurate_bpr,)) as pool:
                    results = pool.map(estimate_rows_from_size_worker, all_csv_files)

                results.sort(key=lambda x: x[1])  # file size asc
                smallest_5_percent = results[: max(1, len(results) // 20)]

                min_required = edge_skip + max_rows
                problematic = [
                    (f, size, rows)
                    for f, size, rows in smallest_5_percent
                    if rows != -1 and rows < min_required
                ]
                size_check_info["checked_files"] = len(smallest_5_percent)
                size_check_info["min_required_rows_estimate"] = min_required
                size_check_info["problematic_count"] = len(problematic)
                size_check_info["problematic_examples"] = [
                    {
                        "file": os.path.basename(f),
                        "estimated_rows": rows,
                        "size_bytes": size,
                    }
                    for f, size, rows in problematic[:25]
                ]

                logger.info("Size-check: checked %d smallest files (~5%%).", len(smallest_5_percent))
                if problematic:
                    logger.warning(
                        "Size-check: %d files may have insufficient rows (need ~%d). Consider --require_full_length.",
                        len(problematic),
                        min_required,
                    )
                else:
                    logger.info("Size-check: all checked files appear to have sufficient rows.")

        # Create / append to processing log (if present)
        processing_log_path = find_processing_log(os.path.dirname(base_output_root))
        if processing_log_path:
            with open(processing_log_path, "a", encoding="utf-8") as f:
                f.write("\n7. Normalization\n")
                f.write(f"   Started at: {start_time.isoformat()}\n\n")
                f.write("   Normalization Parameters:\n")
                if cfg.calculate_norm:
                    f.write(f"   - Mode: Calculating and saving to {os.path.basename(norm_params_path)}\n")
                else:
                    f.write(f"   - Mode: Loading from {os.path.basename(norm_params_path)}\n")
                f.write(f"   - Edge Skip: {edge_skip} frames\n")
                f.write(f"   - Max Rows: {max_rows} (0 means all rows)\n")
                f.write(f"   - Require Full Length: {require_full_length}\n\n")

        # Canonical columns and normalization parameters
        norm_params_mode = "calculate" if cfg.calculate_norm else "load"
        bad_schema: list[Dict[str, object]] = []
        skipped_short_normparams: list[str] = []

        if cfg.calculate_norm:
            first_cols, err = read_csv_header_columns(all_csv_files[0])
            if err is not None or first_cols is None:
                raise RuntimeError(
                    f"Could not read header from first CSV file: {all_csv_files[0]} ({err})"
                )
            canonical_columns = first_cols

            all_csv_files_good, bad_schema = filter_files_by_schema(all_csv_files, canonical_columns, max_examples=10)

            norm_df, skipped_short_normparams = calculate_normalization_params(
                csv_files=all_csv_files_good,
                canonical_columns=canonical_columns,
                norm_params_path=norm_params_path,
                norm_params_dir=norm_params_dir,
                num_workers=num_workers,
                edge_skip=edge_skip,
                max_rows=max_rows,
                require_full_length=require_full_length,
                batch_size=batch_size,
                log_queue=log_queue,
                log_level=cfg.log_level,
            )
        else:
            if not os.path.exists(norm_params_path):
                raise FileNotFoundError(
                    f"Normalization parameters file not found at: {norm_params_path}\n"
                    f"Run again with --calculate_norm to create it."
                )
            norm_df = pd.read_csv(norm_params_path)
            canonical_columns = norm_df["Header"].tolist()

            all_csv_files_good, bad_schema = filter_files_by_schema(all_csv_files, canonical_columns, max_examples=10)

            logger.info("Loaded normalization parameters from: %s", norm_params_path)
            logger.info("Canonical columns: %d", len(canonical_columns))

        # If only calculating norm params, stop here
        if cfg.norm_only:
            logger.info("Done (--norm_only). No HDF5 files were created.")

            # finalize manifest
            if manifest is not None:
                manifest["results"] = {
                    "mode": norm_params_mode,
                    "norm_only": True,
                    "canonical_column_count": len(canonical_columns),
                    "input_csv_total": len(all_csv_files),
                    "input_csv_good_schema": len(all_csv_files_good),
                    "schema_bad_count": len(bad_schema),
                    "schema_bad_examples": summarize_bad_schema(bad_schema),
                    "skipped_short_normparams_count": len(skipped_short_normparams),
                    "size_check": size_check_info,
                    "processing_log_file": os.path.abspath(processing_log_path)
                    if processing_log_path
                    else None,
                }
            return

        # Filter for existing completed outputs when clobber is disabled.
        csv_files_to_process = list(all_csv_files_good)
        existing_outputs_skipped = 0
        invalid_outputs_reprocessed = 0
        if not cfg.clobber:
            expected_n_cols = len(canonical_columns)
            keep: list[str] = []
            for csv_file in csv_files_to_process:
                base_name = os.path.basename(csv_file)
                out_name = base_name.replace(".csv", "_normalized.h5")
                out_path = os.path.join(base_output_folder, out_name)

                if not os.path.exists(out_path):
                    keep.append(csv_file)
                    continue

                ok, n_rows, reason = inspect_normalized_h5(
                    out_path,
                    expected_n_cols=expected_n_cols,
                    max_rows=max_rows,
                    require_full_length=require_full_length,
                )
                if not ok:
                    invalid_outputs_reprocessed += 1
                    logger.warning(
                        "Existing output for %s failed integrity check (%s); will reprocess.",
                        os.path.basename(out_path),
                        reason,
                    )
                    keep.append(csv_file)
                    continue

                # If labels are enabled and a label CSV exists, require a valid label output too.
                if labels_input_folder and labels_output_folder:
                    label_csv_path, label_h5_path = label_paths_for_csv(
                        csv_base_name=base_name,
                        labels_input_dir=labels_input_folder,
                        labels_output_dir=labels_output_folder,
                    )
                    if label_csv_path is not None:
                        if label_h5_path is None or not os.path.exists(label_h5_path):
                            invalid_outputs_reprocessed += 1
                            logger.warning(
                                "Missing label output for %s; will reprocess.",
                                os.path.basename(out_path),
                            )
                            keep.append(csv_file)
                            continue
                        ok_lab, lab_reason = inspect_label_h5(label_h5_path, expected_len=n_rows)
                        if not ok_lab:
                            invalid_outputs_reprocessed += 1
                            logger.warning(
                                "Label output for %s failed integrity check (%s); will reprocess.",
                                os.path.basename(label_h5_path),
                                lab_reason,
                            )
                            keep.append(csv_file)
                            continue

                # Both outputs exist and look complete.
                existing_outputs_skipped += 1

            csv_files_to_process = keep
            logger.info(
                "Clobber disabled: skipped %d completed files; %d will be processed.",
                existing_outputs_skipped,
                len(csv_files_to_process),
            )
            if invalid_outputs_reprocessed:
                logger.warning(
                    "Clobber disabled: %d existing outputs failed integrity checks and will be regenerated.",
                    invalid_outputs_reprocessed,
                )

        if not csv_files_to_process:
            logger.info("No files to normalize (all outputs exist, or all files were schema-mismatched).")

            if manifest is not None:
                manifest["results"] = {
                    "mode": norm_params_mode,
                    "norm_only": False,
                    "canonical_column_count": len(canonical_columns),
                    "input_csv_total": len(all_csv_files),
                    "input_csv_good_schema": len(all_csv_files_good),
                    "schema_bad_count": len(bad_schema),
                    "schema_bad_examples": summarize_bad_schema(bad_schema),
                    "existing_outputs_skipped": existing_outputs_skipped,
                    "invalid_outputs_reprocessed": invalid_outputs_reprocessed,
                    "normalized": {
                        "attempted": 0,
                        "success": 0,
                        "skipped_short": 0,
                        "failed": 0,
                        "labels_saved": 0,
                    },
                    "skipped_short_normparams_count": len(skipped_short_normparams),
                    "size_check": size_check_info,
                    "processing_log_file": os.path.abspath(processing_log_path)
                    if processing_log_path
                    else None,
                }
            return

        means = norm_df["Mean"].to_numpy(dtype=np.float32, copy=False)
        stds = norm_df["StdDev"].to_numpy(dtype=np.float32, copy=False)

        logger.info("Normalizing %d files with %d workers...", len(csv_files_to_process), num_workers)

        if num_workers <= 1:
            init_normalize_worker(
                canonical_columns=canonical_columns,
                means=means,
                stds=stds,
                edge_skip=edge_skip,
                max_rows=max_rows,
                require_full_length=require_full_length,
                output_dir=base_output_folder,
                labels_input_dir=labels_input_folder,
                labels_output_dir=labels_output_folder,
                log_queue=log_queue,
                log_level=cfg.log_level,
            )
            results = [normalize_one_csv(f) for f in csv_files_to_process]
        else:
            with Pool(
                processes=num_workers,
                initializer=init_normalize_worker,
                initargs=(
                    canonical_columns,
                    means,
                    stds,
                    edge_skip,
                    max_rows,
                    require_full_length,
                    base_output_folder,
                    labels_input_folder,
                    labels_output_folder,
                    log_queue,
                    cfg.log_level,
                ),
            ) as pool:
                results = pool.map(normalize_one_csv, csv_files_to_process)

        successes = [r for r in results if r.get("status") == "success"]
        failed = [r for r in results if r.get("status") == "failed"]
        skipped_short = [r for r in results if r.get("status") == "skipped_short"]

        labels_saved_count = sum(1 for r in successes if r.get("labels_saved"))
        logger.info(
            "Done. Success: %d | Skipped(short): %d | Failed: %d | Labels saved: %d",
            len(successes),
            len(skipped_short),
            len(failed),
            labels_saved_count,
        )

        # Update processing log file.
        if processing_log_path:
            with open(processing_log_path, "a", encoding="utf-8") as f:
                f.write("   Processing Results:\n")
                f.write(f"   - Successfully normalized: {len(successes)} files\n")
                f.write(f"   - Skipped (short): {len(skipped_short)} files\n")
                f.write(f"   - Failed to normalize: {len(failed)} files\n\n")

        # finalize manifest results
        if manifest is not None:
            manifest["results"] = {
                "mode": norm_params_mode,
                "norm_only": False,
                "canonical_column_count": len(canonical_columns),
                "input_csv_total": len(all_csv_files),
                "input_csv_good_schema": len(all_csv_files_good),
                "schema_bad_count": len(bad_schema),
                "schema_bad_examples": summarize_bad_schema(bad_schema),
                "existing_outputs_skipped": existing_outputs_skipped,
                "skipped_short_normparams_count": len(skipped_short_normparams),
                "size_check": size_check_info,
                "normalized": {
                    "attempted": len(csv_files_to_process),
                    "success": len(successes),
                    "skipped_short": len(skipped_short),
                    "failed": len(failed),
                    "labels_saved": labels_saved_count,
                    "failed_examples": [
                        {
                            "file": os.path.basename(str(r.get("file"))),
                            "error": r.get("error"),
                        }
                        for r in failed[:25]
                    ],
                    "skipped_short_examples": [
                        {
                            "file": os.path.basename(str(r.get("file"))),
                            "rows": r.get("rows"),
                        }
                        for r in skipped_short[:25]
                    ],
                },
                "processing_log_file": os.path.abspath(processing_log_path)
                if processing_log_path
                else None,
            }

    except Exception as e:
        logger.exception("Fatal error in keyprenorm")
        if manifest is not None:
            manifest["status"] = "failed"
            manifest["error"] = str(e)
        raise
    finally:
        end_time = datetime.now()
        if manifest is not None and manifest_path is not None:
            manifest["finished_at"] = end_time.isoformat()
            manifest["duration_seconds"] = (end_time - start_time).total_seconds()
            if manifest.get("status") == "running":
                manifest["status"] = "complete"
            atomic_write_json(manifest_path, manifest)

        # -------------------------
        # Registry update (best-effort)
        # -------------------------
        if bool(getattr(cfg, "registry_enabled", False)) and registry_csv_path and registry_project_root:
            try:
                from .registry_update import update_registry_from_keyprenorm_outputs

                update_registry_from_keyprenorm_outputs(
                    registry_csv=Path(registry_csv_path),
                    project_root=Path(registry_project_root),
                    keypre_csv_files=[Path(p) for p in all_csv_files],
                    normalized_h5_dir=Path(base_output_folder),
                    norm_params_path=Path(norm_params_path),
                    lock_timeout_sec=int(getattr(cfg, "registry_lock_timeout_sec", 120)),
                    lock_stale_sec=int(getattr(cfg, "registry_lock_stale_sec", 60 * 60)),
                )
            except Exception:
                logger.exception("Registry update (keyprenorm) failed")

        # Stop listener last (after final logs/manifest write)
        try:
            listener.stop()
        except Exception:
            pass
