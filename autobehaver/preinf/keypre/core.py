"""Discover DeepLabCut CSVs and coordinate KeyPre workers and run manifests.

Recreate worker pools between configured file chunks.
"""

from __future__ import annotations

from autobehaver.cli_summary import reported_stage, completion_details, validation_details

from autobehaver.progress import progress

import argparse
import gc
import glob
import logging
import os
import sys
import time
from collections import Counter
from dataclasses import asdict
from datetime import datetime
from ..execution import plan_execution, worker_results as execute_workers
from pathlib import Path

import numpy as np
import pandas as pd

from autobehaver.validation import StageValidationReport
from autobehaver.resource_monitor import record_work
from autobehaver.completion import verified_run, file_signature

from .config import KeypreConfig
from .dlc_io import load_dlc_csv_flat
from .logging_utils import configure_logging
from .manifest import _atomic_write_json, _build_run_manifest
from .utils import find_processing_log
from .worker import init_worker, process_csv_file
from .contract import inspect_keypre_output, output_path

logger = logging.getLogger(__name__)


@reported_stage("KeyPre")
@verified_run
def run_keypre(*, cfg: KeypreConfig, args: argparse.Namespace | None = None) -> None:
    """Derive KeyPre features for selected DeepLabCut CSVs and validate the outputs."""

    # Logging first, so subsequent messages go through logging
    configure_logging(cfg.log_level)

    start_time_wall = time.time()
    start_time_iso = datetime.now().isoformat()

    input_folder = cfg.input_dir
    if not os.path.isdir(input_folder):
        raise FileNotFoundError(f"Input folder does not exist: {input_folder}")

    # Resolve input CSVs
    # Priority:
    # registry (if enabled)
    # filesystem scan of input_folder (prefers input/DLC subfolder when present)
    used_registry = False
    registry_csv_path = None
    registry_project_root = None

    csv_files: list[str] = []

    if bool(getattr(cfg, "registry_enabled", False)) and getattr(cfg, "registry_csv", None):
        registry_csv_path = str(getattr(cfg, "registry_csv"))
        reg_path = Path(registry_csv_path).expanduser().resolve()
        if not reg_path.exists():
            raise FileNotFoundError(
                f"Registry enabled but CSV not found: {reg_path}. "
                "Run autocropper + dlc first (to create registry entries), or disable registry for this run."
            )

        registry_project_root = reg_path.parent

        try:
            from autobehaver.registry import read_registry, resolve_registry_path  # lazy import; pandas required
        except Exception as e:
            raise RuntimeError(
                "Failed to import registry helpers (pandas required). "
                "Install pandas in this environment or disable registry."
            ) from e

        df = read_registry(reg_path)

        if not df.empty and "dlc_csv_relpath" in df.columns:
            # Handle null active flags without changing compatibility defaults.
            try:
                import pandas as pd  # type: ignore
            except Exception:  # pragma: no cover
                pd = None  # type: ignore

            def _is_active(val: object) -> bool:
                if val is None:
                    return True
                if pd is not None:
                    try:
                        if pd.isna(val):  # type: ignore[attr-defined]
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
                df2 = df[df["dlc_csv_relpath"].notna()].copy()
            except Exception:
                df2 = df.copy()

            if "active" in df2.columns:
                try:
                    df2 = df2[df2["active"].apply(_is_active)]
                except Exception:
                    pass

            for _, registry_row in df2.iterrows():
                rel = str(registry_row["dlc_csv_relpath"]).replace("\\", "/")
                if not rel.strip():
                    continue
                p = resolve_registry_path(registry_project_root, registry_row, "dlc_csv_relpath")
                if not str(p).lower().endswith(".csv"):
                    continue
                csv_files.append(str(p))

        csv_files = sorted(set(csv_files))
        used_registry = True
        csv_source_folder = input_folder
        logger.info("Using registry for keypre inputs: %s (%d CSV files)", reg_path, len(csv_files))
    else:
        # Prefer DLC subfolder if present
        csv_source_folder = os.path.join(input_folder, "DLC")
        if os.path.exists(csv_source_folder):
            logger.info(
                "Found DLC subfolder; using it as source for CSV files: %s", csv_source_folder
            )
        else:
            csv_source_folder = input_folder
            logger.info(
                "No DLC subfolder found; using input folder directly: %s", input_folder
            )

        # Discover input CSV files
        pattern = os.path.join(csv_source_folder, "*.csv")
        csv_files = glob.glob(pattern)
        csv_files.sort()

    from ..autocropper.selection import filter_included_paths, propagate_selection
    selection_sources = {Path(csv_source_folder), *(Path(path).parent for path in csv_files)}
    output_selection_dir = (cfg.output_dir or os.path.join(
        input_folder, "raw_positional_calcs" if getattr(cfg, "raw_only", False) else "positional_calcs"))
    csv_files = filter_included_paths(csv_files)

    # Optional substring filter (bundling without subfolders)
    if getattr(cfg, 'file_filter', None):
        sub = str(getattr(cfg, 'file_filter')).lower()
        before = len(csv_files)
        from autobehaver.portable_paths import original_name
        csv_files = [p for p in csv_files if sub in original_name(p).lower()]
        csv_files.sort()
        logger.info("Applied file_filter='%s': %d -> %d CSV files", sub, before, len(csv_files))

    from autobehaver.batch_inputs import select_batch_inputs
    csv_files = select_batch_inputs(csv_files, getattr(cfg, "file_list", None))
    missing = [p for p in csv_files if not Path(p).is_file()]
    if missing:
        raise FileNotFoundError(f'Selected DLC CSV inputs are missing at their declared paths: {missing[:25]}')

    if not csv_files:
        if used_registry:
            raise FileNotFoundError(
                "No DLC CSV files discovered from registry. "
                "Check that the registry has dlc_csv_relpath entries (run the DLC stage), "
                "or disable registry for this run."
            )
        raise FileNotFoundError(f"No DLC CSV files found matching: {pattern}")

    # Determine output directory.
    if cfg.output_dir:
        out_dir = cfg.output_dir
    else:
        out_dir_name = "raw_positional_calcs" if getattr(cfg, "raw_only", False) else "positional_calcs"
        out_dir = os.path.join(input_folder, out_dir_name)
    completion_details(output=out_dir)
    from autobehaver.portable_paths import original_name, recording_name
    planned = {}
    for csv_file in csv_files:
        target = recording_name(Path(original_name(csv_file)).stem).casefold()
        if target in planned and planned[target] != csv_file:
            raise ValueError(f'Multiple DLC inputs map to the same recording output: {planned[target]}; {csv_file}. Select one DLC result per recording.')
        planned[target] = csv_file
    if cfg.dry_run:
        completion_details(status='dry run complete')
        logger.info('Dry run: discovered %d CSV file(s); planned output: %s', len(csv_files), out_dir)
        return
    os.makedirs(out_dir, exist_ok=True)
    propagate_selection(selection_sources, output_selection_dir)

    # Optional: pre-skip jobs if outputs already exist and clobber is False.
    # Validate compatible outputs before scheduling workers.
    pre_skipped_existing = 0
    invalid_existing_reprocessed = 0
    pre_skipped_files: set[str] = set()
    validated_rows = {}
    validated_inspections = {}
    csv_files_all = list(csv_files)
    if (not cfg.clobber) and csv_files:
        to_process: list[str] = []
        for p in csv_files:
            out_path = str(output_path(p, out_dir))
            if os.path.exists(out_path):
                ok, reason, _rows, _columns = inspect_keypre_output(p, out_path, movie_dim=cfg.movie_dim, raw_only=cfg.raw_only, require_receipt=True)
                if ok:
                    validated_rows[p] = _rows
                    validated_inspections[p] = (file_signature(p), file_signature(out_path),
                                               (ok, reason, _rows, _columns))
                    pre_skipped_existing += 1
                    pre_skipped_files.add(p)
                else:
                    invalid_existing_reprocessed += 1
                    logger.warning(
                        "Existing keypre output failed validation (%s); regenerating: %s",
                        reason,
                        out_path,
                    )
                    to_process.append(p)
            else:
                to_process.append(p)
        if pre_skipped_existing:
            logger.info(
                "Pre-skip enabled (clobber disabled): %d/%d inputs already have outputs in %s",
                pre_skipped_existing,
                len(csv_files),
                out_dir,
            )
        csv_files = to_process

    # Build a Namespace-like object for the manifest builder if caller didn't provide args
    if args is None:
        allowed = {
            "input_dir",
            "num_workers",
            "memory_gb",
            "clobber",
            "movie_dim",
            "output_dir",
            "dry_run",
            "log_level",
            "raw_only",
        }
        args = argparse.Namespace(**{k: v for k, v in asdict(cfg).items() if k in allowed})

    plan = plan_execution("keypre", files=len(csv_files), columns=84 if cfg.raw_only else 829,
                          num_workers=cfg.num_workers, memory_gb=cfg.memory_gb)
    # Run manifest (reproducibility)
    manifest_path, manifest = _build_run_manifest(
        args=args,
        input_folder=input_folder,
        csv_source_folder=csv_source_folder,
        out_dir=out_dir,
        csv_files=csv_files_all,
        start_time_iso=start_time_iso,
        entrypoint=sys.argv[0] if sys.argv else None,
    )
    manifest["execution_plan"] = asdict(plan)
    _atomic_write_json(manifest_path, manifest)
    logger.info("Run manifest: %s", manifest_path)

    # Share only the minimal config required by workers
    run_config = {
        "output_dir": out_dir,
        "chunk_rows": plan.chunk_rows,
        "clobber": cfg.clobber,
        "movie_dim": cfg.movie_dim,
        "raw_only": bool(getattr(cfg, "raw_only", False)),
    }

    # Make sure the main process uses the same run-time config as workers.
    try:
        from .worker import set_run_config

        set_run_config(run_config)
    except Exception:
        # Do not fail the run if something odd happens here.
        logger.debug("Unable to set main-process run config", exc_info=True)

    logger.info(
        "Starting keypoint preprocessing: %d CSV files | num_workers=%d | clobber=%s | movie_dim=%d | raw_only=%s",
        len(csv_files),
        plan.workers,
        cfg.clobber,
        cfg.movie_dim,
        bool(getattr(cfg, "raw_only", False)),
    )

    # Optional: write to processing log if it exists
    log_file = find_processing_log(os.path.dirname(input_folder))
    if log_file:
        with open(log_file, "a", encoding="utf-8") as f:
            f.write("\n6. Keypoints Pre-processing\n")
            f.write(f"   Started at: {start_time_iso}\n")
            f.write(f"   Input folder: {input_folder}\n")
            f.write(f"   CSV source folder: {csv_source_folder}\n")
            f.write(f"   Output folder: {out_dir}\n")
            f.write(f"   num_workers: {plan.workers}\n")
            f.write(f"   clobber: {cfg.clobber}\n")
            f.write(f"   movie_dim: {cfg.movie_dim}\n\n")
            f.write(f"   raw_only: {bool(getattr(cfg, 'raw_only', False))}\n\n")

    status = "complete"
    error_msg: str | None = None

    # Chunked pool recreation strategy (prevents long-run memory growth)
    chunk_size = int(cfg.chunk_size)
    file_chunks = [csv_files[i : i + chunk_size] for i in range(0, len(csv_files), chunk_size)]

    total_successful = 0
    total_failed = 0
    total_skipped = 0
    failed_jobs: list[dict] = []
    worker_results: dict[str, dict] = {}
    output_csv_count = None

    try:
        for i, chunk in enumerate(file_chunks):
            logger.info("Processing chunk %d/%d (%d files)", i + 1, len(file_chunks), len(chunk))
            results = execute_workers(process_csv_file, chunk, num_workers=plan.workers,
                                      initializer=init_worker, initargs=(run_config, cfg.log_level),
                                      tasks_per_worker=min(16, int(cfg.pool_maxtasksperchild)))

            for r in progress(results, total=len(chunk), desc=f'KeyPre files (group {i+1}/{len(file_chunks)})'):
                input_path = str(r.get("file", ""))
                if input_path:
                    worker_results[input_path] = r
                    if r.get("inspection"):
                        validated_inspections[input_path] = r["inspection"]
                if r.get("status") == "success":
                    total_successful += 1
                elif r.get("status") == "failed":
                    total_failed += 1
                    failed_jobs.append(r)
                elif r.get("status") == "skipped":
                    total_skipped += 1

            del results
            gc.collect()

        validation = StageValidationReport.for_items(
            "keypre",
            csv_files_all,
            metadata={"output_directory": os.path.abspath(out_dir)},
        )
        output_destinations = [
            os.path.abspath(
                os.path.join(
                    out_dir,
                    os.path.basename(input_csv).replace(".csv", "_positional_calcs.csv"),
                )
            )
            for input_csv in csv_files_all
        ]
        duplicate_destinations = sorted(
            path for path, count in Counter(output_destinations).items() if count > 1
        )
        if duplicate_destinations:
            validation.add_issue(
                "output_path_collision",
                "error",
                f"{len(duplicate_destinations)} output path(s) are shared by multiple inputs.",
                details={"examples": duplicate_destinations[:25]},
            )

        valid_registry_inputs = []
        for input_csv in csv_files_all:
            output_csv = str(output_path(input_csv, out_dir))
            result = worker_results.get(input_csv)
            if result is not None and result.get("status") == "failed":
                validation.add_outcome(
                    input_csv,
                    "failed",
                    reason=str(result.get("error") or "worker failed"),
                    details={"output": output_csv},
                )
                continue
            expected_rows = result.get('n_rows') if result is not None else validated_rows.get(input_csv)
            ok, reason, rows, columns = inspect_keypre_output(input_csv, output_csv,
                movie_dim=cfg.movie_dim, raw_only=cfg.raw_only, require_receipt=True,
                expected_rows=expected_rows, cached_inspection=validated_inspections.get(input_csv))
            if not ok:
                validation.add_outcome(
                    input_csv,
                    "failed",
                    reason=reason,
                    details={"output": output_csv},
                )
                continue
            reused = input_csv in pre_skipped_files or (
                result is not None and result.get("status") == "skipped"
            )
            valid_registry_inputs.append(input_csv)
            validation.add_outcome(
                input_csv,
                "reused" if reused else "succeeded",
                reason="existing output passed integrity checks" if reused else None,
                details={"output": output_csv, "rows": rows, "columns": columns},
            )

        if (
            bool(getattr(cfg, "registry_enabled", False))
            and registry_csv_path
            and registry_project_root
        ):
            try:
                from autobehaver import __version__ as autobehaver_version
                from .registry_update import update_registry_from_keypre_outputs

                update_registry_from_keypre_outputs(
                    registry_csv=Path(registry_csv_path),
                    project_root=Path(registry_project_root),
                    dlc_csv_files=[Path(p) for p in valid_registry_inputs],
                    output_dir=Path(out_dir),
                    keypre_version=str(autobehaver_version),
                    lock_timeout_sec=int(getattr(cfg, "registry_lock_timeout_sec", 120)),
                    lock_stale_sec=int(getattr(cfg, "registry_lock_stale_sec", 60 * 60)),
                )
            except Exception as exc:
                logger.exception("Registry update (keypre) failed")
                validation.add_issue(
                    "registry_update_failed",
                    "error",
                    "Keypre outputs were created, but the project registry could not be updated.",
                    details={"error": str(exc)},
                )

        validation_path = os.path.join(out_dir, "KEYPRE_VALIDATION.json")
        run_validation_path = os.path.join(out_dir, f"keypre_validation_{manifest['run_id']}.json")
        validation.metadata["run_id"] = manifest["run_id"]
        validation.write(run_validation_path)
        from autobehaver.completion import preserve_previous
        preserve_previous(validation_path)
        validation.write(validation_path)
        manifest["validation_report"] = run_validation_path
        manifest["input_outcomes"] = validation.to_dict()["outcomes"]
        validation_details(validation, unit="Files", output=out_dir)
        validation.raise_for_errors()

        # Ensure output files exist
        output_csv_count = len(glob.glob(os.path.join(out_dir, "*.csv")))
        if output_csv_count == 0:
            raise FileNotFoundError(
                "No processed CSV files found. Please check the processing step."
            )

        # Write final processing results to log (optional)
        if log_file:
            with open(log_file, "a", encoding="utf-8") as f:
                f.write("   Processing Results:\n")
                f.write(f"   - Successfully processed: {total_successful} files\n")
                f.write(f"   - Failed to process: {total_failed} files\n")
                f.write(
                    f"   - Skipped existing: {total_skipped + int(pre_skipped_existing)} files "
                    f"(pre-skip={int(pre_skipped_existing)}, worker-skip={total_skipped})\n\n"
                )

                if failed_jobs:
                    f.write("   Failed files:\n")
                    for result in failed_jobs:
                        f.write(
                            f"   - {os.path.basename(result.get('file',''))}: {result.get('error')}\n"
                        )
                    f.write("\n")

        logger.info("Validated all requested positional calculations: %s", out_dir)

        duration = time.time() - start_time_wall
        minutes = int(duration // 60)
        seconds = duration % 60
        logger.info("Finalizing processing records.")
        logger.info("Total execution time: %d minutes and %.2f seconds.", minutes, seconds)

    except Exception as e:
        status = "failed"
        error_msg = repr(e)
        logger.exception("Fatal error during keypoint preprocessing")
        raise

    finally:
        finished_at = datetime.now().isoformat()
        record_work("keypre_files", completed=total_successful, failed=total_failed,
                    skipped=total_skipped + int(pre_skipped_existing))
        duration_seconds = time.time() - start_time_wall

        manifest["status"] = status
        manifest["finished_at"] = finished_at
        manifest["duration_seconds"] = duration_seconds
        manifest["error"] = error_msg
        manifest["results"] = {
            "total_input_csv": len(csv_files_all),
            "total_successful": total_successful,
            "total_failed": total_failed,
            "total_skipped": total_skipped + int(pre_skipped_existing),
            "pre_skipped_existing": int(pre_skipped_existing),
            "invalid_existing_reprocessed": int(invalid_existing_reprocessed),
            "output_csv_count": output_csv_count,
            "validation_report": manifest.get("validation_report"),
            "chunk_size": chunk_size,
            "chunk_count": len(file_chunks),
            "maxtasksperchild": int(cfg.pool_maxtasksperchild),
            "processing_log_file": os.path.abspath(log_file) if log_file else None,
            "failed_jobs": [
                {
                    "file": os.path.basename(r.get("file", "")),
                    "error": r.get("error"),
                }
                for r in failed_jobs
            ],
        }

        try:
            _atomic_write_json(manifest_path, manifest)
            logger.info("Finalized run manifest: %s", manifest_path)
        except Exception:
            # Don't fail the run if manifest can't be written
            completion_details(status="complete with warnings" if status != "failed" else "failed")
            logger.exception("Failed to write run manifest: %s", manifest_path)
