"""Coordinate schema validation, normalization, and output verification for KeyPreNorm.

Reject selected inputs with inconsistent schemas before processing. Record
run manifests and original-frame metadata alongside normalized HDF5 outputs.
"""

from __future__ import annotations

from autobehaver.cli_summary import reported_stage, completion_details, validation_details

from autobehaver.progress import progress

import argparse
from autobehaver.resource_monitor import record_work
import glob
import logging
import os
from autobehaver.portable_paths import normalized_path, original_name
from datetime import datetime
from dataclasses import asdict
from ..execution import plan_execution, worker_results
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

from autobehaver.validation import StageValidationReport
from autobehaver.completion import verified_run, file_signature, file_digest

from . import __stage_version__
from .config import KeypreNormConfig
from .inspect_outputs import inspect_label_h5, inspect_normalized_h5
from .labels import label_paths_for_csv
from .logging_utils import setup_logging_queue
from .manifest import atomic_write_json, build_run_manifest
from .normparams import calculate_normalization_params
from .normalize import init_normalize_worker, normalize_one_csv
from .identity import normalization_identity, label_identity
from .schema import (
    filter_files_by_schema,
    find_processing_log,
    read_csv_header_columns,
    resolve_glob_pattern,
    summarize_bad_schema,
)
logger = logging.getLogger("keyprenorm")


def _source_csv_rows(csv_file: str) -> int:
    """Count source rows by reading only the first CSV column."""
    with pd.read_csv(csv_file, header=0, usecols=[0], chunksize=4096) as reader:
        return sum(len(frame) for frame in reader)


def _expected_normalized_rows(csv_file: str, *, edge_skip: int) -> int:
    """Return the source row count after trimming both preprocessing edges."""
    return max(0, _source_csv_rows(csv_file) - 2 * int(edge_skip))


def _validate_normalization_parameters(norm_df: pd.DataFrame, *, source: str) -> None:
    """Require ordered, unique feature names and finite means and nonnegative deviations."""
    required = ["Header", "Mean", "StdDev"]
    if list(norm_df.columns) != required:
        raise ValueError(
            f"Normalization parameters {source} must have exactly these columns in order: {required}"
        )
    headers = norm_df["Header"].map(lambda value: str(value).strip())
    if headers.empty or headers.eq("").any():
        raise ValueError(f"Normalization parameters {source} contain blank feature names")
    if headers.duplicated().any():
        duplicates = sorted(set(headers[headers.duplicated(keep=False)].tolist()))[:25]
        raise ValueError(
            f"Normalization parameters {source} contain duplicate feature names: {duplicates}"
        )
    try:
        means = pd.to_numeric(norm_df["Mean"], errors="raise").to_numpy(dtype=np.float64)
        stddevs = pd.to_numeric(norm_df["StdDev"], errors="raise").to_numpy(dtype=np.float64)
    except Exception as exc:
        raise ValueError(f"Normalization parameters {source} contain non-numeric values") from exc
    if not np.isfinite(means).all() or not np.isfinite(stddevs).all():
        raise ValueError(f"Normalization parameters {source} contain non-finite values")
    if (stddevs < 0).any():
        raise ValueError(f"Normalization parameters {source} contain negative standard deviations")


def _write_keyprenorm_validation(
    *,
    all_csv_files: list[str],
    canonical_columns: list[str],
    base_output_folder: str,
    edge_skip: int,
    bad_schema: list[Dict[str, object]],
    results: list[Dict[str, object]] | None = None,
    reused_files: set[str] | None = None,
    norm_only: bool = False,
    aborted_reason: str | None = None,
    labels_input_folder: str | None = None,
    labels_output_folder: str | None = None,
    verified_outputs: dict | None = None,
    normalization_parameters: tuple | None = None,
) -> StageValidationReport:
    """Build per-input outcomes after checking normalized features and any paired labels."""
    report = StageValidationReport.for_items(
        "keyprenorm",
        all_csv_files,
        metadata={
            "output_directory": os.path.abspath(base_output_folder),
            "norm_only": bool(norm_only),
        },
    )
    bad_by_file = {str(item.get("file")): item for item in bad_schema}
    result_by_file = {str(item.get("file")): item for item in (results or [])}
    reused = set(reused_files or set())

    for csv_file in all_csv_files:
        if csv_file in bad_by_file:
            item = bad_by_file[csv_file]
            report.add_outcome(
                csv_file,
                "failed",
                reason=f"input schema validation failed: {item.get('reason')}",
                details={
                    "missing": list(item.get("missing") or [])[:25],
                    "extra": list(item.get("extra") or [])[:25],
                    "error": item.get("error"),
                },
            )
            continue

        if aborted_reason is not None:
            report.add_outcome(csv_file, "failed", reason=aborted_reason)
            continue

        if norm_only:
            report.add_outcome(
                csv_file, "succeeded",
                reason="input contributed to validated normalization parameters",
            )
            continue

        result = result_by_file.get(csv_file)
        if result is not None and result.get("status") == "failed":
            reason = result.get("error") or "normalization failed"
            report.add_outcome(csv_file, "failed", reason=str(reason))
            continue

        output_path = str(normalized_path(csv_file, base_output_folder))
        try:
            cached = (verified_outputs or {}).get(csv_file)
            if cached and os.path.isfile(output_path) and file_digest(output_path) != cached['output_digest']:
                report.add_outcome(csv_file, 'failed',
                    reason='normalized output content changed after initial verification',
                    details={'output': output_path})
                continue
            cache_current = bool(cached and os.path.isfile(output_path) and file_signature(csv_file) == cached['source_signature']
                                 and file_signature(output_path) == cached['output_signature']
                                 and file_digest(output_path) == cached['output_digest'])
            source_rows = (cached['source_rows'] if cache_current and 'source_rows' in cached else
                           int(result['source_rows']) if result is not None and 'source_rows' in result else
                           _source_csv_rows(csv_file))
            expected_rows = max(0, source_rows - 2 * int(edge_skip))
            expected_identity = (normalization_identity(csv_file, canonical_columns,
                *normalization_parameters, edge_skip) if normalization_parameters is not None else None)
        except Exception as exc:
            report.add_outcome(csv_file, "failed", reason=f"input CSV is unreadable: {exc}")
            continue
        if cache_current and (expected_identity is None or expected_identity == cached['identity']):
            ok, rows, reason = True, cached['rows'], 'ok'
        else:
            ok, rows, reason = inspect_normalized_h5(
                output_path, expected_n_cols=len(canonical_columns), expected_rows=expected_rows,
                expected_columns=canonical_columns, expected_identity=expected_identity,
                expected_source_rows=source_rows, expected_edge_skip=edge_skip)
        if not ok:
            report.add_outcome(
                csv_file,
                "failed",
                reason=f"normalized output failed integrity validation: {reason}",
                details={"output": output_path},
            )
            continue
        if labels_input_folder and labels_output_folder:
            label_csv_path, label_h5_path = label_paths_for_csv(
                csv_base_name=original_name(csv_file),
                labels_input_dir=labels_input_folder,
                labels_output_dir=labels_output_folder,
            )
            if label_csv_path is not None or (label_h5_path and os.path.exists(label_h5_path)):
                if label_h5_path is None or not os.path.exists(label_h5_path):
                    report.add_outcome(
                        csv_file,
                        "failed",
                        reason="source labels exist but the normalized label output is missing",
                        details={"output": output_path},
                    )
                    continue
                label_ok, label_reason = inspect_label_h5(label_h5_path, expected_len=rows,
                    expected_identity=label_identity(label_csv_path, expected_identity) if label_csv_path is not None and expected_identity is not None else None,
                    expected_feature_identity=expected_identity,
                    expected_source_rows=source_rows, expected_edge_skip=edge_skip)
                if not label_ok:
                    report.add_outcome(
                        csv_file,
                        "failed",
                        reason=f"normalized label output failed integrity validation: {label_reason}",
                        details={"output": label_h5_path},
                    )
                    continue
        is_reused = csv_file in reused
        report.add_outcome(
            csv_file,
            "reused" if is_reused else "succeeded",
            reason="existing output passed integrity checks" if is_reused else None,
            details={"output": output_path, "rows": rows, "columns": len(canonical_columns)},
        )

    return report


@reported_stage("Normalization")
@verified_run
def run_keyprenorm(*, cfg: KeypreNormConfig, args: argparse.Namespace) -> None:
    """Validate input schemas, calculate or load parameters, and publish normalized outputs."""

    base_input_folder = cfg.input_dir
    input_folder = base_input_folder
    if cfg.labels_dir and not Path(cfg.labels_dir).is_dir():
        raise FileNotFoundError(f'Explicit labels directory does not exist: {cfg.labels_dir}')

    num_workers = int(cfg.num_workers)
    edge_skip = int(cfg.edge_skip)
    batch_size = int(cfg.batch_size)

    if num_workers < 0:
        raise ValueError("--num_workers must be >= 0 (0 = auto)")

    if batch_size <= 0:
        raise ValueError("--batch_size must be >= 1")

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

    def _update_registry_after_valid_outputs() -> str | None:
        """Update normalized output paths and return an error message on failure."""
        if not (
            bool(getattr(cfg, "registry_enabled", False))
            and registry_csv_path
            and registry_project_root
        ):
            return None
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
        except Exception as exc:
            logger.exception("Registry update (keyprenorm) failed")
            return str(exc)
        return None

    def _finish_validation_report(
        report: StageValidationReport,
        *,
        update_registry: bool,
    ) -> None:
        """Publish run and latest validation reports, then raise for recorded errors."""
        if update_registry and report.status != "failed":
            registry_error = _update_registry_after_valid_outputs()
            if registry_error is not None:
                report.add_issue(
                    "registry_update_failed",
                    "error",
                    "KeypreNorm outputs were created, but the project registry could not be updated.",
                    details={"error": registry_error},
                )
        latest_report = Path(base_output_folder) / "KEYPRENORM_VALIDATION.json"
        if manifest is not None:
            report.metadata["run_id"] = manifest["run_id"]
            run_report = Path(base_output_folder) / f"keyprenorm_validation_{manifest['run_id']}.json"
            report.write(run_report)
            manifest["validation_report"] = str(run_report)
            manifest["input_outcomes"] = report.to_dict()["outcomes"]
        from autobehaver.completion import preserve_previous
        preserve_previous(latest_report)
        report.write(latest_report)
        validation_details(report, unit="Files", output=base_output_folder)
        report.raise_for_errors()

    try:
        # Discover inputs
        # Priority:
        # registry (if enabled)
        # filesystem scan of input_folder
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
                from autobehaver.registry import read_registry, resolve_registry_path
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

                files: list[str] = []
                for _, registry_row in df2.iterrows():
                    rel = str(registry_row["keypre_csv_relpath"]).replace("\\", "/")
                    if not rel.strip():
                        continue
                    pth = resolve_registry_path(registry_project_root, registry_row, "keypre_csv_relpath")
                    if not str(pth).lower().endswith(".csv"):
                        continue
                    files.append(str(pth))

                all_csv_files = sorted(set(files))

            used_registry = True
            logger.info("Using registry for keyprenorm inputs: %s (%d CSV files)", reg_path, len(all_csv_files))
        else:
            all_csv_files = sorted(glob.glob(csv_glob))

        from ..autocropper.selection import filter_included_paths, propagate_selection
        propagate_selection({Path(input_folder), *(Path(path).parent for path in all_csv_files)}, base_output_folder)
        all_csv_files = filter_included_paths(all_csv_files)

        # Optional substring filter (pipeline convenience)
        if getattr(cfg, "file_filter", None):
            needle = str(cfg.file_filter).lower()
            before = len(all_csv_files)
            all_csv_files = [f for f in all_csv_files if needle in original_name(f).lower()]
            logger.info("Applied file_filter='%s': %d/%d CSVs selected.", cfg.file_filter, len(all_csv_files), before)

        from autobehaver.batch_inputs import select_batch_inputs
        all_csv_files = select_batch_inputs(all_csv_files, getattr(cfg, "file_list", None))
        missing = [p for p in all_csv_files if not Path(p).is_file()]
        if missing:
            raise FileNotFoundError(f'Selected KeyPre CSV inputs are missing at their declared paths: {missing[:25]}')
        destinations = {}
        for source in all_csv_files:
            destination = str(normalized_path(source, base_output_folder)).casefold()
            if destination in destinations and destinations[destination] != source:
                raise ValueError(f'Normalization output collision: {destinations[destination]}; {source}')
            destinations[destination] = source

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
                raise FileNotFoundError(f'Explicit labels directory does not exist: {labels_input_folder}')

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
                f.write("   - Row policy: retain the full movie after both preprocessing edges\n\n")

        # Canonical columns and normalization parameters
        norm_params_mode = "calculate" if cfg.calculate_norm else "load"
        bad_schema: list[Dict[str, object]] = []

        if cfg.calculate_norm:
            first_cols, err = read_csv_header_columns(all_csv_files[0])
            if err is not None or first_cols is None:
                raise RuntimeError(
                    f"Could not read header from first CSV file: {all_csv_files[0]} ({err})"
                )
            canonical_columns = first_cols

            all_csv_files_good, bad_schema = filter_files_by_schema(all_csv_files, canonical_columns, max_examples=10)
        else:
            if not os.path.exists(norm_params_path):
                raise FileNotFoundError(
                    f"Normalization parameters file not found at: {norm_params_path}\n"
                    f"Run again with --calculate_norm to create it."
                )
            norm_df = pd.read_csv(norm_params_path)
            _validate_normalization_parameters(norm_df, source=norm_params_path)
            canonical_columns = norm_df["Header"].tolist()

            all_csv_files_good, bad_schema = filter_files_by_schema(all_csv_files, canonical_columns, max_examples=10)

            logger.info("Loaded normalization parameters from: %s", norm_params_path)
            logger.info("Canonical columns: %d", len(canonical_columns))

        if bad_schema:
            if manifest is not None:
                manifest["results"] = {
                    "mode": norm_params_mode,
                    "norm_only": bool(cfg.norm_only),
                    "input_csv_total": len(all_csv_files),
                    "input_csv_good_schema": len(all_csv_files_good),
                    "schema_bad_count": len(bad_schema),
                    "schema_bad_examples": summarize_bad_schema(bad_schema),
                }
            report = _write_keyprenorm_validation(
                all_csv_files=all_csv_files,
                canonical_columns=list(canonical_columns),
                base_output_folder=base_output_folder,
                edge_skip=edge_skip,
                bad_schema=bad_schema,
                aborted_reason="stage was not run because another selected input failed schema validation",
                labels_input_folder=labels_input_folder,
                labels_output_folder=labels_output_folder,
            )
            _finish_validation_report(report, update_registry=False)

        plan = plan_execution('keyprenorm', files=len(all_csv_files_good), columns=len(canonical_columns),
                              num_workers=num_workers, memory_gb=cfg.memory_gb)
        num_workers = plan.workers
        manifest['execution_plan'] = asdict(plan)
        atomic_write_json(manifest_path, manifest)

        if cfg.calculate_norm:
            norm_df = calculate_normalization_params(
                csv_files=all_csv_files_good,
                canonical_columns=canonical_columns,
                norm_params_path=norm_params_path,
                norm_params_dir=norm_params_dir,
                num_workers=num_workers,
                edge_skip=edge_skip,
                batch_size=batch_size,
                log_queue=log_queue,
                log_level=cfg.log_level,
                chunk_rows=plan.chunk_rows,
            )
            _validate_normalization_parameters(norm_df, source=norm_params_path)

        # If only calculating norm params, stop here
        if cfg.norm_only:
            completion_details(stage="Normalization parameters")
            logger.info("Parameters ready; validating and saving records. No HDF5 files requested.")

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
                    "processing_log_file": os.path.abspath(processing_log_path)
                    if processing_log_path
                    else None,
                }
            report = _write_keyprenorm_validation(
                all_csv_files=all_csv_files,
                canonical_columns=list(canonical_columns),
                base_output_folder=base_output_folder,
                edge_skip=edge_skip,
                bad_schema=bad_schema,
                norm_only=True,
                labels_input_folder=labels_input_folder,
                labels_output_folder=labels_output_folder,
            )
            _finish_validation_report(report, update_registry=False)
            return

        means = norm_df["Mean"].to_numpy(dtype=np.float32, copy=False)
        stds = norm_df["StdDev"].to_numpy(dtype=np.float32, copy=False)

        # Filter for existing completed outputs when clobber is disabled.
        csv_files_to_process = list(all_csv_files_good)
        existing_outputs_skipped = 0
        existing_output_files: set[str] = set()
        invalid_outputs_reprocessed = 0
        verified_outputs = {}
        if not cfg.clobber:
            expected_n_cols = len(canonical_columns)
            keep: list[str] = []
            for csv_file in csv_files_to_process:
                base_name = original_name(csv_file)
                out_path = str(normalized_path(csv_file, base_output_folder))

                if not os.path.exists(out_path):
                    keep.append(csv_file)
                    continue

                expected_identity = normalization_identity(csv_file, canonical_columns, means, stds,
                    edge_skip)
                source_rows = _source_csv_rows(csv_file)
                ok, n_rows, reason = inspect_normalized_h5(
                    out_path,
                    expected_n_cols=expected_n_cols,
                    expected_rows=max(0, source_rows - 2 * int(edge_skip)),
                    expected_columns=canonical_columns,
                    expected_identity=expected_identity,
                    expected_source_rows=source_rows,
                    expected_edge_skip=edge_skip,
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

                # Validate existing paired annotations even if their source CSV was withdrawn.
                if labels_input_folder and labels_output_folder:
                    label_csv_path, label_h5_path = label_paths_for_csv(
                        csv_base_name=base_name,
                        labels_input_dir=labels_input_folder,
                        labels_output_dir=labels_output_folder,
                    )
                    if label_csv_path is not None or (label_h5_path and os.path.exists(label_h5_path)):
                        if label_h5_path is None or not os.path.exists(label_h5_path):
                            invalid_outputs_reprocessed += 1
                            logger.warning(
                                "Missing label output for %s; will reprocess.",
                                os.path.basename(out_path),
                            )
                            keep.append(csv_file)
                            continue
                        ok_lab, lab_reason = inspect_label_h5(label_h5_path, expected_len=n_rows,
                            expected_identity=label_identity(label_csv_path, expected_identity) if label_csv_path is not None else None,
                            expected_feature_identity=expected_identity,
                            expected_source_rows=source_rows, expected_edge_skip=edge_skip)
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
                existing_output_files.add(csv_file)
                verified_outputs[csv_file] = dict(rows=n_rows, source_rows=source_rows, identity=expected_identity,
                    source_signature=file_signature(csv_file), output_signature=file_signature(out_path),
                    output_digest=file_digest(out_path))

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
            record_work("normalized_files", skipped=existing_outputs_skipped)

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
                        "failed": 0,
                        "labels_saved": 0,
                    },
                    "processing_log_file": os.path.abspath(processing_log_path)
                    if processing_log_path
                    else None,
                }
            report = _write_keyprenorm_validation(
                all_csv_files=all_csv_files,
                canonical_columns=list(canonical_columns),
                base_output_folder=base_output_folder,
                edge_skip=edge_skip,
                bad_schema=bad_schema,
                reused_files=existing_output_files,
                labels_input_folder=labels_input_folder,
                labels_output_folder=labels_output_folder,
                verified_outputs=verified_outputs,
                normalization_parameters=(means, stds),
            )
            _finish_validation_report(report, update_registry=True)
            return

        means = norm_df["Mean"].to_numpy(dtype=np.float32, copy=False)
        stds = norm_df["StdDev"].to_numpy(dtype=np.float32, copy=False)

        logger.info("Normalizing %d files with %d workers...", len(csv_files_to_process), num_workers)

        results = list(progress(worker_results(normalize_one_csv, csv_files_to_process, num_workers=num_workers,
            initializer=init_normalize_worker, initargs=(canonical_columns, means, stds, edge_skip,
                base_output_folder, labels_input_folder, labels_output_folder, log_queue,
                cfg.log_level, plan.chunk_rows)), total=len(csv_files_to_process), desc='Normalizing files'))
        for result in results:
            if result.get('status') == 'success' and result.get('verification'):
                verified_outputs[result['file']] = result['verification']

        successes = [r for r in results if r.get("status") == "success"]
        failed = [r for r in results if r.get("status") == "failed"]
        record_work("normalized_files", completed=len(successes), failed=len(failed),
                    skipped=existing_outputs_skipped)

        labels_saved_count = sum(1 for r in successes if r.get("labels_saved"))
        logger.info(
            "Workers returned: %d succeeded | %d failed | %d labels saved; final validation follows",
            len(successes),
            len(failed),
            labels_saved_count,
        )

        # Update processing log file.
        if processing_log_path:
            with open(processing_log_path, "a", encoding="utf-8") as f:
                f.write("   Processing Results:\n")
                f.write(f"   - Successfully normalized: {len(successes)} files\n")
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
                "normalized": {
                    "attempted": len(csv_files_to_process),
                    "success": len(successes),
                    "failed": len(failed),
                    "labels_saved": labels_saved_count,
                    "failed_examples": [
                        {
                            "file": os.path.basename(str(r.get("file"))),
                            "error": r.get("error"),
                        }
                        for r in failed[:25]
                    ],
                },
                "processing_log_file": os.path.abspath(processing_log_path)
                if processing_log_path
                else None,
            }

        report = _write_keyprenorm_validation(
            all_csv_files=all_csv_files,
            canonical_columns=list(canonical_columns),
            base_output_folder=base_output_folder,
            edge_skip=edge_skip,
            bad_schema=bad_schema,
            results=results,
            reused_files=existing_output_files,
            labels_input_folder=labels_input_folder,
            labels_output_folder=labels_output_folder,
            verified_outputs=verified_outputs,
            normalization_parameters=(means, stds),
        )
        _finish_validation_report(report, update_registry=True)

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

        # Stop listener last (after final logs/manifest write)
        try:
            listener.stop()
        except Exception:
            pass
