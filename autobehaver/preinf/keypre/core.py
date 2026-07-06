"""Keypre run orchestrator.

This module implements the folder-level execution engine for keypre:
  - DLC-folder discovery (prefers an input/DLC subfolder when present)
  - chunked multiprocessing with periodic Pool recreation
  - per-run JSON manifest writing

The per-file computations are performed by :mod:`autobehaver.preinf.keypre.worker`.
"""

from __future__ import annotations

import argparse
import gc
import glob
import logging
import os
import sys
import time
from dataclasses import asdict
from datetime import datetime
from multiprocessing import Pool
from pathlib import Path

from .config import KeypreConfig
from .logging_utils import configure_logging
from .manifest import _atomic_write_json, _build_run_manifest
from .utils import find_processing_log
from .worker import init_worker, process_csv_file

logger = logging.getLogger(__name__)


def run_keypre(*, cfg: KeypreConfig, args: argparse.Namespace | None = None) -> None:
    """Run keypre on a folder of DLC CSV outputs."""

    # Logging first, so subsequent messages go through logging
    configure_logging(cfg.log_level)

    start_time_wall = time.time()
    start_time_iso = datetime.now().isoformat()

    input_folder = cfg.input_dir
    if not os.path.isdir(input_folder):
        raise FileNotFoundError(f"Input folder does not exist: {input_folder}")

    # -------------------------
    # Resolve input CSVs
    # -------------------------
    # Priority:
    #   1) registry (if enabled)
    #   2) filesystem scan of input_folder (prefers input/DLC subfolder when present)
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
            from autobehaver.registry import read_registry  # lazy import; pandas required
        except Exception as e:
            raise RuntimeError(
                "Failed to import registry helpers (pandas required). "
                "Install pandas in this environment or disable registry."
            ) from e

        df = read_registry(reg_path)

        if not df.empty and "dlc_csv_relpath" in df.columns:
            # pandas is optional here; used only for isna convenience.
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

            rels = [
                str(x).replace("\\", "/")
                for x in df2["dlc_csv_relpath"].tolist()
                if str(x).strip() != ""
            ]

            for rel in rels:
                p = (registry_project_root / rel).resolve()
                if not p.exists():
                    # Fallback: registry may have been created on a different mirror.
                    cand1 = Path(input_folder) / Path(rel).name
                    cand2 = Path(input_folder) / "DLC" / Path(rel).name
                    if cand1.exists():
                        p = cand1.resolve()
                    elif cand2.exists():
                        p = cand2.resolve()
                if not p.exists():
                    continue
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

    # Optional substring filter (bundling without subfolders)
    if getattr(cfg, 'file_filter', None):
        sub = str(getattr(cfg, 'file_filter')).lower()
        before = len(csv_files)
        csv_files = [p for p in csv_files if sub in os.path.basename(p).lower()]
        csv_files.sort()
        logger.info("Applied file_filter='%s': %d -> %d CSV files", sub, before, len(csv_files))

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
    os.makedirs(out_dir, exist_ok=True)

    # Optional: pre-skip jobs if outputs already exist and clobber is False.
    # This avoids scheduling large pools for work we already have.
    pre_skipped_existing = 0
    csv_files_all = list(csv_files)
    if (not cfg.clobber) and csv_files:
        to_process: list[str] = []
        for p in csv_files:
            out_name = os.path.basename(p).replace(".csv", "_positional_calcs.csv")
            out_path = os.path.join(out_dir, out_name)
            if os.path.exists(out_path):
                pre_skipped_existing += 1
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

    if cfg.dry_run:
        logger.info(
            "Dry run: discovered %d CSV file(s) via %s",
            len(csv_files_all),
            "registry" if used_registry else "filesystem scan",
        )
        logger.info("CSV source folder: %s", csv_source_folder)
        logger.info("Planned output directory: %s", out_dir)
        logger.info(
            "Settings: num_workers=%d | clobber=%s | movie_dim=%d",
            cfg.num_workers,
            cfg.clobber,
            cfg.movie_dim,
        )
        if getattr(cfg, "raw_only", False):
            logger.info("Mode: raw_only (export raw x/y/likelihood only)")
        return

    # Build a Namespace-like object for the manifest builder if caller didn't provide args
    if args is None:
        allowed = {
            "input_dir",
            "num_workers",
            "clobber",
            "movie_dim",
            "output_dir",
            "dry_run",
            "log_level",
            "raw_only",
        }
        args = argparse.Namespace(**{k: v for k, v in asdict(cfg).items() if k in allowed})

    # --- Run manifest (reproducibility) ---
    manifest_path, manifest = _build_run_manifest(
        args=args,
        input_folder=input_folder,
        csv_source_folder=csv_source_folder,
        out_dir=out_dir,
        csv_files=csv_files,
        start_time_iso=start_time_iso,
        entrypoint=sys.argv[0] if sys.argv else None,
    )
    logger.info("Run manifest: %s", manifest_path)

    # Share only the minimal config required by workers
    run_config = {
        "output_dir": out_dir,
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
        cfg.num_workers,
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
            f.write(f"   num_workers: {cfg.num_workers}\n")
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
    output_csv_count = None

    try:
        for i, chunk in enumerate(file_chunks):
            logger.info("Processing chunk %d/%d (%d files)", i + 1, len(file_chunks), len(chunk))
            with Pool(
                processes=cfg.num_workers,
                maxtasksperchild=int(cfg.pool_maxtasksperchild),
                initializer=init_worker,
                initargs=(run_config, cfg.log_level),
            ) as pool:
                results = pool.map(process_csv_file, chunk)

            for r in results:
                if r.get("status") == "success":
                    total_successful += 1
                elif r.get("status") == "failed":
                    total_failed += 1
                    failed_jobs.append(r)
                elif r.get("status") == "skipped":
                    total_skipped += 1

            del results
            gc.collect()

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

        logger.info("Processed and saved all raw positional calculations: %s", out_dir)

        duration = time.time() - start_time_wall
        minutes = int(duration // 60)
        seconds = duration % 60
        logger.info("--- Processing Complete ---")
        logger.info("Total execution time: %d minutes and %.2f seconds.", minutes, seconds)

    except Exception as e:
        status = "failed"
        error_msg = repr(e)
        logger.exception("Fatal error during keypoint preprocessing")
        raise

    finally:
        finished_at = datetime.now().isoformat()
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
            "output_csv_count": output_csv_count,
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
            logger.exception("Failed to write run manifest: %s", manifest_path)

        # -------------------------
        # Registry update (best-effort)
        # -------------------------
        if bool(getattr(cfg, "registry_enabled", False)) and registry_csv_path and registry_project_root:
            try:
                from autobehaver import __version__ as autobehaver_version
                from .registry_update import update_registry_from_keypre_outputs

                update_registry_from_keypre_outputs(
                    registry_csv=Path(registry_csv_path),
                    project_root=Path(registry_project_root),
                    dlc_csv_files=[Path(p) for p in csv_files_all],
                    output_dir=Path(out_dir),
                    keypre_version=str(autobehaver_version),
                    lock_timeout_sec=int(getattr(cfg, "registry_lock_timeout_sec", 120)),
                    lock_stale_sec=int(getattr(cfg, "registry_lock_stale_sec", 60 * 60)),
                )
            except Exception:
                logger.exception("Registry update (keypre) failed")
