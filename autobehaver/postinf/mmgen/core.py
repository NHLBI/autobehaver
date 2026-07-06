"""Core runner for mmgen.

This module owns the configuration object and the single top-level callable
entrypoint for master-matrix generation. Runtime settings are now resolved once
per run and passed explicitly into the orchestrator instead of mutating
module-level variables across the package.
"""

from __future__ import annotations

import os
import random
import socket
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np

from . import pipeline
from .append_mode import run_mmgen_append_new_data
from .locks import DirLock
from .runtime import MMGenRuntimeSettings
from .shard_merge import merge_shard_temp_stores
from .matrix_parts import finalize_shard_temp_store, get_shard_output_dir, merge_matrix_part_files


@dataclass(init=False)
class MMGenConfig:
    """Configuration for master-matrix generation.

    Canonical path fields use the user-facing names:
    ``inference_csv_dir``, ``poscalc_dir``, and ``output_dir``.
    """

    inference_csv_dir: str
    poscalc_dir: str
    output_dir: str

    inf_tag: str = "_normalized_inference"
    min_bout: int = 5
    edge_skip: int = 5

    num_workers: int = 16
    random_seed: int = 42

    make_measurements: bool = True
    calculate_occupancy: bool = True
    calculate_postural_features: bool = True
    generate_quiver_plots: bool = True
    run_transition_statistics: bool = True
    make_detailed_transition_outputs: bool = True

    filter_features_by_bout_count: bool = True
    min_bout_count_for_features: int = 6
    min_bout_count_for_transitions: int = 6

    keep_temp_files: bool = True
    emit_master_matrix: bool = True
    emit_full_outputs: bool = True

    resume: bool = False
    finalize_only: bool = False
    process_only: bool = False
    file_filter: str | None = None
    input_batch_size: int = 20

    chamber_diameter_px: Optional[int] = 530
    mm_per_px: Optional[float] = 0.0169896744624207
    video_fps: Optional[float] = 40.0

    num_shards: int = 1
    shard_index: int = 0
    shard_by: str = "hash"
    merge_shards: bool = False
    matrix_first_shards: bool = False
    finalize_shard_matrix: bool = False
    merge_matrix_parts: bool = False
    append_new_data: bool = False
    plan_only: bool = False
    append_batch_id: str | None = None

    def __init__(
        self,
        *,
        inference_csv_dir: str | None = None,
        poscalc_dir: str | None = None,
        output_dir: str | None = None,
        inf_tag: str = "_normalized_inference",
        min_bout: int = 5,
        edge_skip: int = 5,
        num_workers: int = 16,
        random_seed: int = 42,
        make_measurements: bool = True,
        calculate_occupancy: bool = True,
        calculate_postural_features: bool = True,
        generate_quiver_plots: bool = True,
        run_transition_statistics: bool = True,
        make_detailed_transition_outputs: bool = True,
        filter_features_by_bout_count: bool = True,
        min_bout_count_for_features: int = 6,
        min_bout_count_for_transitions: int = 6,
        keep_temp_files: bool = True,
        emit_master_matrix: bool = True,
        emit_full_outputs: bool = True,
        resume: bool = False,
        finalize_only: bool = False,
        process_only: bool = False,
        file_filter: str | None = None,
        input_batch_size: int = 20,
        chamber_diameter_px: Optional[int] = 530,
        mm_per_px: Optional[float] = 0.0169896744624207,
        video_fps: Optional[float] = 40.0,
        num_shards: int = 1,
        shard_index: int = 0,
        shard_by: str = "hash",
        merge_shards: bool = False,
        matrix_first_shards: bool = False,
        finalize_shard_matrix: bool = False,
        merge_matrix_parts: bool = False,
        append_new_data: bool = False,
        plan_only: bool = False,
        append_batch_id: str | None = None,
        **kwargs: Any,
    ) -> None:
        if kwargs:
            unexpected = ", ".join(sorted(kwargs.keys()))
            raise TypeError(f"Unexpected MMGenConfig argument(s): {unexpected}")

        self.inference_csv_dir = str(inference_csv_dir or "")
        self.poscalc_dir = str(poscalc_dir or "")
        self.output_dir = str(output_dir or "")
        if self.inference_csv_dir.strip() == "":
            raise TypeError("MMGenConfig requires inference_csv_dir")
        if self.poscalc_dir.strip() == "":
            raise TypeError("MMGenConfig requires poscalc_dir")
        if self.output_dir.strip() == "":
            raise TypeError("MMGenConfig requires output_dir")

        self.inf_tag = str(inf_tag)
        self.min_bout = int(min_bout)
        self.edge_skip = int(edge_skip)
        self.num_workers = int(num_workers)
        self.random_seed = int(random_seed)
        self.make_measurements = bool(make_measurements)
        self.calculate_occupancy = bool(calculate_occupancy)
        self.calculate_postural_features = bool(calculate_postural_features)
        self.generate_quiver_plots = bool(generate_quiver_plots)
        self.run_transition_statistics = bool(run_transition_statistics)
        self.make_detailed_transition_outputs = bool(make_detailed_transition_outputs)
        self.filter_features_by_bout_count = bool(filter_features_by_bout_count)
        self.min_bout_count_for_features = int(min_bout_count_for_features)
        self.min_bout_count_for_transitions = int(min_bout_count_for_transitions)
        self.keep_temp_files = bool(keep_temp_files)
        self.emit_master_matrix = bool(emit_master_matrix)
        self.emit_full_outputs = bool(emit_full_outputs)
        self.resume = bool(resume)
        self.finalize_only = bool(finalize_only)
        self.process_only = bool(process_only)
        self.file_filter = file_filter
        self.input_batch_size = int(input_batch_size)
        self.chamber_diameter_px = None if chamber_diameter_px is None else int(chamber_diameter_px)
        self.mm_per_px = None if mm_per_px is None else float(mm_per_px)
        self.video_fps = None if video_fps is None else float(video_fps)
        self.num_shards = int(num_shards)
        self.shard_index = int(shard_index)
        self.shard_by = str(shard_by)
        self.merge_shards = bool(merge_shards)
        self.matrix_first_shards = bool(matrix_first_shards)
        self.finalize_shard_matrix = bool(finalize_shard_matrix)
        self.merge_matrix_parts = bool(merge_matrix_parts)
        self.append_new_data = bool(append_new_data)
        self.plan_only = bool(plan_only)
        self.append_batch_id = append_batch_id

    @property
    def runtime_settings(self) -> MMGenRuntimeSettings:
        """Return explicit runtime settings for this config."""
        return MMGenRuntimeSettings.from_config(self)



def _acquire_mmgen_shard_lifecycle_lock(*, output_dir: str, num_shards: int, shard_index: int, shard_by: str) -> tuple[DirLock, Path]:
    shard_dir = Path(get_shard_output_dir(output_dir, int(num_shards), int(shard_index)))
    work_dir = shard_dir / "_mmgen_work"
    work_dir.mkdir(parents=True, exist_ok=True)
    shard_lock_dir = work_dir / "_mmgen_shard.lock"
    shard_lock = DirLock(shard_lock_dir, timeout_sec=0.0)
    try:
        shard_lock.acquire(
            metadata={
                "kind": "mmgen_shard_matrix_first_lifecycle",
                "num_shards": int(num_shards),
                "shard_index": int(shard_index),
                "shard_by": str(shard_by),
                "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                "slurm_array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID"),
                "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
                "host": socket.gethostname(),
            }
        )
    except Exception as e:
        raise RuntimeError(
            "[mmgen] Shard lifecycle lock already exists. Another shard processing/finalize job may already be running "
            "for the same (num_shards, shard_index).\n"
            f"  shard: {int(shard_index)}/{int(num_shards)-1}\n"
            f"  lock: {shard_lock_dir}\n\n"
            "If you are sure no shard job is running, delete the lock directory and retry. "
            f"(Original error: {e})"
        )
    return shard_lock, shard_lock_dir

def run_mmgen(cfg: MMGenConfig) -> None:
    """Run master matrix generation."""

    if not os.path.isdir(cfg.inference_csv_dir):
        raise FileNotFoundError(f"Inference CSV directory not found: {cfg.inference_csv_dir}")
    if not os.path.isdir(cfg.poscalc_dir):
        raise FileNotFoundError(f"Keypoint-feature directory not found: {cfg.poscalc_dir}")

    os.makedirs(cfg.output_dir, exist_ok=True)

    np.random.seed(int(cfg.random_seed))
    random.seed(int(cfg.random_seed))
    runtime_settings = MMGenRuntimeSettings.from_config(cfg)

    if bool(cfg.append_new_data):
        if not bool(cfg.emit_master_matrix):
            raise ValueError("append_new_data requires master_matrix output to be enabled")
        run_mmgen_append_new_data(cfg)
        return

    if bool(cfg.matrix_first_shards) and int(cfg.num_shards) <= 1:
        raise ValueError("matrix_first_shards requires num_shards > 1")
    if bool(cfg.matrix_first_shards) and str(cfg.shard_by) != "row_identity":
        raise ValueError("matrix_first_shards requires --shard_by row_identity so each final animal stays within one shard")
    if bool(cfg.finalize_shard_matrix) and int(cfg.num_shards) <= 1:
        raise ValueError("finalize_shard_matrix requires num_shards > 1")
    if bool(cfg.finalize_shard_matrix) and str(cfg.shard_by) != "row_identity":
        raise ValueError("finalize_shard_matrix requires --shard_by row_identity so each final animal stays within one shard")

    if bool(cfg.merge_matrix_parts):
        merge_lock_dir = Path(cfg.output_dir) / "_mmgen_work" / "_mmgen_matrix_merge.lock"
        os.makedirs(str(merge_lock_dir.parent), exist_ok=True)
        merge_lock = DirLock(merge_lock_dir, timeout_sec=0.0)
        try:
            merge_lock.acquire(
                metadata={
                    "kind": "mmgen_matrix_merge",
                    "num_shards": int(cfg.num_shards),
                    "host": socket.gethostname(),
                    "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                }
            )
        except Exception as e:
            raise RuntimeError(
                "[mmgen] Matrix-part merge lock already exists. Another matrix merge job may already be running for this output directory.\n"
                f"  lock: {merge_lock_dir}\n\n"
                "If you are sure no matrix merge job is running, delete the lock directory and retry. "
                f"(Original error: {e})"
            )
        try:
            merge_matrix_part_files(
                output_dir=cfg.output_dir,
                num_shards=(int(cfg.num_shards) if int(cfg.num_shards) > 1 else None),
                mm_per_px=(0.0169896744624207 if cfg.mm_per_px is None else float(cfg.mm_per_px)),
                video_fps=(40.0 if cfg.video_fps is None else float(cfg.video_fps)),
            )
        finally:
            try:
                merge_lock.release()
            except Exception:
                pass
        return

    if bool(cfg.finalize_shard_matrix):
        finalize_shard_temp_store(
            inference_csv_dir=cfg.inference_csv_dir,
            poscalc_dir=cfg.poscalc_dir,
            output_dir=cfg.output_dir,
            min_bout=int(cfg.min_bout),
            edge_skip=int(cfg.edge_skip),
            inf_tag=str(cfg.inf_tag),
            runtime_settings=runtime_settings,
            file_filter=cfg.file_filter,
            num_shards=int(cfg.num_shards),
            shard_index=int(cfg.shard_index),
            shard_by=str(cfg.shard_by),
        )
        return

    if bool(cfg.merge_shards):
        merge_lock_dir = Path(cfg.output_dir) / "_mmgen_work" / "_mmgen_merge.lock"
        os.makedirs(str(merge_lock_dir.parent), exist_ok=True)
        merge_lock = DirLock(merge_lock_dir, timeout_sec=0.0)
        try:
            merge_lock.acquire(
                metadata={
                    "kind": "mmgen_merge",
                    "num_shards": int(cfg.num_shards),
                    "host": socket.gethostname(),
                    "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                }
            )
        except Exception as e:
            raise RuntimeError(
                "[mmgen] Merge lock already exists. Another merge/finalize job may already be running for this output directory.\n"
                f"  lock: {merge_lock_dir}\n\n"
                "If you are sure no merge job is running, delete the lock directory and retry. "
                f"(Original error: {e})"
            )

        try:
            merge_shard_temp_stores(output_folder=cfg.output_dir, num_shards=int(cfg.num_shards), strict_missing=True)
            pipeline.run_post_inference_analysis(
                cfg.inference_csv_dir,
                cfg.poscalc_dir,
                cfg.output_dir,
                int(cfg.min_bout),
                int(cfg.edge_skip),
                str(cfg.inf_tag),
                runtime_settings=runtime_settings,
                resume=False,
                finalize_only=True,
                process_only=False,
                file_filter=cfg.file_filter,
                num_shards=1,
                shard_index=0,
                shard_by="hash",
            )
        finally:
            try:
                merge_lock.release()
            except Exception:
                pass
        return

    if int(cfg.num_shards) > 1:
        if bool(cfg.finalize_only):
            raise ValueError(
                "finalize_only cannot be used with num_shards > 1. "
                "Use --finalize_shard_matrix for one shard or --merge_shards / --merge_matrix_parts for the final step."
            )
        if bool(cfg.matrix_first_shards):
            shard_lifecycle_lock, _ = _acquire_mmgen_shard_lifecycle_lock(
                output_dir=cfg.output_dir,
                num_shards=int(cfg.num_shards),
                shard_index=int(cfg.shard_index),
                shard_by=str(cfg.shard_by),
            )
            try:
                pipeline.run_post_inference_analysis(
                    cfg.inference_csv_dir,
                    cfg.poscalc_dir,
                    cfg.output_dir,
                    int(cfg.min_bout),
                    int(cfg.edge_skip),
                    str(cfg.inf_tag),
                    runtime_settings=runtime_settings,
                    resume=bool(cfg.resume),
                    finalize_only=False,
                    process_only=True,
                    file_filter=cfg.file_filter,
                    num_shards=int(cfg.num_shards),
                    shard_index=int(cfg.shard_index),
                    shard_by=str(cfg.shard_by),
                    caller_holds_shard_lock=True,
                )
                finalize_shard_temp_store(
                    inference_csv_dir=cfg.inference_csv_dir,
                    poscalc_dir=cfg.poscalc_dir,
                    output_dir=cfg.output_dir,
                    min_bout=int(cfg.min_bout),
                    edge_skip=int(cfg.edge_skip),
                    inf_tag=str(cfg.inf_tag),
                    runtime_settings=runtime_settings,
                    file_filter=cfg.file_filter,
                    num_shards=int(cfg.num_shards),
                    shard_index=int(cfg.shard_index),
                    shard_by=str(cfg.shard_by),
                    caller_holds_shard_lock=True,
                )
            finally:
                try:
                    shard_lifecycle_lock.release()
                except Exception:
                    pass
            return
        if not bool(cfg.process_only):
            print(
                "[mmgen] num_shards > 1 detected. For safety, forcing process_only=True for this sharded path. "
                "After all shards complete, run with --merge_shards to finalize, or switch to --matrix_first_shards for shard-local matrix parts."
            )
            cfg.process_only = True

    pipeline.run_post_inference_analysis(
        cfg.inference_csv_dir,
        cfg.poscalc_dir,
        cfg.output_dir,
        int(cfg.min_bout),
        int(cfg.edge_skip),
        str(cfg.inf_tag),
        runtime_settings=runtime_settings,
        resume=bool(cfg.resume),
        finalize_only=bool(cfg.finalize_only),
        process_only=bool(cfg.process_only),
        file_filter=cfg.file_filter,
        num_shards=int(cfg.num_shards),
        shard_index=int(cfg.shard_index),
        shard_by=str(cfg.shard_by),
    )
