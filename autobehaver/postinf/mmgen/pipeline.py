"""Pipeline orchestration for mmgen.

This module owns the high-level orchestration for master matrix generation:
- discover and validate inputs
- create centralized animal mapping
- dispatch per-recording workers via multiprocessing
- aggregate intermediate results in a temporary on-disk HDF5 store
- run post-aggregation analyses
- build the final MASTER_FEATURE_MATRIX.csv

The module keeps orchestration responsibilities centralized while using explicit
runtime settings and sharding helpers.

Notes
-----
- Runtime settings are passed in explicitly via :class:`MMGenRuntimeSettings`.
  The orchestration layer no longer depends on mutable cross-module globals.
"""

from __future__ import annotations

import gc
import json
import os
import socket
from datetime import datetime, timezone
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from .hdf_store import (
    HDF_KEYS,
    cleanup_temp_store,
    file_has_any_rows,
    get_done_source_files,
    hdf_safe_append,
    initialize_temp_store,
    open_store,
    purge_file_from_store,
)
from .feature_matrix import MMGenFinalizationSources, create_feature_matrix_from_sources
from .ledger import (
    build_row_identity_summary_df,
    build_source_ledger_df,
    write_batch_manifest_json,
    write_row_identity_summary_csv,
    write_source_ledger_csv,
)
from .worker import process_video_data

# Analysis helpers used during finalization and report generation.
from .kinematics import (
    KIN_FRAMES_SIGNAL_COLS,
    PREDEFINED_KIN_FRAMES_COLS,
    _reconstruct_kinematic_table_from_aggregates,
    calculate_and_save_kinematic_summary,
    calculate_and_save_timeweighted_kinematic_summary,
)
from .transitions import (
    _is_impossible_pose_orient_transition,
    calculate_and_save_transition_analysis,
)
from .occupancy import calculate_and_save_occupancy_analysis
from .measurements import calculate_and_save_pooled_measurements
from .postural import calculate_and_save_postural_summary
from .time_budgets import calculate_and_save_time_budgets
from .mapping import create_animal_mapping
from .naming import collapse_individual_id, parse_group_and_individual

from .batching import make_batches
from .discovery import list_inference_csv_files, print_inference_file_diagnostic
from .sharding import select_files_for_shard
from .locks import DirLock
from .runtime import MMGenRuntimeSettings

def run_post_inference_analysis(
    inference_csv_dir,
    poscalc_dir,
    output_dir,
    min_bout,
    edge_skip,
    inf_tag,
    *,
    runtime_settings: MMGenRuntimeSettings | None = None,
    resume: bool = False,
    finalize_only: bool = False,
    process_only: bool = False,
    file_filter: str | None = None,
    num_shards: int = 1,
    shard_index: int = 0,
    shard_by: str = "hash",
    shards_dirname: str = "_mmgen_shards",
    selected_files: list[str] | None = None,
    drop_all_na_feature_columns: bool = True,
    write_provenance_artifacts: bool = True,
    spatial_in_memory_for_matrix: bool = False,
    caller_holds_shard_lock: bool = False,
):
    """
    Manages the analysis pipeline using a robust, on-disk batch aggregation
    strategy to ensure memory safety and prevent race conditions.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if runtime_settings is None:
        runtime_settings = MMGenRuntimeSettings()

    num_workers = int(runtime_settings.num_workers)
    input_batch_size = int(runtime_settings.input_batch_size)
    emit_master_matrix = bool(runtime_settings.emit_master_matrix)
    emit_full_outputs = bool(runtime_settings.emit_full_outputs)
    make_measurements = bool(runtime_settings.make_measurements)
    calculate_occupancy = bool(runtime_settings.calculate_occupancy)
    calculate_postural_features = bool(runtime_settings.calculate_postural_features)
    make_detailed_transition_outputs = bool(runtime_settings.make_detailed_transition_outputs)
    run_transition_statistics = bool(runtime_settings.run_transition_statistics)
    generate_quiver_plots = bool(runtime_settings.generate_quiver_plots)
    effective_mm_per_px = 0.0169896744624207 if runtime_settings.mm_per_px is None else float(runtime_settings.mm_per_px)
    effective_video_fps = 40.0 if runtime_settings.video_fps is None else float(runtime_settings.video_fps)
    effective_chamber_diameter_px = runtime_settings.chamber_diameter_px

    if selected_files is None:
        inf_files = list_inference_csv_files(inference_csv_dir)
    else:
        # Explicit file selection is primarily used by append/update mode.
        # Preserve caller order while deduplicating.
        inf_files = list(dict.fromkeys([str(f) for f in selected_files]))
    if file_filter is not None and str(file_filter).strip() != "":
        needle = str(file_filter).strip().lower()
        inf_files = [f for f in inf_files if needle in str(f).lower()]

    # --------------------------------------------------------------------
    # Deterministic sharding (optional)
    # --------------------------------------------------------------------
    if int(num_shards) > 1:
        inf_files = select_files_for_shard(
            inf_files,
            num_shards=int(num_shards),
            shard_index=int(shard_index),
            strategy=str(shard_by),
        )
        print(
            f"[mmgen] Sharding enabled: shard {int(shard_index)}/{int(num_shards)-1} "
            f"({len(inf_files)} files selected; strategy={shard_by})."
        )

    # --------------------------------------------------------------------
    # Working output directory
    # --------------------------------------------------------------------
    # For sharded runs, each shard writes its own temp_store.h5 to avoid
    # race conditions and file corruption.
    effective_output_dir = output_dir
    if int(num_shards) > 1:
        effective_output_dir = os.path.join(
            output_dir,
            str(shards_dirname),
            f"shard_{int(shard_index):03d}_of_{int(num_shards):03d}",
        )
        os.makedirs(effective_output_dir, exist_ok=True)

    # mmgen working directory for progress files (keeps output folder tidy)
    work_dir = os.path.join(effective_output_dir, "_mmgen_work")
    os.makedirs(work_dir, exist_ok=True)
    progress_csv_path = os.path.join(work_dir, "mmgen_progress.csv")
    temp_store_path = os.path.join(effective_output_dir, 'temp_store.h5')
    shard_done_path = Path(work_dir) / "SHARD_DONE.json" if int(num_shards) > 1 else None

    if not inf_files:
        if int(num_shards) > 1 and shard_done_path is not None:
            # Strict sharded merge expects every requested shard to emit a
            # completion marker. When a shard receives no files, emit an empty
            # temp store plus a zero-count done marker rather than silently
            # disappearing.
            with open_store(temp_store_path, mode='w'):
                pass
            shard_manifest_path = Path(work_dir) / "SHARD_MANIFEST.json"
            shard_manifest_payload = {
                "created_at": datetime.now(timezone.utc).isoformat(),
                "num_shards": int(num_shards),
                "shard_index": int(shard_index),
                "shard_by": str(shard_by),
                "n_selected_files": 0,
                "selected_files": [],
                "file_filter": (None if file_filter is None else str(file_filter)),
            }
            shard_manifest_path.write_text(json.dumps(shard_manifest_payload, indent=2, sort_keys=True))
            done_payload = {
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "num_shards": int(num_shards),
                "shard_index": int(shard_index),
                "shard_by": str(shard_by),
                "n_inference_files_selected": 0,
                "n_committed_files": 0,
                "temp_store_path": str(Path(temp_store_path).resolve()),
                "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                "host": socket.gethostname(),
            }
            shard_done_path.write_text(json.dumps(done_payload, indent=2, sort_keys=True))
            print("[mmgen] No inference files selected for this shard. Wrote empty shard temp store and completion marker.")
            return

        print("No inference files found. Aborting.")
        return

    # --------------------------------------------------------------------
    # Per-shard safety lock (prevents accidentally running the same shard twice)
    # --------------------------------------------------------------------
    shard_lock: DirLock | None = None
    shard_lock_dir: Path | None = None
    if int(num_shards) > 1 and not bool(caller_holds_shard_lock):
        shard_lock_dir = Path(work_dir) / "_mmgen_shard.lock"

        meta = {
            "kind": "mmgen_shard",
            "num_shards": int(num_shards),
            "shard_index": int(shard_index),
            "shard_by": str(shard_by),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "slurm_array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID"),
            "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
            "host": socket.gethostname(),
        }

        shard_lock = DirLock(shard_lock_dir, timeout_sec=0.0)
        try:
            shard_lock.acquire(metadata=meta)
        except Exception as e:
            raise RuntimeError(
                "[mmgen] Shard lock already exists. This usually means another job is already running "
                "for the same (num_shards, shard_index), or a previous job crashed and left a stale lock.\n"
                f"  shard: {int(shard_index)}/{int(num_shards)-1}\n"
                f"  lock: {shard_lock_dir}\n\n"
                "If you are sure no shard job is running, delete the lock directory and retry. "
                f"(Original error: {e})"
            )

    # Write a per-shard manifest (helpful when debugging/continuing failed runs)
    if int(num_shards) > 1:
        try:
            manifest_path = os.path.join(work_dir, "shard_manifest.csv")
            pd.DataFrame({"inference_csv": sorted(map(str, inf_files))}).to_csv(manifest_path, index=False)
            shard_manifest_path = Path(work_dir) / "SHARD_MANIFEST.json"
            shard_manifest_payload = {
                "created_at": datetime.now(timezone.utc).isoformat(),
                "num_shards": int(num_shards),
                "shard_index": int(shard_index),
                "shard_by": str(shard_by),
                "n_selected_files": int(len(inf_files)),
                "selected_files": sorted(map(str, inf_files)),
                "file_filter": (None if file_filter is None else str(file_filter)),
                "effective_output_dir": str(Path(effective_output_dir).resolve()),
            }
            shard_manifest_path.write_text(json.dumps(shard_manifest_payload, indent=2, sort_keys=True))
        except Exception:
            pass

    # ========================================================================
    # DIAGNOSTIC: Print all input files and their parsed labels
    # ========================================================================
    print_inference_file_diagnostic(inf_files)

    # ========================================================================
    # CREATE CENTRALIZED ANIMAL MAPPING
    # ========================================================================
    # Mapping is only required for *finalization* (collapsing per-movie IDs to
    # per-animal IDs). In sharded process-only runs, skip mapping creation to
    # avoid multiple shards racing to write ANIMAL_MAPPING.* into the same
    # output folder.
    animal_mapping = {}
    file_to_animal = {}
    if not (process_only and int(num_shards) > 1):
        print("\n" + "="*80)
        print("CREATING CENTRALIZED ANIMAL MAPPING")
        print("="*80)
        animal_mapping, file_to_animal = create_animal_mapping(
            inference_csv_dir,
            poscalc_dir,
            output_dir,
        )

        if not animal_mapping:
            print("ERROR: Failed to create animal mapping. Aborting.")
            return

        print(f"Successfully created mapping for {len(animal_mapping)} animals.")
        print("="*80 + "\n")
    else:
        print(
            "[mmgen] Sharded process-only run: skipping animal mapping creation "
            "(mapping will be created during merged finalization)."
        )

    # CREATE CONFIG AND INITIALIZE TEMP FILES
    config = {
        'make_measurements': runtime_settings.make_measurements,
        'calculate_occupancy': runtime_settings.calculate_occupancy,
        'calculate_postural_features': runtime_settings.calculate_postural_features,
        'animal_mapping': animal_mapping,
        'runtime_settings': runtime_settings,
        'file_to_animal': file_to_animal,
    }

    # Single HDF5 store for all temporary tables

    # Resume requires keeping the temp store around.
    effective_keep_temp_files = bool(runtime_settings.keep_temp_files or resume or finalize_only or process_only)
    if (resume or finalize_only) and not runtime_settings.keep_temp_files:
        print(
            "[mmgen] Warning: keep_temp_files is False, but resume/finalize_only requires the temp store. "
            "Forcing keep_temp_files=True for this run."
        )

    # Determine which files are already committed in the temp store.
    store_exists = os.path.exists(temp_store_path)
    done_files: set[str] = set()
    pending_files = list(inf_files)

    if resume or finalize_only or process_only:
        if store_exists:
            done_files = get_done_source_files(temp_store_path)

        if finalize_only and not store_exists:
            raise FileNotFoundError(
                f"finalize_only was requested but no temp store exists at: {temp_store_path}"
            )

        if resume and store_exists:
            # If the store exists but has no commit markers, we cannot safely resume.
            # (Would risk double-counting.)
            with open_store(temp_store_path, mode='r') as s:
                has_commit_key = HDF_KEYS['video_lengths_tmp'] in s
            if not has_commit_key:
                raise RuntimeError(
                    "Resume requested, but existing temp_store.h5 does not contain the "
                    "video_lengths_tmp commit table required for safe resume. "
                    "Delete temp_store.h5 and re-run from scratch, or run without --resume."
                )

        pending_files = [f for f in inf_files if f not in done_files]

        if resume:
            print(
                f"[mmgen] Resume enabled: {len(done_files)} done, {len(pending_files)} pending (of {len(inf_files)} total)."
            )

        if process_only and store_exists and not resume:
            print(
                "[mmgen] Warning: process_only was requested without resume, and temp_store.h5 already exists. "
                "This may double-count files if re-run. Consider adding --resume."
            )

        if finalize_only:
            print(
                "[mmgen] finalize_only enabled: skipping per-file processing and running analysis from existing temp store."
            )
            if pending_files:
                raise RuntimeError(
                    "finalize_only was requested, but temp_store.h5 does not contain all selected inference files. "
                    f"Missing commits for {len(pending_files)} file(s). "
                    "Re-run processing (or use --resume) before finalizing."
                )
            pending_files = []
    else:
        # Not resuming: ignore any existing temp store contents. We will
        # re-initialize (overwrite) the temp store and process all files.
        done_files = set()
        pending_files = list(inf_files)


    # Initialize the temp store only for fresh runs.
    if not store_exists or (store_exists and not resume and not finalize_only and not process_only):
        print("Initializing temporary HDF5 tables...")
        # Pass the config to the initialization run as well
        first_basename = pending_files[0] if pending_files else inf_files[0]
        first_file_args = (
            os.path.join(inference_csv_dir, first_basename),
            poscalc_dir,
            min_bout,
            edge_skip,
            inf_tag,
            config,
        )
        (bout_dict, _, _, pos_df, post_df, _, _, kin_df, _) = process_video_data(first_file_args)

        initialize_temp_store(
            temp_store_path,
            bout_dict=bout_dict,
            pos_df=pos_df,
            post_df=post_df,
            kin_df=kin_df,
            predefined_kin_cols=PREDEFINED_KIN_FRAMES_COLS,
        )
        store_exists = True
    else:
        if resume:
            print("[mmgen] Using existing temp_store.h5 (resume mode).")
        elif finalize_only:
            print("[mmgen] Using existing temp_store.h5 (finalize_only mode).")
        elif process_only:
            print("[mmgen] Using existing temp_store.h5 (process_only mode).")

    # SET UP BATCH PROCESSING
    batch_size = int(runtime_settings.input_batch_size)
    batches = make_batches(pending_files, batch_size)
    num_batches = len(batches)
    print(
        f"Total files to process: {len(pending_files)}. Processing in {num_batches} batches of up to {batch_size} files each."
    )

    # Per-recording bout/sleep summaries are appended to the temp HDF store
    # to avoid unbounded in-memory accumulation on large datasets.

    # --------------------------------------------------------------------
    # Helper: append one progress row to a CSV (best-effort).
    # --------------------------------------------------------------------
    def _append_progress_row(row: dict) -> None:
        try:
            import csv
            from datetime import datetime, timezone

            os.makedirs(os.path.dirname(progress_csv_path), exist_ok=True)
            file_exists = os.path.exists(progress_csv_path)

            out = dict(row)
            out["processed_at_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")

            with open(progress_csv_path, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(out.keys()))
                if not file_exists:
                    writer.writeheader()
                writer.writerow(out)
        except Exception:
            pass

    # PROCESS BATCHES AND APPEND TO DISK
    worker_failures: list[dict[str, str]] = []
    for i, batch_files in enumerate(batches):
        
        print(f"\n--- Processing Batch {i + 1}/{num_batches} ---")
        
        # Append the config dictionary to the arguments for each worker
        args_list = [
            (os.path.join(inference_csv_dir, f), poscalc_dir, min_bout, edge_skip, inf_tag, config)
            for f in batch_files
        ]

        # -----------------------------------------------------------------
        # Resume safety: before re-processing a batch, purge any *partial*
        # rows for those files (i.e., file has rows in the store but is not
        # committed in video_lengths_tmp).
        #
        # This must happen *before* we open the HDFStore for append, to avoid
        # nested HDFStore opens on the same file.
        # -----------------------------------------------------------------
        if resume and os.path.exists(temp_store_path):
            for f in batch_files:
                if f in done_files:
                    continue
                try:
                    _, ind_id, _ = parse_group_and_individual(str(f))
                    if file_has_any_rows(temp_store_path, source_file=str(f), individual_id=ind_id):
                        purge_file_from_store(temp_store_path, source_file=str(f), individual_id=ind_id)
                except Exception as e:
                    print(f"[mmgen-resume] Warning: pre-batch purge check failed for {f}: {e}")
        
        # Stream results one-by-one to avoid holding large per-file DataFrames in memory
        with Pool(processes=int(runtime_settings.num_workers)) as pool:
            with open_store(temp_store_path, mode='a') as store:
                for j, (
                    bout_features_dict,
                    measurement_result,
                    video_length,
                    positional_df,
                    postural_features_df,
                    bout_summary_df,
                    sleep_summary_df,
                    kin_frames_df,
                    failure_info,
                ) in enumerate(
                    tqdm(pool.imap(process_video_data, args_list), total=len(args_list), desc="Analyzing batch")
                ):
                    # Append per-recording bout/sleep summary tables to disk (avoid RAM accumulation)
                    if bout_summary_df is not None and not bout_summary_df.empty:
                        hdf_safe_append(store, HDF_KEYS['bout_summary_tmp'], bout_summary_df)
                    if sleep_summary_df is not None and not sleep_summary_df.empty:
                        hdf_safe_append(store, HDF_KEYS['sleep_summary_tmp'], sleep_summary_df)
                    
                    # Always track denominator (total frames) per movie, regardless of bout extraction.
                    # Use the filename to derive the per-movie individual ID.
                    this_file = args_list[j][0]
                    filename = os.path.basename(this_file)
                    group_name, individual_id, sex = parse_group_and_individual(filename)

                    if failure_info is not None:
                        failure_row = dict(failure_info)
                        failure_row.setdefault('source_file', filename)
                        worker_failures.append({k: ('' if v is None else str(v)) for k, v in failure_row.items()})

                    if bout_features_dict:
                        for suffix, df_bout in bout_features_dict.items():
                            if df_bout is not None and not df_bout.empty:
                                key = HDF_KEYS[f"bouts{suffix}"]
                                hdf_safe_append(store, key, df_bout)

                                # Build per-file aggregates for this smoothing level
                                base_keys = ['individual','group','sex','category','bout_type']
                                g = df_bout.groupby(base_keys, observed=True)

                                # Bout length aggregates
                                agg_len = g['length'].agg(['count','sum']).rename(columns={'count':'n_bouts','sum':'sum_length'})
                                sum_length_sq = g['length'].apply(lambda x: np.square(x.astype('float64')).sum())
                                agg_len['sum_length_sq'] = sum_length_sq

                                # Kinematic scalar columns: ends with _mean, _std, _asym, _min, _max, _median, or _skewness
                                kin_cols = [c for c in df_bout.columns if c.endswith(('_mean', '_std', '_asym', '_min', '_max', '_median', '_skewness'))]
                                if kin_cols:
                                    kin_sum   = g[kin_cols].sum()
                                    kin_sumsq = g[kin_cols].apply(lambda df: np.square(df.astype('float64')).sum())
                                    kin_cnt   = g[kin_cols].count()
                                    agg_out = agg_len.join([
                                        kin_sum.add_suffix('__sum'),
                                        kin_sumsq.add_suffix('__sum_sq'),
                                        kin_cnt.add_suffix('__count'),
                                    ])
                                else:
                                    agg_out = agg_len

                                agg_out = agg_out.reset_index()
                                # Carry source_file to support safe resume/purging.
                                agg_out['source_file'] = filename
                                agg_key = HDF_KEYS[f'agg{suffix}_tmp']
                                hdf_safe_append(store, agg_key, agg_out)

                    # Per-file transition counts (category-specific), only instantaneous level
                    inst_df = bout_features_dict.get('_inst') if bout_features_dict else None
                    if inst_df is not None and not inst_df.empty:
                        for (src, cat), sub in inst_df.groupby(['source_file','category'], observed=True):
                            sub = sub.sort_values('start')
                            if len(sub) < 2:
                                continue

                            # Transition sequence on the native bout labels for this category
                            seq = sub['bout_type'].astype(str).to_numpy()
                            frm = seq[:-1]
                            to  = seq[ 1:]
                            t = pd.crosstab(pd.Series(frm, name='from_behavior'),
                                            pd.Series(to,  name='to_behavior'))
                            if t.size == 0:
                                # Still allow pose-collapsed transitions below for Pose+Orient
                                t = None
                                tidy = None
                            else:
                                tidy = t.stack().reset_index(name='count')

                                # Filter out self-transitions (where from_behavior == to_behavior)
                                tidy = tidy[tidy['from_behavior'] != tidy['to_behavior']]

                                # Filter out physically impossible pose+orient transitions
                                if cat == 'Pose+Orient':
                                    impossible_mask = tidy.apply(
                                        lambda row: _is_impossible_pose_orient_transition(
                                            row['from_behavior'], row['to_behavior']
                                        ), axis=1
                                    )
                                    n_impossible = impossible_mask.sum()
                                    if n_impossible > 0:
                                        tidy = tidy[~impossible_mask]

                            # Append category-native transitions if any remain
                            if tidy is not None and not tidy.empty and tidy['count'].sum() > 0:
                                tidy['source_file'] = src
                                tidy['individual'] = sub['individual'].iloc[0]
                                tidy['group']      = sub['group'].iloc[0]
                                tidy['category']   = cat
                                hdf_safe_append(store, HDF_KEYS['transitions_tmp'], tidy)

                            # ------------------------------------------------------------
                            # ALSO produce POSE-ONLY transitions (collapsed from Pose+Orient)
                            #
                            # After removing pose-only kinematics, we no longer have a
                            # category='Pose' kinematics table to derive pose transitions from.
                            # However, pose-only transitions are often the most interpretable.
                            #
                            # We recover them here by collapsing "Pose + Orient" bout labels
                            # down to pose only (e.g. "Walking + Floor" -> "Walking") and
                            # computing transitions on that collapsed sequence.
                            # ------------------------------------------------------------
                            if cat == 'Pose+Orient':
                                pose_seq = (
                                    sub['bout_type']
                                    .astype(str)
                                    .str.split(r'\s\+\s', n=1)
                                    .str[0]
                                    .to_numpy()
                                )
                                if len(pose_seq) >= 2:
                                    frm_p = pose_seq[:-1]
                                    to_p  = pose_seq[1:]
                                    t_p = pd.crosstab(pd.Series(frm_p, name='from_behavior'),
                                                      pd.Series(to_p,  name='to_behavior'))
                                    if t_p.size > 0:
                                        tidy_p = t_p.stack().reset_index(name='count')
                                        tidy_p = tidy_p[tidy_p['from_behavior'] != tidy_p['to_behavior']]
                                        if not tidy_p.empty and tidy_p['count'].sum() > 0:
                                            tidy_p['source_file'] = src
                                            tidy_p['individual'] = sub['individual'].iloc[0]
                                            tidy_p['group']      = sub['group'].iloc[0]
                                            tidy_p['category']   = 'Pose'
                                            hdf_safe_append(store, HDF_KEYS['transitions_tmp'], tidy_p)

                    # Measurements: append compact per-recording summary rows to disk
                    # (prevents long-run memory accumulation from raw per-frame tables).
                    if runtime_settings.make_measurements and measurement_result is not None:
                        try:
                            meas_df = pd.DataFrame([measurement_result])
                            # Carry source_file for purge/resume safety.
                            meas_df['source_file'] = filename
                            hdf_safe_append(store, HDF_KEYS['measurements_tmp'], meas_df)
                        except Exception as e:
                            print(f"Warning: failed to append measurement summary for {filename}: {e}")

                    if positional_df is not None and not positional_df.empty:
                        hdf_safe_append(store, HDF_KEYS['positional'], positional_df)

                    if postural_features_df is not None and not postural_features_df.empty:
                        hdf_safe_append(store, HDF_KEYS['postural'], postural_features_df)

                    if kin_frames_df is not None and not kin_frames_df.empty:
                        hdf_safe_append(store, HDF_KEYS['kin_frames'], kin_frames_df)

                    # ----------------------------------------------------------------
                    # Commit marker for resume: one row per processed inference CSV.
                    # Write this *last* so its presence implies the file was fully
                    # appended across all tables.
                    # ----------------------------------------------------------------
                    if isinstance(video_length, (int, np.integer)) and int(video_length) > 0:
                        try:
                            vid_df = pd.DataFrame(
                                [
                                    {
                                        'source_file': filename,
                                        'group': group_name,
                                        'individual': individual_id,
                                        'sex': sex,
                                        'n_frames': int(video_length),
                                    }
                                ]
                            )
                            hdf_safe_append(store, HDF_KEYS['video_lengths_tmp'], vid_df)
                            _append_progress_row(vid_df.iloc[0].to_dict())
                            # Update in-memory done set so later batches don't attempt
                            # to purge/re-handle this file.
                            done_files.add(filename)
                        except Exception as e:
                            print(f"[mmgen] Warning: failed to append video length commit row for {filename}: {e}")

                    # Free per-file objects aggressively (kin_frames can be large)
                    try:
                        del bout_features_dict, measurement_result, positional_df, postural_features_df, bout_summary_df, sleep_summary_df, kin_frames_df, failure_info, inst_df
                    except Exception:
                        pass

                    # Periodic GC helps long overnight runs
                    if (j + 1) % 10 == 0:
                        gc.collect()


    # Safety: every selected source file must be fully committed before we
    # either declare a shard complete or run finalization.
    selected_file_set = set(map(str, inf_files))
    committed_file_set = set(map(str, done_files))
    missing_commits = sorted(selected_file_set - committed_file_set)
    if missing_commits:
        failed_csv_path = os.path.join(work_dir, "FAILED_SOURCE_FILES.csv")
        try:
            failures_df = pd.DataFrame({"source_file": missing_commits})
            if worker_failures:
                failure_details_df = pd.DataFrame(worker_failures)
                if not failure_details_df.empty:
                    failure_details_df = (
                        failure_details_df.drop_duplicates(subset=["source_file"], keep="last")
                        [[c for c in ["source_file", "error_stage", "error_type", "error_message", "traceback"] if c in failure_details_df.columns]]
                    )
                    failures_df = failures_df.merge(failure_details_df, on="source_file", how="left")
            failures_df.to_csv(failed_csv_path, index=False)
        except Exception:
            pass
        raise RuntimeError(
            "mmgen did not fully commit all selected inference files before completion. "
            f"Missing {len(missing_commits)} file(s). "
            f"See {failed_csv_path} for the recorded list. "
            "This usually means one or more worker files failed during processing."
        )

    # If this was a process-only run, stop here (leave temp store intact).
    if process_only:
        # For sharded runs, write a completion marker and release the shard lock.
        if int(num_shards) > 1 and shard_done_path is not None:
            try:
                done_payload = {
                    "completed_at": datetime.now(timezone.utc).isoformat(),
                    "num_shards": int(num_shards),
                    "shard_index": int(shard_index),
                    "shard_by": str(shard_by),
                    "n_inference_files_selected": int(len(inf_files)),
                    "n_committed_files": int(len(done_files)),
                    "temp_store_path": str(Path(temp_store_path).resolve()),
                    "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                    "host": socket.gethostname(),
                }
                shard_done_path.write_text(json.dumps(done_payload, indent=2, sort_keys=True))
            except Exception:
                pass

        # Release lock only on successful completion.
        if shard_lock is not None:
            try:
                shard_lock.release()
            except Exception:
                pass

        print("[mmgen] process_only enabled: skipping final analysis + feature matrix generation.")
        return

    # LOAD FINAL AGGREGATED DATA FROM DISK AND RUN ANALYSIS
    print("\n--- All batches processed. Loading aggregated data for final analysis. ---")
    try:
        # Direct finalization sources populated during analysis steps
        animal_summary = None
        animal_sleep_summary = None
        time_budget_sheets = None
        measurements_df = None
        postural_summary_df = None
        kinematic_sheets = None
        tw_kinematic_sheets = None
        transition_pose_tidy_df = None
        transition_pose_orient_tidy_df = None
        spatial_data_path = None
        spatial_df = None

        # Read aggregates instead of full tables
        agg_tables = {}
        for suffix in ['_inst']:
            key = HDF_KEYS[f'agg{suffix}_tmp']
            try:
                agg_tables[suffix] = pd.read_hdf(temp_store_path, key=key)
            except (FileNotFoundError, KeyError):
                agg_tables[suffix] = pd.DataFrame()

        # Collapse movies → individual animals using centralized mapping
        # Use mapping if available, otherwise fall back to collapse function
        file_to_animal_map = config.get('file_to_animal', {})
        animal_mapping_dict = config.get('animal_mapping', {})
        individual_to_animal_map = {}
        for animal_id, animal_info in animal_mapping_dict.items():
            for individual_id in animal_info.get('individual_ids', []):
                if individual_id is not None:
                    individual_to_animal_map[str(individual_id)] = str(animal_id)
        
        def _collapse_individual_with_mapping(id_):
            """Collapse individual ID using mapping if available, otherwise use collapse function."""
            id_ = str(id_)
            # Try to find in mapping first
            if id_ in file_to_animal_map:
                return file_to_animal_map[id_]
            # Try to find by matching raw individual_id in animal_mapping
            if id_ in individual_to_animal_map:
                return individual_to_animal_map[id_]
            # Fallback to collapse function
            return collapse_individual_id(id_)

        for suffix, df in agg_tables.items():
            if not df.empty:
                agg_tables[suffix]['individual'] = agg_tables[suffix]['individual'].map(_collapse_individual_with_mapping)

        # Build animal_total_frames from per-file commit table (supports resume)
        animal_total_frames: dict[str, int] = {}
        try:
            lengths_df = pd.read_hdf(temp_store_path, key=HDF_KEYS['video_lengths_tmp'])
        except (FileNotFoundError, KeyError):
            lengths_df = pd.DataFrame()

        if lengths_df is not None and not lengths_df.empty:
            # Guard against duplicates (should not happen, but resume safety).
            if 'source_file' in lengths_df.columns:
                lengths_df = lengths_df.drop_duplicates(subset=['source_file'], keep='last')
            if 'individual' in lengths_df.columns and 'n_frames' in lengths_df.columns:
                for _, row in lengths_df.iterrows():
                    ind_id = row.get('individual')
                    n_frames = row.get('n_frames')
                    if ind_id is None or pd.isna(ind_id):
                        continue
                    try:
                        n_frames_i = int(n_frames)
                    except Exception:
                        continue
                    animal_id = _collapse_individual_with_mapping(str(ind_id))
                    animal_total_frames[animal_id] = animal_total_frames.get(animal_id, 0) + n_frames_i

            # Write a user-friendly index for debugging/progress tracking.
            try:
                lengths_out = os.path.join(work_dir, 'mmgen_file_index.csv')
                lengths_df.to_csv(lengths_out, index=False)
            except Exception:
                pass

        # Load per-recording bout/sleep summaries from disk (written during batch processing)
        try:
            master_bout_summary_df = pd.read_hdf(temp_store_path, key=HDF_KEYS['bout_summary_tmp'])
        except (FileNotFoundError, KeyError):
            master_bout_summary_df = pd.DataFrame()

        try:
            master_sleep_df = pd.read_hdf(temp_store_path, key=HDF_KEYS['sleep_summary_tmp'])
        except (FileNotFoundError, KeyError):
            master_sleep_df = pd.DataFrame()

        # Build 1_Individual_Bout_Summary.csv from the comprehensive features
        if master_bout_summary_df is not None and not master_bout_summary_df.empty:
            # Collapse movie-level individual IDs to animal IDs
            master_bout_summary_df['animal_id'] = master_bout_summary_df['individual'].map(_collapse_individual_with_mapping)

            # Define aggregation: sum for count columns, mean for everything else
            count_cols = ['n_bouts', 'total_length_frames']
            numeric_cols = master_bout_summary_df.select_dtypes(include=[np.number]).columns.tolist()

            agg_dict = {}
            for col in numeric_cols:
                if col in count_cols:
                    agg_dict[col] = 'sum'
                else:
                    agg_dict[col] = 'mean'

            animal_summary = (
                master_bout_summary_df
                .groupby(['animal_id', 'group', 'sex', 'category', 'bout_type'], observed=True)
                .agg(agg_dict)
                .reset_index()
            )

            animal_summary.rename(columns={'animal_id': 'individual'}, inplace=True)

            if emit_full_outputs:
                output_path = os.path.join(output_dir, '1_Individual_Bout_Summary.csv')
                animal_summary.to_csv(output_path, index=False)
                print(f"Saved new comprehensive bout summary to {output_path}")
        else:
            print("No bout summary data was generated to create 1_Individual_Bout_Summary.csv.")

        # Build 12_Sleep_Features.csv from sleep summaries
        if master_sleep_df is not None and not master_sleep_df.empty:
            # Collapse movie-level individual IDs to animal IDs
            master_sleep_df['animal_id'] = master_sleep_df['individual'].map(_collapse_individual_with_mapping)

            # Define aggregation: sum for count columns, mean for everything else
            # sleep_total_length_frames is needed for sleep_time_budget calculation but will be dropped before saving
            sleep_count_cols = ['n_sleep_bouts', 'sleep_total_length_frames']
            sleep_numeric_cols = master_sleep_df.select_dtypes(include=[np.number]).columns.tolist()

            sleep_agg_dict = {}
            for col in sleep_numeric_cols:
                if col in sleep_count_cols:
                    sleep_agg_dict[col] = 'sum'
                else:
                    sleep_agg_dict[col] = 'mean'

            animal_sleep_summary = (
                master_sleep_df
                .groupby(['animal_id', 'group', 'sex', 'bout_type'], observed=True)
                .agg(sleep_agg_dict)
                .reset_index()
            )

            # Recalculate sleep_time_budget after aggregation
            if 'sleep_total_length_frames' in animal_sleep_summary.columns:
                animal_sleep_summary['total_frames'] = animal_sleep_summary['animal_id'].map(animal_total_frames)
                animal_sleep_summary['sleep_time_budget'] = (
                    animal_sleep_summary['sleep_total_length_frames'] /
                    animal_sleep_summary['total_frames']
                ).fillna(0.0)
                animal_sleep_summary = animal_sleep_summary.drop(columns=['total_frames'])

            # Drop non-normalized columns before saving (keep only normalized sleep_time_budget)
            if 'sleep_total_length_frames' in animal_sleep_summary.columns:
                animal_sleep_summary = animal_sleep_summary.drop(columns=['sleep_total_length_frames'])
            if 'total_sleep_time_s' in animal_sleep_summary.columns:
                animal_sleep_summary = animal_sleep_summary.drop(columns=['total_sleep_time_s'])

            animal_sleep_summary.rename(columns={'animal_id': 'individual'}, inplace=True)

            if emit_full_outputs:
                sleep_output_path = os.path.join(output_dir, '12_Sleep_Features.csv')
                animal_sleep_summary.to_csv(sleep_output_path, index=False)
                print(f"Saved sleep features to {sleep_output_path}")
        else:
            print("No sleep summary data was generated to create 12_Sleep_Features.csv.")

        # The rest of the analysis can proceed, now in numerical order of outputs.

        # 2. Time Budgets
        if master_bout_summary_df is not None and not master_bout_summary_df.empty:
            time_budget_sheets = calculate_and_save_time_budgets(
                master_bout_summary_df,
                animal_total_frames,
                output_dir,
                save_outputs=emit_full_outputs,
            )

        # 3. Standardized Measurements
        if runtime_settings.make_measurements:
            try:
                meas_input_df = pd.read_hdf(temp_store_path, key=HDF_KEYS['measurements_tmp'])
            except (FileNotFoundError, KeyError):
                meas_input_df = pd.DataFrame()

            if meas_input_df is not None and not meas_input_df.empty:
                measurements_df = calculate_and_save_pooled_measurements(
                    meas_input_df,
                    output_dir,
                    mm_per_px=effective_mm_per_px,
                    random_seed=int(runtime_settings.random_seed),
                    save_outputs=emit_full_outputs,
                )
            else:
                print("No measurement summaries were generated (missing orientation or no valid frames).")
      
        # 4. Postural Features
        if runtime_settings.calculate_postural_features:
            try:
                master_postural_df = pd.read_hdf(temp_store_path, key=HDF_KEYS['postural'])
                if master_postural_df is not None and not master_postural_df.empty:
                    
                    # Normalization logic
                    if measurements_df is not None and 'length_avg_mean' in measurements_df.columns:
                        print("Applying stable normalizer to postural features...")
                        # Create a mapping from animal_id to its stable body length
                        # Use mapping to collapse individual IDs to animal IDs
                        file_to_animal_map = config.get('file_to_animal', {})
                        animal_mapping_dict = config.get('animal_mapping', {})
                        
                        def _collapse_for_measurements(ind_id):
                            """Collapse individual_id using mapping if available."""
                            # Try to find in mapping first
                            if ind_id in file_to_animal_map:
                                return file_to_animal_map[ind_id]
                            # Try to find by matching individual_id in animal_mapping
                            for animal_id, animal_info in animal_mapping_dict.items():
                                if ind_id in animal_info['individual_ids']:
                                    return animal_id
                            # Fallback to collapse function
                            return collapse_individual_id(ind_id)
                        
                        measurements_df['animal_id'] = measurements_df['individual_id'].map(_collapse_for_measurements)

                        # Aggregate the measurements by animal_id, taking the mean. This ensures one row per animal.
                        animal_measurements = measurements_df.groupby('animal_id').mean(numeric_only=True)

                        # Now, create the normalizer map from the aggregated data, which has a unique index.
                        normalizer_map = animal_measurements['length_avg_mean']
                        
                        # Map the normalizer to the postural dataframe
                        # Use the same collapse function for consistency
                        master_postural_df['animal_id'] = master_postural_df['individual'].map(_collapse_for_measurements)
                        master_postural_df['normalizer'] = master_postural_df['animal_id'].map(normalizer_map)
                        
                        # *** ASSERT CHECK: Ensure thorax width normalizer is positive ***
                        if master_postural_df['normalizer'].isnull().any():
                            missing_norm_animals = master_postural_df[master_postural_df['normalizer'].isnull()]['animal_id'].unique()
                            print(f"Warning: Normalizer (length_avg_mean) was missing for {len(missing_norm_animals)} animals in postural data. Normalized extensions will be NaN for them.")
                            assert not master_postural_df['normalizer'].isnull().all(), "Normalizer mapping failed for ALL animals!"
                        else:
                            print(" - Assert Check Passed: Normalizer mapped successfully for all animals in postural data.")
                        # Ensure normalizer is positive where it exists
                        valid_normalizers = master_postural_df['normalizer'].dropna()
                        if not valid_normalizers.empty:
                            assert (valid_normalizers > 0).all(), "Mapped normalizer values must be positive."
                        print(" - Assert Check Passed: Mapped normalizer values are positive.")

                        # Normalize all raw extension columns
                        for col in master_postural_df.columns:
                            if col.startswith('extension_raw_'):
                                new_col_name = col.replace('_raw_', '_norm_')
                                master_postural_df[new_col_name] = master_postural_df[col] / master_postural_df['normalizer']

                    postural_summary_df = calculate_and_save_postural_summary(
                        master_postural_df,
                        output_dir,
                        save_outputs=emit_full_outputs,
                    )

                del master_postural_df
            except (FileNotFoundError, KeyError):
                print("Could not load postural data; skipping postural analysis.")
                pass

        # Analyses based on HDF5 aggregates
        inst_agg = agg_tables.get('_inst', pd.DataFrame())
        if not inst_agg.empty:
            # 5. Kinematic Features (BB) - aggregated per-bout stats
            kin_tables = _reconstruct_kinematic_table_from_aggregates(
                agg_tables,
                min_bout_count_for_features=int(runtime_settings.min_bout_count_for_features),
            )
            kinematic_sheets = calculate_and_save_kinematic_summary(
                kin_tables,
                output_dir,
                min_bout_count_for_features=int(runtime_settings.min_bout_count_for_features),
                save_outputs=emit_full_outputs,
            )

        # 6, 7, 8. Detailed Transition Analysis
            if runtime_settings.make_detailed_transition_outputs:
                try:
                    # Load the pre-aggregated transition counts, which is memory-safe
                    key_transitions_tmp = HDF_KEYS['transitions_tmp']
                    aggregated_transitions = pd.read_hdf(temp_store_path, key=key_transitions_tmp)
                    
                    if aggregated_transitions is not None and not aggregated_transitions.empty:
                        # Pass the aggregated table, not the raw bouts
                        transition_results = calculate_and_save_transition_analysis(
                            aggregated_transitions,
                            output_dir,
                            run_transition_statistics=run_transition_statistics,
                            save_outputs=emit_full_outputs,
                        )
                        transition_pose_tidy_df = transition_results.get('pose_tidy_df')
                        transition_pose_orient_tidy_df = transition_results.get('pose_orient_tidy_df')
                    else:
                        print("No aggregated transition data found; skipping detailed transition outputs (6/7/8).")
                except (FileNotFoundError, KeyError) as e:
                    print(f"Could not load aggregated transitions for detailed analysis: {e}")
            else:
                print("\nDetailed transition outputs disabled. Skipping transition analyses.")
        else:
            print("\nNo bout features were extracted. Skipping BB Kinematic and Transition analyses.")

        # 5b. KinematicTW Features (TW) - aggregated per-frame stats from /kin_frames
        # This runs independently of inst_agg because TW depends on /kin_frames,
        # not on the per-bout aggregate tables.
        tw_kinematic_sheets = calculate_and_save_timeweighted_kinematic_summary(
            temp_store_path,
            output_dir,
            video_fps=effective_video_fps,
            save_outputs=emit_full_outputs,
            num_workers=int(runtime_settings.num_workers),
            individual_to_animal=individual_to_animal_map,
        )

        # 9, 10, 11. Spatial Occupancy and Orientation Analysis
        if runtime_settings.calculate_occupancy:
            try:
                positional_key = HDF_KEYS['positional']
                key_exists = False
                with open_store(temp_store_path, mode='r') as store:
                    if positional_key in store:
                        key_exists = True
                
                if key_exists:
                    transformed_output_path = None
                    return_transformed_df = bool((not emit_full_outputs) and spatial_in_memory_for_matrix)
                    if not emit_full_outputs and not return_transformed_df:
                        transformed_output_path = os.path.join(work_dir, 'temp_transformed_positional_data.csv')
                    spatial_result = calculate_and_save_occupancy_analysis(
                        temp_store_path,
                        positional_key,
                        output_dir,
                        chamber_diameter_px=effective_chamber_diameter_px,
                        generate_quiver_plots=generate_quiver_plots,
                        save_outputs=emit_full_outputs,
                        transformed_output_path=transformed_output_path,
                        num_workers=int(runtime_settings.num_workers),
                        return_transformed_df=return_transformed_df,
                    )
                    if return_transformed_df:
                        if isinstance(spatial_result, pd.DataFrame):
                            spatial_df = spatial_result
                        else:
                            spatial_df = None
                    else:
                        spatial_data_path = spatial_result
                else:
                    print("Could not find positional data key in HDF5 store; skipping occupancy analysis.")
                
            except (FileNotFoundError, KeyError) as e:
                print(f"Could not load positional data; skipping occupancy analysis. Error: {e}")
                pass
        
        # --- CREATE MASTER FEATURE MATRIX ---
        master_matrix_df = None
        if emit_master_matrix:
            print("\n--- Generating Master Feature Matrix ---")
            sources = MMGenFinalizationSources(
                bout_summary_df=animal_summary,
                time_budget_sheets=time_budget_sheets,
                measurements_df=measurements_df,
                postural_summary_df=postural_summary_df,
                kinematic_sheets=kinematic_sheets,
                tw_kinematic_sheets=tw_kinematic_sheets,
                transition_pose_tidy_df=transition_pose_tidy_df,
                transition_pose_orient_tidy_df=transition_pose_orient_tidy_df,
                sleep_summary_df=animal_sleep_summary,
                spatial_df=spatial_df,
                spatial_data_path=spatial_data_path,
            )
            master_matrix_df = create_feature_matrix_from_sources(
                output_dir,
                sources,
                filter_features_by_bout_count=bool(runtime_settings.filter_features_by_bout_count),
                min_bout_count_for_features=int(runtime_settings.min_bout_count_for_features),
                min_bout_count_for_transitions=int(runtime_settings.min_bout_count_for_transitions),
                mm_per_px=effective_mm_per_px,
                video_fps=effective_video_fps,
                drop_all_na_columns=bool(drop_all_na_feature_columns),
                spatial_num_workers=(int(runtime_settings.num_workers) if spatial_df is not None else 1),
            )
        else:
            print("\n--- Skipping Master Feature Matrix generation (--no_master_matrix) ---")

        # --- WRITE LIGHTWEIGHT PROVENANCE ARTIFACTS ---
        if bool(write_provenance_artifacts):
            try:
                source_ledger_df = build_source_ledger_df(
                    inference_csv_dir=inference_csv_dir,
                    poscalc_dir=poscalc_dir,
                    lengths_df=lengths_df,
                    animal_mapping=animal_mapping,
                    file_to_animal=file_to_animal,
                    inf_tag=inf_tag,
                )
                write_source_ledger_csv(output_dir, source_ledger_df)
                print(f"Saved source ledger to {os.path.join(output_dir, 'MMGEN_SOURCE_LEDGER.csv')}")

                row_identity_df = build_row_identity_summary_df(
                    source_ledger_df=source_ledger_df,
                    master_matrix_df=master_matrix_df,
                )
                write_row_identity_summary_csv(output_dir, row_identity_df)
                print(f"Saved row identity summary to {os.path.join(output_dir, 'MMGEN_ROW_IDENTITY_SUMMARY.csv')}")

                write_batch_manifest_json(
                    output_dir=output_dir,
                    inference_csv_dir=inference_csv_dir,
                    poscalc_dir=poscalc_dir,
                    inf_tag=inf_tag,
                    min_bout=min_bout,
                    edge_skip=edge_skip,
                    resume=resume,
                    finalize_only=finalize_only,
                    process_only=process_only,
                    file_filter=file_filter,
                    num_shards=num_shards,
                    shard_index=shard_index,
                    shard_by=shard_by,
                    emit_master_matrix=emit_master_matrix,
                    emit_full_outputs=emit_full_outputs,
                    lengths_df=lengths_df,
                    source_ledger_df=source_ledger_df,
                    row_identity_df=row_identity_df,
                    master_matrix_df=master_matrix_df,
                    animal_mapping=animal_mapping,
                )
                print(f"Saved batch manifest to {os.path.join(output_dir, 'MMGEN_BATCH_MANIFEST.json')}")
            except Exception as e:
                print(f"Warning: failed to write mmgen provenance artifacts: {e}")
    finally:
        # CLEAN UP TEMP FILES
        cleanup_temp_store(temp_store_path, keep_temp_files=effective_keep_temp_files)
        gc.collect()

        # If we acquired a shard lock, release it only on *successful* completion.
        # If an exception is propagating, keep the lock as a strong signal that
        # this shard's outputs may be incomplete.
        if shard_lock is not None:
            try:
                import sys

                if sys.exc_info()[0] is None:
                    shard_lock.release()
            except Exception:
                pass



