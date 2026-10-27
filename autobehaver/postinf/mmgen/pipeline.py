"""Discover and validate MMGen inputs, process recordings, and aggregate results in
HDF5.

Finalize animal-level feature tables, the declared master matrix, and provenance using
explicit runtime settings.
"""

from __future__ import annotations

import gc
import inspect
from functools import wraps
import json
import os
import socket
from datetime import datetime, timezone
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from autobehaver.progress import progress as tqdm

from autobehaver.validation import StageValidationReport
from autobehaver.completion import verified_run
from autobehaver.portable_paths import original_name

from .hdf_store import (
    HDF_KEYS,
    HDF_MIN_ITEMSIZE,
    build_hdf_min_itemsize_from_observed_strings,
    cleanup_temp_store,
    file_has_any_rows,
    get_done_source_files,
    hdf_safe_append,
    initialize_temp_store,
    open_store,
    purge_file_from_store,
)
from .feature_matrix import (
    MMGenFinalizationSources,
    create_feature_matrix_from_sources,
    validate_master_matrix_dataframe,
)
from .ledger import (
    build_row_identity_summary_df,
    build_source_ledger_df,
    write_batch_manifest_json,
    write_row_identity_summary_csv,
    write_source_ledger_csv,
)
from .worker import process_video_data, initialize_mmgen_worker

# Analysis helpers used during finalization and report generation.
from .kinematics import (
    _reconstruct_kinematic_table_from_aggregates,
    calculate_and_save_kinematic_summary,
)
from .transitions import (
    _is_impossible_pose_orient_transition,
    calculate_and_save_transition_analysis,
)
from .occupancy import calculate_and_save_occupancy_analysis
from .measurements import calculate_and_save_pooled_measurements
from .integrity import (calculation_identity, source_identities, bind_store_identity,
                        validate_commits, empty_inventory, content_hash, frame_fingerprint, commit_fingerprint, IDENTITY_FILENAME)
from autobehaver.atomic_io import atomic_write_json, atomic_write_dataframe_csv, sha256_file
from .postural import calculate_and_save_postural_summary, calculate_postural_summary_from_store
from .summary_pooling import pool_bout_summaries, pool_sleep_summaries
from .time_budgets import calculate_and_save_time_budgets
from .mapping import create_animal_mapping
from .naming import collapse_individual_id, parse_group_and_individual

from .batching import make_batches, ordered_worker_results
from .discovery import list_inference_csv_files, print_inference_file_diagnostic
from .cohort import bind_shard_cohort
from .locks import ShardLock
from .publication import canonical_generation
from .runtime import MMGenRuntimeSettings
from .recording_commits import append_recording_commits, read_recording_commits


def _write_mmgen_validation(
    *,
    output_dir: str,
    selected_files: list[str],
    committed_files: set[str],
    initially_committed_files: set[str],
    master_matrix_df: pd.DataFrame | None = None,
) -> StageValidationReport:
    """Record selected-source coverage and master-matrix availability checks for this
    run.
    """
    report = StageValidationReport.for_items(
        "mmgen",
        [str(file) for file in selected_files],
        metadata={"output_directory": str(Path(output_dir).resolve())},
    )
    committed = {str(file) for file in committed_files}
    initially_committed = {str(file) for file in initially_committed_files}
    for source_file in selected_files:
        item = str(source_file)
        if item not in committed:
            report.add_outcome(item, "failed", reason="source was not committed to the MMGen store")
        elif item in initially_committed:
            report.add_outcome(
                item,
                "reused",
                reason="existing MMGen commit marker was reused during resume/finalization",
            )
        else:
            report.add_outcome(item, "succeeded")

    if master_matrix_df is not None:
        try:
            validate_master_matrix_dataframe(master_matrix_df)
        except Exception as exc:
            report.add_issue(
                "master_matrix_integrity",
                "error",
                f"Published master feature matrix failed validation: {exc}",
            )
        else:
            feature_columns = [
                column
                for column in master_matrix_df.select_dtypes(include=[np.number]).columns
                if column != "total_frames_analyzed"
            ]
            if feature_columns:
                missing_fraction = master_matrix_df[feature_columns].isna().mean()
                high_missing = missing_fraction[missing_fraction > 0.25].sort_values(ascending=False)
                if not high_missing.empty:
                    report.add_issue(
                        "high_feature_missingness",
                        "warning",
                        f"{len(high_missing)} numeric feature(s) are missing for more than 25% of animals.",
                        details={
                            "examples": {
                                str(column): float(value)
                                for column, value in high_missing.head(25).items()
                            }
                        },
                    )

    report.write(Path(output_dir) / "MMGEN_VALIDATION.json")
    return report


def _build_identity_hdf_min_itemsize(inf_files: list[str]) -> dict[str, int]:
    """Preflight HDF string widths from the selected inference filenames.

    HDF5 table string columns are fixed-width after creation.  The group,
    individual, sex, and source_file columns are all derived from filenames, so
    we can size them before the first table append instead of discovering too
    late that a later batch contains a longer label.
    """

    observed: dict[str, list[object]] = {
        "source_file": [],
        "group": [],
        "group_label": [],
        "individual": [],
        "individual_id": [],
        "sex": [],
    }
    parse_error_examples: list[str] = []

    for f in inf_files or []:
        basename = os.path.basename(str(f))
        observed["source_file"].append(basename)
        try:
            group, individual_id, sex = parse_group_and_individual(str(f))
            animal_id = collapse_individual_id(individual_id)

            observed["group"].append(group)
            observed["group_label"].append(group)
            observed["individual"].extend([individual_id, animal_id])
            observed["individual_id"].extend([individual_id, animal_id])
            observed["sex"].append(sex)
        except Exception:
            if len(parse_error_examples) < 5:
                parse_error_examples.append(basename)

    min_itemsize = build_hdf_min_itemsize_from_observed_strings(observed)

    # Report the resolved dynamic string widths once per run.
    show_cols = ["group", "group_label", "individual", "individual_id", "source_file", "sex"]
    shown = ", ".join(f"{c}={int(min_itemsize.get(c, HDF_MIN_ITEMSIZE.get(c, 64)))}" for c in show_cols)
    print(f"[mmgen] HDF string-width preflight: inspected {len(inf_files or [])} candidate file name(s).")
    print(f"[mmgen] HDF string-width preflight: {shown}")
    if parse_error_examples:
        print(
            "[mmgen] Warning: could not parse some filename-derived identities during HDF preflight; "
            f"examples={parse_error_examples}"
        )

    return min_itemsize


def _run_post_inference_analysis_owned(
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
    file_list: str | None = None,
    drop_all_na_feature_columns: bool = True,
    write_provenance_artifacts: bool = True,
    spatial_in_memory_for_matrix: bool = False,
    caller_holds_shard_lock: bool = False,
):
    """Process and finalize the selected MMGen recording set using ordered workers and
    HDF aggregation.
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
    if file_list:
        from autobehaver.batch_inputs import select_batch_inputs
        selected = set(select_batch_inputs([os.path.join(inference_csv_dir, str(f)) for f in inf_files], file_list))
        inf_files = [f for f in inf_files if os.path.join(inference_csv_dir, str(f)) in selected]
    if file_filter is not None and str(file_filter).strip() != "":
        needle = str(file_filter).strip().lower()
        inf_files = [f for f in inf_files if needle in original_name(os.path.join(inference_csv_dir, str(f))).lower()]

    # Preflight string widths from the full selected input set *before*
    # sharding. This keeps classic process-only shards compatible for later
    # HDF-store merging, while still working for matrix-first shard-local runs.
    hdf_preflight_files = list(inf_files)

    # Deterministic sharding (optional)
    if int(num_shards) > 1:
        if not 0 <= int(shard_index) < int(num_shards):
            raise ValueError(f'shard_index must be in [0, {int(num_shards)-1}], got {shard_index}')
        cohort = bind_shard_cohort(Path(output_dir) / str(shards_dirname), inf_files,
            inference_csv_dir=inference_csv_dir, num_shards=int(num_shards), shard_by=str(shard_by))
        partition = {record['source_file'] for record in cohort['files']
                     if record['shard_index'] == int(shard_index)}
        inf_files = [name for name in inf_files if name in partition]
        print(
            f"[mmgen] Sharding enabled: shard {int(shard_index)}/{int(num_shards)-1} "
            f"({len(inf_files)} files selected; strategy={shard_by})."
        )

    # Working output directory
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

    store_exists = os.path.exists(temp_store_path)
    reuse = bool(store_exists and (resume or finalize_only or process_only))
    if store_exists and process_only and not resume:
        raise RuntimeError('Existing process-only store requires explicit --resume; duplicate processing rejected.')
    from .finalize_compatibility import resolve_finalization_identity, publish_execution_record
    effective_identity, execution_record = resolve_finalization_identity(
        calculation_identity(runtime_settings, min_bout=min_bout, edge_skip=edge_skip, inf_tag=inf_tag),
        work_dir, finalize_only=finalize_only)
    identity_payload = bind_store_identity(work_dir,
        effective_identity,
        source_identities(inf_files, inference_csv_dir=inference_csv_dir,
                          poscalc_dir=poscalc_dir, inf_tag=inf_tag,
                          source_frame_offset=runtime_settings.source_frame_offset), reuse=reuse)

    if int(num_shards) > 1:
        shard_manifest = {
            'created_at': datetime.now(timezone.utc).isoformat(),
            'num_shards': int(num_shards), 'shard_index': int(shard_index), 'shard_by': str(shard_by),
            'n_selected_files': len(inf_files), 'selected_files': sorted(map(str, inf_files)),
            'cohort_sha256': cohort['cohort_sha256'],
            'file_filter': None if file_filter is None else str(file_filter),
            'effective_output_dir': str(Path(effective_output_dir).resolve()),
        }
        atomic_write_json(shard_manifest, Path(work_dir) / 'SHARD_MANIFEST.json')
        atomic_write_dataframe_csv(pd.DataFrame({'inference_csv': sorted(map(str, inf_files))}),
                                   Path(work_dir) / 'shard_manifest.csv', index=False)

    if not inf_files:
        if int(num_shards) > 1 and shard_done_path is not None:
            # Strict sharded merge expects every requested shard to emit a
            # completion marker. When a shard receives no files, emit an empty
            # temp store plus a zero-count done marker rather than silently
            # disappearing.
            with open_store(temp_store_path, mode='w'):
                pass
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
            done_payload.update(cohort_sha256=cohort['cohort_sha256'], calculation_sha256=identity_payload['calculation_sha256'], sources_sha256=identity_payload['sources_sha256'], temp_store_sha256=sha256_file(temp_store_path))
            atomic_write_json(done_payload, shard_done_path)
            print("[mmgen] No inference files selected for this shard. Wrote empty shard temp store and completion marker.")
            return

        raise FileNotFoundError("No inference files matched the MMGen input selection.")

    # HDF string-width preflight
    hdf_min_itemsize = _build_identity_hdf_min_itemsize(
        [os.path.join(inference_csv_dir, str(f)) for f in (hdf_preflight_files if hdf_preflight_files else inf_files)]
    )

    # DIAGNOSTIC: Print all input files and their parsed labels
    print_inference_file_diagnostic(inf_files, inference_csv_dir)

    # CREATE CENTRALIZED ANIMAL MAPPING
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
            raise RuntimeError("Failed to create a non-empty animal mapping for selected inputs.")

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
        'hdf_min_itemsize': hdf_min_itemsize,
    }

    current_inventory = empty_inventory()
    current_fingerprints = {}
    def _hdf_append(store, key, df) -> None:
        if df is not None and not df.empty:
            if 'source_file' not in df or not df['source_file'].astype(str).eq(filename).all():
                raise RuntimeError(f'Invalid recording provenance in {key}: {filename}')
        hdf_safe_append(store, key, df, min_itemsize_map=hdf_min_itemsize)
        if key in current_inventory and df is not None:
            current_inventory[key] += len(df)
            if not df.empty:
                stop = int(store.get_storer(key).nrows)
                saved = store.select(key, start=stop-len(df), stop=stop)
                current_fingerprints[key] = frame_fingerprint(saved)

    # Single HDF5 store for all temporary tables

    # Resume requires keeping the temp store around.
    effective_keep_temp_files = bool(runtime_settings.keep_temp_files or resume or finalize_only or process_only)
    if (resume or finalize_only or process_only) and not runtime_settings.keep_temp_files:
        print("[mmgen] Retaining the temporary store required by this resume/processing/finalization stage.")

    # Determine which files are already committed in the temp store.
    store_exists = os.path.exists(temp_store_path)
    done_files: set[str] = set()
    initially_committed_files: set[str] = set()
    pending_files = list(inf_files)

    if resume or finalize_only or process_only:
        if store_exists:
            done_files = validate_commits(temp_store_path, identity_payload, selected_files=inf_files,
                                          allow_uncommitted=bool(resume and not finalize_only))
            initially_committed_files = set(done_files)

        if finalize_only and not store_exists:
            raise FileNotFoundError(
                f"finalize_only was requested but no temp store exists at: {temp_store_path}"
            )

        # validate_commits also checks empty or uncommitted native stores.
        # Interrupted sources remain pending and their rows are purged below;
        # failure before the first recording commits is therefore retryable.

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
        with open_store(temp_store_path, mode='w'):
            pass
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

    # Helper: append one progress row to a CSV (best-effort).
    def _append_progress_row(row: dict) -> None:
        """Append one recording's progress record without interrupting scientific
        processing on a log-write error.
        """
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
    args_list = [
        (os.path.join(inference_csv_dir, f), poscalc_dir, min_bout, edge_skip, inf_tag, None)
        for f in pending_files
    ]
    # Purge interrupted rows before opening the single writer.
    if resume and os.path.exists(temp_store_path):
        for f in pending_files:
            _, ind_id, _ = parse_group_and_individual(os.path.join(inference_csv_dir, str(f)))
            if file_has_any_rows(temp_store_path, source_file=str(f), individual_id=ind_id):
                purge_file_from_store(temp_store_path, source_file=str(f), individual_id=ind_id)
    results = ordered_worker_results(process_video_data, args_list,
        num_workers=int(runtime_settings.num_workers), initializer=initialize_mmgen_worker,
        initargs=(config,))
    with open_store(temp_store_path, mode=('a' if pending_files else 'r')) as store:
        for j, (
            bout_features_dict, measurement_result, video_length, positional_df,
            postural_features_df, bout_summary_df, sleep_summary_df,
            failure_info,
        ) in enumerate(tqdm(results, total=len(args_list), desc="Analyzing recordings")):
            this_file = args_list[j][0]
            filename = os.path.basename(this_file)
            group_name, individual_id, sex = parse_group_and_individual(this_file)
            current_inventory = empty_inventory()
            current_fingerprints = {}
            if failure_info is not None:
                failure_row = dict(failure_info)
                failure_row.setdefault('source_file', filename)
                worker_failures.append({k: ('' if v is None else str(v)) for k,v in failure_row.items()})
                continue
            if not isinstance(video_length, (int, np.integer)) or int(video_length) <= 0:
                raise RuntimeError(f'Invalid frame denominator for {filename}')
            if runtime_settings.make_measurements and measurement_result is None:
                raise RuntimeError(f'Requested measurement summary missing for {filename}')
            if bout_summary_df is None or bout_summary_df.empty:
                raise RuntimeError(f'Required bout summary missing for {filename}')
            _hdf_append(store, HDF_KEYS['bout_summary_tmp'], bout_summary_df)
            if sleep_summary_df is not None and not sleep_summary_df.empty:
                _hdf_append(store, HDF_KEYS['sleep_summary_tmp'], sleep_summary_df)

            if bout_features_dict:
                for suffix, df_bout in bout_features_dict.items():
                    if str(suffix).startswith('__'):
                        continue
                    if df_bout is not None and not df_bout.empty:
                        key = HDF_KEYS[f"bouts{suffix}"]
                        _hdf_append(store, key, df_bout)

                        # Build per-file aggregates for this smoothing level
                        base_keys = ['individual','group','sex','category','bout_type']
                        g = df_bout.groupby(base_keys, observed=True)

                        # Bout length aggregates
                        agg_len = g['length'].agg(['count','sum']).rename(columns={'count':'n_bouts','sum':'sum_length'})
                        sum_length_sq = g['length'].apply(lambda x: np.square(x.astype('float64')).sum())
                        agg_len['sum_length_sq'] = sum_length_sq

                        # Kinematic scalar columns: ends with _mean, _std, _min, _max, _median, or _skewness
                        kin_cols = [c for c in df_bout.columns if c.endswith(('_mean', '_std', '_min', '_max', '_median', '_skewness'))]
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
                        _hdf_append(store, agg_key, agg_out)

            # Native transitions are computed before any kinematic eligibility filter.
            transition_counts = bout_features_dict.get('__transition_counts__') if bout_features_dict else None
            if transition_counts is None:
                if failure_info is None:
                    raise RuntimeError('Worker did not return verified native transition counts.')
            elif not transition_counts.empty:
                _hdf_append(store, HDF_KEYS['transitions_tmp'], transition_counts)
            inst_df = None  # Release this temporary result with the per-recording payload.

            # Retain bounded candidates for the final per-animal measurement sample.
            if runtime_settings.make_measurements and measurement_result is not None:
                meas_df = measurement_result.copy() if isinstance(measurement_result, pd.DataFrame) else pd.DataFrame([measurement_result])
                meas_df['source_file'] = filename
                _hdf_append(store, HDF_KEYS['measurements_tmp'], meas_df)

            if positional_df is not None and not positional_df.empty:
                _hdf_append(store, HDF_KEYS['positional'], positional_df)

            if postural_features_df is not None and not postural_features_df.empty:
                _hdf_append(store, HDF_KEYS['postural'], postural_features_df)


            # Commit marker for resume: one row per processed inference CSV.
            # Write this *last* so its presence implies the file was fully
            # appended across all tables.
            store.flush(fsync=True)
            vid_df = pd.DataFrame([{
                'source_file':filename, 'group':group_name, 'individual':individual_id,
                'sex':sex, 'n_frames':int(video_length),
                'calculation_sha256':identity_payload['calculation_sha256'],
                'input_sha256':content_hash(identity_payload['sources'][filename]),
                'output_inventory':json.dumps(current_inventory, sort_keys=True),
                'output_fingerprints':json.dumps(current_fingerprints, sort_keys=True),
            }])
            vid_df['commit_sha256'] = commit_fingerprint(vid_df.iloc[0])
            append_recording_commits(store, vid_df)
            store.flush(fsync=True)
            done_files.add(filename)
            progress_row = vid_df.iloc[0].to_dict()
            for column in ('pose_gap_filled_frames', 'orient_gap_filled_frames'):
                progress_row[column] = int(bout_summary_df[column].iloc[0])
            _append_progress_row(progress_row)

            # Release per-recording objects after committing their outputs.
            try:
                del bout_features_dict, measurement_result, positional_df, postural_features_df, bout_summary_df, sleep_summary_df, failure_info, inst_df
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

    # Finalize-only already verified every scientific table before this read-only
    # section. The caller holds the canonical/shard owner throughout, so repeat
    # the scan only when processing could have changed the store.
    if not finalize_only:
        done_files = validate_commits(temp_store_path, identity_payload, selected_files=inf_files)

    # If this was a process-only run, stop here (leave temp store intact).
    if process_only:
        # For sharded runs, write a completion marker and release the shard lock.
        if int(num_shards) > 1 and shard_done_path is not None:
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
            done_payload.update(cohort_sha256=cohort['cohort_sha256'], calculation_sha256=identity_payload['calculation_sha256'],
                sources_sha256=identity_payload['sources_sha256'], temp_store_sha256=sha256_file(temp_store_path))
            atomic_write_json(done_payload, shard_done_path)

        validation = _write_mmgen_validation(
            output_dir=effective_output_dir,
            selected_files=list(inf_files),
            committed_files=set(done_files),
            initially_committed_files=initially_committed_files,
            master_matrix_df=None,
        )
        validation.raise_for_errors()

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
        transition_pose_tidy_df = None
        transition_pose_orient_tidy_df = None
        spatial_data_path = None
        spatial_df = None

        def progress(label):
            print(f"[mmgen] {datetime.now(timezone.utc).isoformat()} Finalize: {label}", flush=True)

        # Read aggregates instead of full tables
        progress('load compact summaries')
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
            with open_store(temp_store_path, mode='r') as lengths_store:
                lengths_df = read_recording_commits(lengths_store)
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

            animal_summary = pool_bout_summaries(master_bout_summary_df)

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

            animal_sleep_summary = pool_sleep_summaries(master_sleep_df)

            if emit_full_outputs:
                sleep_output_path = os.path.join(output_dir, '12_Sleep_Features.csv')
                animal_sleep_summary.to_csv(sleep_output_path, index=False)
                print(f"Saved sleep features to {sleep_output_path}")
        else:
            print("No sleep summary data was generated to create 12_Sleep_Features.csv.")

        # Finalize the enabled output families.

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
            progress('body measurements')
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
            progress('postural summaries')
            try:
                normalizer_map = None
                if measurements_df is not None and 'length_avg_mean' in measurements_df.columns:
                    measurement_ids = {value: _collapse_individual_with_mapping(value)
                                       for value in measurements_df['individual_id'].unique()}
                    measurements_df['animal_id'] = measurements_df['individual_id'].map(measurement_ids)
                    normalizer_map = measurements_df.groupby('animal_id').mean(numeric_only=True)['length_avg_mean']
                postural_summary_df = calculate_postural_summary_from_store(
                    temp_store_path, output_dir, normalizer_map=normalizer_map,
                    individual_to_animal=individual_to_animal_map, save_outputs=emit_full_outputs)
            except (FileNotFoundError, KeyError) as exc:
                if any(json.loads(v).get(HDF_KEYS['postural'], 0) for v in lengths_df['output_inventory']):
                    raise RuntimeError('Persisted postural analysis failed') from exc

        # Analyses based on HDF5 aggregates
        bb_minimum = int(runtime_settings.min_bout_count_for_features) if runtime_settings.filter_features_by_bout_count else 0
        inst_agg = agg_tables.get('_inst', pd.DataFrame())
        if not inst_agg.empty:
            progress('bout-based kinematics')
            # 5. Kinematic Features (BB) - aggregated per-bout stats
            kin_tables = _reconstruct_kinematic_table_from_aggregates(
                agg_tables,
                min_bout_count_for_features=bb_minimum,
            )
            kinematic_sheets = calculate_and_save_kinematic_summary(
                kin_tables,
                output_dir,
                min_bout_count_for_features=bb_minimum,
                save_outputs=emit_full_outputs,
            )

        else:
            print('No eligible BB kinematic bouts; categorical transitions remain independent.')

        if runtime_settings.make_detailed_transition_outputs:
            progress('transition summaries')
            with open_store(temp_store_path, mode='r') as store:
                key = HDF_KEYS['transitions_tmp']
                aggregated_transitions = store.select(key) if key in store else pd.DataFrame()
            if not aggregated_transitions.empty:
                transition_results = calculate_and_save_transition_analysis(
                    aggregated_transitions, output_dir,
                    run_transition_statistics=run_transition_statistics, save_outputs=emit_full_outputs)
                transition_pose_tidy_df = transition_results.get('pose_tidy_df')
                transition_pose_orient_tidy_df = transition_results.get('pose_orient_tidy_df')

        # 9, 10, 11. Spatial Occupancy and Orientation Analysis
        occupancy_completed = False
        if runtime_settings.calculate_occupancy:
            progress('occupancy and orientation plots')
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
                    if spatial_result is None:
                        raise RuntimeError('Requested spatial analysis produced no transformed data')
                    occupancy_completed = True
                else:
                    print("Could not find positional data key in HDF5 store; skipping occupancy analysis.")
                
            except (FileNotFoundError, KeyError) as exc:
                if any(json.loads(v).get(HDF_KEYS['positional'], 0) for v in lengths_df['output_inventory']):
                    raise RuntimeError('Persisted occupancy analysis failed') from exc
                pass
        
        # CREATE MASTER FEATURE MATRIX
        master_matrix_df = None
        if emit_master_matrix:
            progress('master matrix and spatial summaries')
            print("\n--- Generating Master Feature Matrix ---")
            sources = MMGenFinalizationSources(
                bout_summary_df=animal_summary,
                time_budget_sheets=time_budget_sheets,
                measurements_df=measurements_df,
                postural_summary_df=postural_summary_df,
                kinematic_sheets=kinematic_sheets,
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
                spatial_num_workers=int(runtime_settings.num_workers),
            )
        else:
            print("\n--- Skipping Master Feature Matrix generation (--no_master_matrix) ---")

        # WRITE LIGHTWEIGHT PROVENANCE ARTIFACTS
        if bool(write_provenance_artifacts):
            source_ledger_df = build_source_ledger_df(
                inference_csv_dir=inference_csv_dir,
                poscalc_dir=poscalc_dir,
                lengths_df=lengths_df,
                animal_mapping=animal_mapping,
                file_to_animal=file_to_animal,
                inf_tag=inf_tag,
                source_frame_offset=runtime_settings.source_frame_offset,
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
                source_frame_offset=int(runtime_settings.source_frame_offset),
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

        validation = _write_mmgen_validation(
            output_dir=output_dir,
            selected_files=list(inf_files),
            committed_files=set(done_files),
            initially_committed_files=initially_committed_files,
            master_matrix_df=master_matrix_df,
        )
        validation.raise_for_errors()
        required = []
        if emit_master_matrix:
            required += ['MASTER_FEATURE_MATRIX.csv', 'MASTER_FEATURE_MATRIX_UNITS.csv', 'MASTER_FEATURE_MATRIX_SCHEMA.json']
        family_status = {}
        if emit_full_outputs:
            families = {
                'bout_summary': (animal_summary, ['1_Individual_Bout_Summary.csv']),
                'time_budgets': (time_budget_sheets, ['2_Time_Budgets.xlsx']),
                'measurements': (measurements_df, ['3_Standardized_Measurements.xlsx']),
                'postural': (postural_summary_df, ['4_Postural_Features.csv']),
                'bb_kinematics': (kinematic_sheets, ['5_Kinematic_Features.xlsx']),
                'pose_transitions': (transition_pose_tidy_df, ['6_Pose-Only_Transitions.xlsx']),
                'combined_transitions': (transition_pose_orient_tidy_df, ['7_Pose+Orient_Transitions.xlsx']),
                'sleep': (animal_sleep_summary, ['12_Sleep_Features.csv']),
                'spatial': (True if occupancy_completed else None, ['MMGEN_OCCUPANCY_OUTPUTS.json']),
            }
            for family, (result, names) in families.items():
                applicable = result is not None and (not hasattr(result, 'empty') or not result.empty) and (not isinstance(result, dict) or bool(result))
                family_status[family] = 'completed' if applicable else 'inapplicable'
                if applicable:
                    required.extend(names)
            if occupancy_completed:
                occupancy_receipt = json.loads((Path(output_dir)/'MMGEN_OCCUPANCY_OUTPUTS.json').read_text())
                for name, expected_hash in occupancy_receipt['artifacts'].items():
                    target = Path(output_dir)/name
                    if not target.is_file() or sha256_file(target) != expected_hash:
                        raise RuntimeError(f'Missing or altered finalized occupancy output: {target}')
                    required.append(name)
        if write_provenance_artifacts:
            required += ['MMGEN_SOURCE_LEDGER.csv', 'MMGEN_ROW_IDENTITY_SUMMARY.csv', 'MMGEN_BATCH_MANIFEST.json']
        publish_execution_record(execution_record, output_dir)
        if execution_record is not None:
            required.append('MMGEN_FINALIZATION_EXECUTION.json')
        artifacts = {}
        for name in required:
            target = Path(output_dir)/name
            if not target.is_file():
                raise RuntimeError(f'Required finalized output missing: {target}')
            artifacts[name] = sha256_file(target)
        atomic_write_json(identity_payload, Path(work_dir) / IDENTITY_FILENAME)
        atomic_write_json({'calculation_sha256':identity_payload['calculation_sha256'],
            'sources_sha256':identity_payload['sources_sha256'], 'artifacts':artifacts,
            'output_families':family_status},
            Path(output_dir)/'MMGEN_OUTPUT_RECEIPT.json')
        if not runtime_settings.keep_temp_files:
            import shutil
            parts = Path(output_dir) / '.occupancy_parts'
            if parts.is_dir():
                shutil.rmtree(parts)

    finally:
        # CLEAN UP TEMP FILES
        import sys
        cleanup_temp_store(temp_store_path, keep_temp_files=(effective_keep_temp_files or sys.exc_info()[0] is not None))
        gc.collect()



@verified_run
@wraps(_run_post_inference_analysis_owned)
def run_post_inference_analysis(*args, **kwargs):
    """Acquire ownership before identity, empty-shard, or output mutations."""
    bound = inspect.signature(_run_post_inference_analysis_owned).bind(*args, **kwargs)
    bound.apply_defaults()
    options = bound.arguments
    if int(options['num_shards']) > 1:
        shard_dir = Path(options['output_dir']) / options['shards_dirname'] / f"shard_{int(options['shard_index']):03d}_of_{int(options['num_shards']):03d}"
        if options['caller_holds_shard_lock']:
            return _run_post_inference_analysis_owned(*bound.args, **bound.kwargs)
        path = shard_dir / '_mmgen_work/_mmgen_shard.lock'
        path.parent.mkdir(parents=True, exist_ok=True)
        with ShardLock(path):
            options['caller_holds_shard_lock'] = True
            return _run_post_inference_analysis_owned(*bound.args, **bound.kwargs)
    with canonical_generation(options['output_dir'], producing=not options['process_only']):
        return _run_post_inference_analysis_owned(*bound.args, **bound.kwargs)
