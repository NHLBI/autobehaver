"""Stage 3 of the staged FeatureImp core DAG: merge subset weights.

This stage is intentionally lightweight. It merges the shard-local subset
performance and performance-record artifacts, writes the canonical merged CSVs
used downstream, and computes/stores subset weights for the second shard wave.
"""

from __future__ import annotations

import json
import os
import traceback
from datetime import datetime
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .logit_ensemble import load_subset_weights_mapping
from .staged_shared import (
    MERGE_DIRNAME,
    MERGED_PERFORMANCE_PERM_FILENAME,
    MERGED_PERFORMANCE_REAL_FILENAME,
    MERGED_PERM_PARTIALS_FILENAME_TEMPLATE,
    MERGED_REAL_PARTIALS_FILENAME,
    MERGED_SUBSET_PERM_FILENAME,
    MERGED_SUBSET_REAL_FILENAME,
    _concat_csvs,
    _discover_completed_shards,
    _directory_lock,
    _json_dump,
    _json_load,
    _jsonl_load,
    _load_partial_npz,
    _merge_partial_npz_paths,
    _safe_relpath,
    _save_partial_npz,
    _sort_performance_records,
    _sort_subset_performance,
    _validate_merged_counts,
    _write_jsonl,
)
from .plan import load_featureimp_plan_manifest
from .shard_runner import (
    PERFORMANCE_RECORDS_PERM_FILENAME,
    PERFORMANCE_RECORDS_REAL_FILENAME,
    PERM_PARTIALS_FILENAME_TEMPLATE,
    REAL_PARTIALS_FILENAME,
    SUBSET_PERFORMANCE_PERM_FILENAME,
    SUBSET_PERFORMANCE_REAL_FILENAME,
    _load_settings_snapshot,
)
from .subset_weighting import SubsetWeightingConfig, compute_and_write_subset_weights

WEIGHTS_RUNNING_FILENAME = "WEIGHTS_RUNNING.json"
WEIGHTS_DONE_FILENAME = "WEIGHTS_DONE.json"
WEIGHTS_FAILED_FILENAME = "WEIGHTS_FAILED.json"
WEIGHTS_LOCK_DIRNAME = "_featureimp_weights.lock"
WEIGHT_STAGE_SUMMARY_FILENAME = "WEIGHT_STAGE_SUMMARY.json"


def _write_fallback_unit_weights(*, real_subset_df: pd.DataFrame, analysis_folder_path: str) -> str:
    committee_subset_dir = os.path.join(analysis_folder_path, "COMMITTEE_SUBSETS")
    os.makedirs(committee_subset_dir, exist_ok=True)
    out_path = os.path.join(committee_subset_dir, "SUBSET_WEIGHTS_REAL.csv")
    df = pd.DataFrame(real_subset_df).copy()
    if df.empty:
        df = pd.DataFrame(columns=["h_hit", "subset_idx", "subset_weight", "subset_weight_nonzero", "subset_weight_reason"])
    if "subset_complete" in df.columns:
        complete = df["subset_complete"].fillna(False).astype(bool)
    else:
        complete = pd.Series(np.ones((len(df),), dtype=bool), index=df.index)
    df["tau_used"] = 0.0
    df["gamma_used"] = 1.0
    df["subset_weight"] = complete.astype(float)
    df["subset_weight_nonzero"] = complete.astype(bool)
    df["subset_weight_reason"] = np.where(complete.values, "unit_complete_subset", "incomplete_subset")
    df.to_csv(out_path, index=False)
    manifest = {
        "manifest_version": 1,
        "stage": "subset_weighting_fallback_unit",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "analysis_folder": analysis_folder_path,
        "settings": {
            "tau": 0.0,
            "gamma": 1.0,
            "tau_source": "fallback_unit_complete_subset",
        },
        "outputs": {"subset_weights_csv": out_path},
        "summary": {
            "n_subsets_total": int(len(df)),
            "n_nonzero_weights": int(np.sum(df["subset_weight_nonzero"].astype(bool))) if len(df) else 0,
        },
    }
    with open(os.path.join(committee_subset_dir, "WEIGHTING_MANIFEST.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    return out_path


def calculate_subset_weights_run(analysis_folder_path: str) -> str:
    analysis_folder_path = os.path.abspath(str(analysis_folder_path))
    if not os.path.isdir(analysis_folder_path):
        raise FileNotFoundError(f"FeatureImp analysis folder not found: {analysis_folder_path}")

    manifest = load_featureimp_plan_manifest(analysis_folder_path)
    if str(manifest.get("status", "")).strip().lower() != "planned":
        raise ValueError(
            f"FeatureImp analysis folder does not have planned status: {manifest.get('status')!r}"
        )

    merge_dir = os.path.join(analysis_folder_path, MERGE_DIRNAME)
    os.makedirs(merge_dir, exist_ok=True)
    done_path = os.path.join(merge_dir, WEIGHTS_DONE_FILENAME)
    if os.path.exists(done_path):
        print(f"Weight merge already completed successfully; reusing existing outputs: {done_path}")
        return analysis_folder_path

    running_path = os.path.join(merge_dir, WEIGHTS_RUNNING_FILENAME)
    failed_path = os.path.join(merge_dir, WEIGHTS_FAILED_FILENAME)
    lock_dir = os.path.join(merge_dir, WEIGHTS_LOCK_DIRNAME)
    if os.path.exists(failed_path):
        try:
            os.remove(failed_path)
        except Exception:
            pass

    _json_dump(
        {
            "status": "running",
            "analysis_folder": analysis_folder_path,
            "started_at": datetime.now().isoformat(timespec="seconds"),
        },
        running_path,
    )

    try:
        with _directory_lock(lock_dir):
            settings_snapshot = _load_settings_snapshot(analysis_folder_path)
            completed_shards, shard_errors = _discover_completed_shards(analysis_folder_path, manifest)
            if shard_errors:
                raise RuntimeError(
                    "Cannot merge FeatureImp subset weights because shard validation failed:\n- "
                    + "\n- ".join(shard_errors)
                )

            real_partial_paths: List[str] = []
            real_perf_paths: List[str] = []
            perm_perf_paths: List[str] = []
            real_subset_paths: List[str] = []
            perm_subset_paths: List[str] = []
            perm_partial_paths: Dict[int, List[str]] = {int(p): [] for p in range(int(manifest.get("n_permutations", 0) or 0))}

            for _, _, _, done in completed_shards:
                artifacts = dict(done.get("artifacts") or {})
                real_partial_paths.append(os.path.join(analysis_folder_path, str(artifacts["real_partials_npz"])))
                real_perf_paths.append(os.path.join(analysis_folder_path, str(artifacts["performance_records_real_jsonl"])))
                perm_perf_paths.append(os.path.join(analysis_folder_path, str(artifacts["performance_records_perm_jsonl"])))
                real_subset_paths.append(os.path.join(analysis_folder_path, str(artifacts["subset_performance_real_csv"])))
                perm_subset_paths.append(os.path.join(analysis_folder_path, str(artifacts["subset_performance_perm_csv"])))
                for perm_key, rel_path in dict(artifacts.get("perm_partials") or {}).items():
                    perm_partial_paths.setdefault(int(perm_key), []).append(os.path.join(analysis_folder_path, str(rel_path)))

            real_partial = _merge_partial_npz_paths(real_partial_paths)
            perm_partials: Dict[int, Dict[str, np.ndarray]] = {}
            for perm_idx, part_paths in sorted(perm_partial_paths.items()):
                if not part_paths:
                    continue
                perm_partials[int(perm_idx)] = _merge_partial_npz_paths(part_paths)

            real_perf_rows = _sort_performance_records([row for path in real_perf_paths for row in _jsonl_load(path)])
            perm_perf_rows = _sort_performance_records([row for path in perm_perf_paths for row in _jsonl_load(path)])
            real_subset_df = _sort_subset_performance(_concat_csvs(real_subset_paths))
            perm_subset_df = _sort_subset_performance(_concat_csvs(perm_subset_paths))
            _validate_merged_counts(
                manifest=manifest,
                real_subset_df=real_subset_df,
                perm_subset_df=perm_subset_df,
                real_perf_rows=real_perf_rows,
                perm_perf_rows=perm_perf_rows,
            )

            _save_partial_npz(os.path.join(merge_dir, MERGED_REAL_PARTIALS_FILENAME), real_partial)
            for perm_idx, payload in perm_partials.items():
                _save_partial_npz(
                    os.path.join(merge_dir, MERGED_PERM_PARTIALS_FILENAME_TEMPLATE.format(perm_idx=int(perm_idx))),
                    payload,
                )
            _write_jsonl(real_perf_rows, os.path.join(merge_dir, MERGED_PERFORMANCE_REAL_FILENAME))
            _write_jsonl(perm_perf_rows, os.path.join(merge_dir, MERGED_PERFORMANCE_PERM_FILENAME))
            real_subset_df.to_csv(os.path.join(merge_dir, MERGED_SUBSET_REAL_FILENAME), index=False)
            perm_subset_df.to_csv(os.path.join(merge_dir, MERGED_SUBSET_PERM_FILENAME), index=False)

            committee_subset_dir = os.path.join(analysis_folder_path, "COMMITTEE_SUBSETS")
            permuted_subset_dir = os.path.join(analysis_folder_path, "PERMUTED_RESULTS", "COMMITTEE_SUBSETS")
            os.makedirs(committee_subset_dir, exist_ok=True)
            if bool(manifest.get("run_permutation_test", False)):
                os.makedirs(permuted_subset_dir, exist_ok=True)

            real_subset_csv_path = os.path.join(committee_subset_dir, "SUBSET_PERFORMANCE_REAL.csv")
            real_subset_df.to_csv(real_subset_csv_path, index=False)
            perm_subset_csv_path = None
            if bool(manifest.get("run_permutation_test", False)):
                perm_subset_csv_path = os.path.join(permuted_subset_dir, "SUBSET_PERFORMANCE_PERMUTED_ALL.csv")
                perm_subset_df.to_csv(perm_subset_csv_path, index=False)

            subset_weight_tau_quantile = float(settings_snapshot.get("subset_weight_tau_quantile", 0.95) or 0.95)
            subset_weight_gamma = float(settings_snapshot.get("subset_weight_gamma", 2.0) or 2.0)
            subset_weights_path = None
            weighting_manifest_path = None
            try:
                _ = compute_and_write_subset_weights(
                    analysis_folder=str(analysis_folder_path),
                    permuted_base_folder=(os.path.join(analysis_folder_path, "PERMUTED_RESULTS") if bool(manifest.get("run_permutation_test", False)) else None),
                    cfg=SubsetWeightingConfig(
                        tau_quantile=subset_weight_tau_quantile,
                        gamma=subset_weight_gamma,
                    ),
                    verbose=True,
                )
            except Exception as e:
                print(f"WARNING: subset-weight calculation failed; falling back to unit complete-subset weights. Error: {e}")

            subset_weights_path = os.path.join(committee_subset_dir, "SUBSET_WEIGHTS_REAL.csv")
            if not os.path.exists(subset_weights_path):
                subset_weights_path = _write_fallback_unit_weights(real_subset_df=real_subset_df, analysis_folder_path=analysis_folder_path)
            weighting_manifest_path = os.path.join(committee_subset_dir, "WEIGHTING_MANIFEST.json")
            if not os.path.exists(weighting_manifest_path):
                weighting_manifest_path = None

            subset_weights = load_subset_weights_mapping(subset_weights_path)
            if not subset_weights or not any(float(w) > 0.0 for w in subset_weights.values()):
                print("WARNING: subset weighting produced no positive weights; falling back to unit complete-subset weights.")
                subset_weights_path = _write_fallback_unit_weights(real_subset_df=real_subset_df, analysis_folder_path=analysis_folder_path)
                weighting_manifest_path = os.path.join(committee_subset_dir, "WEIGHTING_MANIFEST.json")
                subset_weights = load_subset_weights_mapping(subset_weights_path)
            summary = {
                "status": "completed",
                "analysis_folder": analysis_folder_path,
                "completed_at": datetime.now().isoformat(timespec="seconds"),
                "real_task_count": int(len(real_subset_df)),
                "perm_task_count": int(len(perm_subset_df)),
                "real_model_count": int(len(real_perf_rows)),
                "perm_model_count": int(len(perm_perf_rows)),
                "n_nonzero_weights": int(np.sum(np.asarray(list(subset_weights.values()), dtype=float) > 0.0)) if subset_weights else 0,
                "artifacts": {
                    "merged_real_partials_npz": _safe_relpath(os.path.join(merge_dir, MERGED_REAL_PARTIALS_FILENAME), analysis_folder_path),
                    "merged_real_subset_performance_csv": _safe_relpath(os.path.join(merge_dir, MERGED_SUBSET_REAL_FILENAME), analysis_folder_path),
                    "merged_real_performance_jsonl": _safe_relpath(os.path.join(merge_dir, MERGED_PERFORMANCE_REAL_FILENAME), analysis_folder_path),
                    "subset_weights_csv": _safe_relpath(subset_weights_path, analysis_folder_path),
                    "weighting_manifest_json": _safe_relpath(weighting_manifest_path, analysis_folder_path),
                    "canonical_real_subset_performance_csv": _safe_relpath(real_subset_csv_path, analysis_folder_path),
                    "canonical_perm_subset_performance_csv": _safe_relpath(perm_subset_csv_path, analysis_folder_path),
                },
            }
            _json_dump(summary, os.path.join(merge_dir, WEIGHT_STAGE_SUMMARY_FILENAME))
            _json_dump(summary, done_path)
            try:
                os.remove(running_path)
            except Exception:
                pass
            print(f"FeatureImp weight merge completed: {done_path}")
            return analysis_folder_path
    except Exception as e:
        _json_dump(
            {
                "status": "failed",
                "analysis_folder": analysis_folder_path,
                "failed_at": datetime.now().isoformat(timespec="seconds"),
                "error_type": type(e).__name__,
                "error": str(e),
                "traceback": traceback.format_exc(),
            },
            failed_path,
        )
        try:
            os.remove(running_path)
        except Exception:
            pass
        raise


__all__ = [
    "WEIGHTS_RUNNING_FILENAME",
    "WEIGHTS_DONE_FILENAME",
    "WEIGHTS_FAILED_FILENAME",
    "WEIGHTS_LOCK_DIRNAME",
    "WEIGHT_STAGE_SUMMARY_FILENAME",
    "calculate_subset_weights_run",
]
