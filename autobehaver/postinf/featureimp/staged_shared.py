"""Shared artifact loading, verification, reduction, and export helpers for FeatureImp
merging.
"""

from __future__ import annotations

import json
import os
import warnings
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .committee import FixedBinaryLabelEncoder
from .compat import calculate_ensemble_metrics
from .prediction_export_utils import insert_group_separator_rows
from .shard_runner import (
    LOCK_DIRNAME as SHARD_LOCK_DIRNAME,
    PERFORMANCE_RECORDS_PERM_FILENAME,
    PERFORMANCE_RECORDS_REAL_FILENAME,
    PERM_PARTIALS_FILENAME_TEMPLATE,
    REAL_PARTIALS_FILENAME,
    SHARD_DONE_FILENAME,
    SHARD_FAILED_FILENAME,
    SHARD_OUTPUT_DIRNAME,
    SHARD_RUNNING_FILENAME,
    SUBSET_PERFORMANCE_PERM_FILENAME,
    SUBSET_PERFORMANCE_REAL_FILENAME,
    _directory_lock,
    _json_dump,
    _json_load,
    _jsonl_load,
    _load_featureimp_context,
    _load_settings_snapshot,
    _validate_completed_shard,
    _validate_planned_artifact_hashes,
)

MERGE_DIRNAME = "_featureimp_merge"
MERGED_REAL_PARTIALS_FILENAME = "MERGED_REAL_PARTIALS.npz"
MERGED_PERM_PARTIALS_FILENAME_TEMPLATE = "MERGED_PERM_PARTIALS_perm{perm_idx:03d}.npz"
MERGED_PERFORMANCE_REAL_FILENAME = "PERFORMANCE_RECORDS_REAL_MERGED.jsonl"
MERGED_PERFORMANCE_PERM_FILENAME = "PERFORMANCE_RECORDS_PERM_MERGED.jsonl"
MERGED_SUBSET_REAL_FILENAME = "SUBSET_PERFORMANCE_REAL_MERGED.csv"
MERGED_SUBSET_PERM_FILENAME = "SUBSET_PERFORMANCE_PERM_MERGED.csv"
COMMITTEE_POOLED_PERFORMANCE_FILENAME = "COMMITTEE_POOLED_PERFORMANCE.csv"


def _write_jsonl(rows: Iterable[Mapping[str, Any]], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, indent=None, default=str, allow_nan=True))
            f.write("\n")


def _load_partial_npz(path: str) -> Dict[str, np.ndarray]:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    out: Dict[str, np.ndarray] = {}
    with np.load(path, allow_pickle=False) as data:
        for key in data.files:
            if key.startswith("meta__"):
                continue
            out[key] = np.array(data[key])
    return out


def _save_partial_npz(path: str, payload: Mapping[str, Any]) -> None:
    arrays = {k: (np.asarray(v) if not np.isscalar(v) else np.asarray(v)) for k, v in payload.items()}
    np.savez(path, **arrays)


def _concat_csvs(paths: Sequence[str]) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for path in paths:
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path)
        if len(df) == 0:
            continue
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, axis=0, ignore_index=True)


def _merge_partial_npz_paths(paths: Sequence[str]) -> Dict[str, np.ndarray]:
    """Sum partial arrays only when their keys and shapes agree."""
    merged: Optional[Dict[str, np.ndarray]] = None
    for path in paths:
        partial = _load_partial_npz(path)
        if merged is None:
            merged = {k: np.array(v, copy=True) for k, v in partial.items()}
            continue
        if set(partial.keys()) != set(merged.keys()):
            raise ValueError(
                "Partial payload key mismatch during merge. "
                f"Expected keys={sorted(merged.keys())}, observed keys={sorted(partial.keys())}."
            )
        for key, value in partial.items():
            arr = np.asarray(value)
            if merged[key].shape != arr.shape:
                raise ValueError(
                    f"Shape mismatch while merging partial '{key}': expected {merged[key].shape}, got {arr.shape}."
                )
            merged[key] = merged[key] + arr
    if merged is None:
        raise ValueError("No partial archive paths were supplied for merge.")
    return merged


def _sort_performance_records(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    out = [dict(r) for r in rows]
    out.sort(
        key=lambda r: (
            bool(r.get("is_permuted", False)),
            int(r.get("permutation_idx")) if r.get("permutation_idx") not in (None, "") else -1,
            int(r.get("h_hit")) if r.get("h_hit") not in (None, "") else -1,
            int(r.get("subset_idx")) if r.get("subset_idx") not in (None, "") else -1,
            int(r.get("fold_idx")) if r.get("fold_idx") not in (None, "") else -1,
        )
    )
    return out


def _sort_subset_performance(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    sort_cols: List[str] = []
    if "permutation_idx" in df.columns:
        sort_cols.append("permutation_idx")
    if "h_hit" in df.columns:
        sort_cols.append("h_hit")
    if "subset_idx" in df.columns:
        sort_cols.append("subset_idx")
    if sort_cols:
        df = df.sort_values(sort_cols).reset_index(drop=True)
    return df


def _safe_relpath(path: Optional[str], start: str) -> Optional[str]:
    if not path:
        return None
    try:
        return os.path.relpath(str(path), start=str(start))
    except Exception:
        return str(path)


def _load_task_manifest(analysis_folder_path: str) -> List[Dict[str, Any]]:
    """Load typed real/permutation assignments from the frozen task manifest."""
    from .plan import PLAN_DIRNAME, TASK_MANIFEST_FILENAME

    task_manifest_path = os.path.join(analysis_folder_path, PLAN_DIRNAME, TASK_MANIFEST_FILENAME)
    if not os.path.exists(task_manifest_path):
        raise FileNotFoundError(f"Planned task manifest not found: {task_manifest_path}")
    df = pd.read_csv(task_manifest_path)
    if df.empty:
        return []
    records: List[Dict[str, Any]] = []
    for rec in df.to_dict(orient="records"):
        record = dict(rec)
        if pd.isna(record.get("permutation_idx")):
            record["permutation_idx"] = None
        else:
            record["permutation_idx"] = int(record["permutation_idx"])
        for key in ("task_id", "shard_index", "h_hit", "subset_idx", "n_features", "n_splits", "planned_model_count"):
            record[key] = int(record.get(key, 0) or 0)
        for key in ("family", "task_key", "shard_label"):
            record[key] = str(record.get(key, ""))
        records.append(record)
    return records


def _expected_shard_label(active_num_shards: int, shard_index: int) -> str:
    width = max(3, len(str(int(active_num_shards) - 1)))
    return f"shard_{int(shard_index):0{width}d}_of_{int(active_num_shards):0{width}d}"


def _discover_completed_shards(
    analysis_folder_path: str,
    manifest: Mapping[str, Any],
) -> Tuple[List[Tuple[int, str, str, Dict[str, Any]]], List[str]]:
    """Verify exact planned task coverage and collect completed shard artifacts."""
    active_num_shards = int(manifest.get("active_num_shards", 1) or 1)
    _validate_planned_artifact_hashes(analysis_folder_path, manifest)
    task_rows = _load_task_manifest(analysis_folder_path)
    planned_task_count = int(len(task_rows))

    done_entries: List[Tuple[int, str, str, Dict[str, Any]]] = []
    errors: List[str] = []
    observed_task_count = 0

    for shard_index in range(active_num_shards):
        shard_label = _expected_shard_label(active_num_shards, shard_index)
        shard_dir = os.path.join(analysis_folder_path, SHARD_OUTPUT_DIRNAME, shard_label)
        done_path = os.path.join(shard_dir, SHARD_DONE_FILENAME)
        failed_path = os.path.join(shard_dir, SHARD_FAILED_FILENAME)
        running_path = os.path.join(shard_dir, SHARD_RUNNING_FILENAME)
        lock_dir = os.path.join(shard_dir, SHARD_LOCK_DIRNAME)

        from autobehaver.file_locks import file_lock_active
        if os.path.isdir(lock_dir) or file_lock_active(lock_dir):
            errors.append(f"Shard {shard_label} still appears to be running (running file or lock present).")
            continue
        if os.path.exists(failed_path):
            errors.append(f"Shard {shard_label} has a failure marker: {failed_path}")
            continue
        if not os.path.exists(done_path):
            errors.append(f"Shard {shard_label} has no completion marker: {done_path}")
            continue

        done = _json_load(done_path)
        from .plan import PLAN_DIRNAME, PLAN_MANIFEST_FILENAME
        valid, reason = _validate_completed_shard(
            done_path, analysis_folder_path=analysis_folder_path,
            plan_manifest_path=os.path.join(analysis_folder_path, PLAN_DIRNAME, PLAN_MANIFEST_FILENAME),
        )
        if not valid:
            errors.append(f"Shard {shard_label} failed artifact validation: {reason}")
            continue
        if not isinstance(done, dict) or str(done.get("status", "")).strip().lower() != "completed":
            errors.append(f"Shard {shard_label} completion marker is not valid: {done_path}")
            continue

        planned_tasks_for_shard = [r for r in task_rows if int(r.get("shard_index", -1)) == shard_index]
        if int(done.get("task_count", -1)) != len(planned_tasks_for_shard):
            errors.append(
                f"Shard {shard_label} task_count mismatch: done={done.get('task_count')} expected={len(planned_tasks_for_shard)}"
            )
            continue
        try:
            from .integrity import safe_artifact
            for family, artifact in [('real', 'subset_performance_real_csv'), ('perm', 'subset_performance_perm_csv')]:
                frame = pd.read_csv(safe_artifact(analysis_folder_path, done['artifacts'][artifact]))
                expected = {(int(r['h_hit']), int(r['subset_idx']), None if family == 'real' else int(r['permutation_idx'])) for r in planned_tasks_for_shard if r['family'] == family}
                observed = [(int(r['h_hit']), int(r['subset_idx']), None if family == 'real' else int(r['permutation_idx'])) for r in frame.to_dict('records')]
                if len(observed) != len(set(observed)) or set(observed) != expected:
                    raise ValueError(f'{family} subset identities differ from the exact planned tasks')
                perf_artifact = 'performance_records_real_jsonl' if family == 'real' else 'performance_records_perm_jsonl'
                performance = _jsonl_load(str(safe_artifact(analysis_folder_path, done['artifacts'][perf_artifact])))
                observed_folds = [(int(r['h_hit']), int(r['subset_idx']), None if family == 'real' else int(r['permutation_idx']), int(r['fold_idx'])) for r in performance]
                expected_folds = {(*task, fold) for task in expected for fold in range(int(manifest['n_splits']))}
                if len(observed_folds) != len(set(observed_folds)) or set(observed_folds) != expected_folds:
                    raise ValueError(f'{family} fold identities differ from the exact planned models')
        except Exception as exc:
            errors.append(f'Shard {shard_label} semantic inventory validation failed: {exc}')
            continue
        observed_task_count += len(planned_tasks_for_shard)
        done_entries.append((shard_index, shard_label, shard_dir, done))

    if observed_task_count != planned_task_count:
        errors.append(
            f"Observed planned-task coverage across completed shards ({observed_task_count}) does not match task manifest ({planned_task_count})."
        )

    done_entries.sort(key=lambda x: x[0])
    return done_entries, errors


def _validate_merged_counts(
    *,
    manifest: Mapping[str, Any],
    real_subset_df: pd.DataFrame,
    perm_subset_df: pd.DataFrame,
    real_perf_rows: Sequence[Mapping[str, Any]],
    perm_perf_rows: Sequence[Mapping[str, Any]],
) -> None:
    """Reject missing or duplicate subset and fold records after merging."""
    planned_family_counts = dict(manifest.get("planned_family_counts") or {})
    expected_real_tasks = int(planned_family_counts.get("real", 0) or 0)
    expected_perm_tasks = int(planned_family_counts.get("perm", 0) or 0)
    n_splits = int(manifest.get("n_splits", 0) or 0)

    if len(real_subset_df) != expected_real_tasks:
        raise RuntimeError(
            f"Merged real subset rows ({len(real_subset_df)}) do not match planned real tasks ({expected_real_tasks})."
        )
    if len(perm_subset_df) != expected_perm_tasks:
        raise RuntimeError(
            f"Merged permuted subset rows ({len(perm_subset_df)}) do not match planned permuted tasks ({expected_perm_tasks})."
        )
    if len(real_perf_rows) != expected_real_tasks * n_splits:
        raise RuntimeError(
            "Merged real performance-record count does not match planned real models: "
            f"observed={len(real_perf_rows)}, expected={expected_real_tasks * n_splits}."
        )
    if len(perm_perf_rows) != expected_perm_tasks * n_splits:
        raise RuntimeError(
            "Merged permuted performance-record count does not match planned permuted models: "
            f"observed={len(perm_perf_rows)}, expected={expected_perm_tasks * n_splits}."
        )

    if not real_subset_df.empty:
        dup = real_subset_df.duplicated(subset=["h_hit", "subset_idx"]).sum()
        if int(dup) > 0:
            raise RuntimeError(f"Merged real subset-performance table contains {int(dup)} duplicate (h_hit, subset_idx) rows.")
    if not perm_subset_df.empty:
        dup = perm_subset_df.duplicated(subset=["permutation_idx", "h_hit", "subset_idx"]).sum()
        if int(dup) > 0:
            raise RuntimeError(
                f"Merged permuted subset-performance table contains {int(dup)} duplicate (permutation_idx, h_hit, subset_idx) rows."
            )


def _finalize_partial_averages(*, partial: Mapping[str, Any], n_splits: int) -> Dict[str, np.ndarray | int]:
    """Normalize merged SHAP and prediction sums by their contributing counts."""
    insample_probas_sum = np.asarray(partial["insample_probas_sum"], dtype=np.float32)
    oof_probas_sum = np.asarray(partial["oof_probas_sum"], dtype=np.float32)
    insample_shap_sum = np.asarray(partial["insample_shap_sum"], dtype=np.float32)
    oof_shap_sum = np.asarray(partial["oof_shap_sum"], dtype=np.float32)
    train_pred_count = np.asarray(partial["train_pred_count"], dtype=np.int32)
    test_pred_count = np.asarray(partial["test_pred_count"], dtype=np.int32)
    all_groups_probas_sum = np.asarray(partial["all_groups_probas_sum"], dtype=np.float32)
    all_groups_probas_count = int(np.asarray(partial["all_groups_probas_count"]).reshape(()))
    all_groups_shap_sum = np.asarray(partial["all_groups_shap_sum"], dtype=np.float32)
    feature_subset_hit_counts = np.asarray(partial["feature_subset_hit_counts"], dtype=np.int32)
    feature_model_hit_counts = np.asarray(partial["feature_model_hit_counts"], dtype=np.int32)

    safe_train_div = train_pred_count.copy(); safe_train_div[safe_train_div == 0] = 1
    safe_test_div = test_pred_count.copy(); safe_test_div[safe_test_div == 0] = 1
    safe_subset_hits = feature_subset_hit_counts.astype(np.float32); safe_subset_hits[safe_subset_hits == 0] = 1.0
    safe_model_hits = feature_model_hit_counts.astype(np.float32); safe_model_hits[safe_model_hits == 0] = 1.0

    insample_probas = insample_probas_sum / safe_train_div[:, None]
    oof_probas = oof_probas_sum / safe_test_div[:, None]
    all_groups_probas = all_groups_probas_sum / float(all_groups_probas_count) if all_groups_probas_count > 0 else all_groups_probas_sum

    oof_shap = oof_shap_sum / safe_subset_hits[None, :]
    insample_shap = insample_shap_sum / (safe_subset_hits * max(int(n_splits) - 1, 1))[None, :]
    all_groups_shap = all_groups_shap_sum / safe_model_hits[None, :]

    return {
        "insample_probas": insample_probas.astype(np.float32),
        "oof_probas": oof_probas.astype(np.float32),
        "all_groups_probas": all_groups_probas.astype(np.float32),
        "insample_shap": insample_shap.astype(np.float32),
        "oof_shap": oof_shap.astype(np.float32),
        "all_groups_shap": all_groups_shap.astype(np.float32),
        "train_pred_count": train_pred_count,
        "test_pred_count": test_pred_count,
        "all_groups_probas_count": int(all_groups_probas_count),
        "feature_subset_hit_counts": feature_subset_hit_counts,
        "feature_model_hit_counts": feature_model_hit_counts,
    }


def _write_memmap(path: str, array: np.ndarray) -> str:
    arr = np.asarray(array, dtype=np.float32)
    mm = np.memmap(path, dtype=np.float32, mode="w+", shape=arr.shape)
    mm[:] = arr[:]
    del mm
    return path


def _build_hybrid_shap_views(
    *,
    paths: Mapping[str, str],
    n_train: int,
    oof_shap_values: np.ndarray,
    all_groups_avg_shap_values: np.ndarray,
    oof_shap_values_unweighted: Optional[np.ndarray] = None,
    all_groups_avg_shap_values_unweighted: Optional[np.ndarray] = None,
    weighted_feature_usage_fraction: Optional[np.ndarray] = None,
) -> Tuple[List[str], Optional[Dict[str, str]], np.ndarray, Optional[Dict[str, np.ndarray]]]:
    """Construct all-group SHAP views with out-of-fold values in training rows."""
    n_all, n_features = all_groups_avg_shap_values.shape
    hybrid = np.zeros((n_all, n_features), dtype=np.float32)
    hybrid[:n_train, :] = np.asarray(oof_shap_values, dtype=np.float32)
    if n_all > n_train:
        hybrid[n_train:, :] = np.asarray(all_groups_avg_shap_values[n_train:, :], dtype=np.float32)
    shap_folder = str(paths["shap_values_folder"])
    os.makedirs(shap_folder, exist_ok=True)
    primary_path = os.path.join(shap_folder, "all_groups_avg_shap_values.mmap")
    _write_memmap(primary_path, hybrid)
    extra_views: Dict[str, str] = {}
    extra_arrays: Dict[str, np.ndarray] = {}
    if oof_shap_values_unweighted is not None and all_groups_avg_shap_values_unweighted is not None:
        hybrid_u = np.zeros((n_all, n_features), dtype=np.float32)
        hybrid_u[:n_train, :] = np.asarray(oof_shap_values_unweighted, dtype=np.float32)
        if n_all > n_train:
            hybrid_u[n_train:, :] = np.asarray(all_groups_avg_shap_values_unweighted[n_train:, :], dtype=np.float32)
        extra_views["UnweightedEquitable"] = _write_memmap(
            os.path.join(shap_folder, "all_groups_avg_shap_values_UNWEIGHTED.mmap"), hybrid_u,
        )
        extra_arrays["UnweightedEquitable"] = hybrid_u
    return [primary_path], (extra_views or None), hybrid, (extra_arrays or None)


def _build_predictions_export(
    *,
    df_for_training: pd.DataFrame,
    df_for_shap_and_agg: pd.DataFrame,
    training_insample: np.ndarray,
    training_oof: np.ndarray,
    all_groups_probas: np.ndarray,
    le: FixedBinaryLabelEncoder,
    export_label: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Assemble animal predictions with out-of-fold training values and group
    separators.
    """
    num_classes = len(le.classes_)
    prob_cols = [f"Prob_{le.classes_[c]}" for c in range(num_classes)]
    base_info_df = df_for_training[["individual", "group"]].reset_index(drop=True)
    training_predictions = pd.concat([
        base_info_df,
        pd.DataFrame(training_insample, columns=[f"{col}_insample" for col in prob_cols]),
        pd.DataFrame(training_oof, columns=[f"{col}_outofsample" for col in prob_cols]),
        pd.DataFrame(training_oof, columns=prob_cols),
    ], axis=1)

    all_groups_probas_df = pd.DataFrame(all_groups_probas, columns=prob_cols)
    all_groups_base_info = df_for_shap_and_agg[["individual", "group"]].reset_index(drop=True)
    training_individuals = set(df_for_training["individual"].values)
    non_training_mask = ~all_groups_base_info["individual"].isin(training_individuals)
    non_training_mask_array = non_training_mask.to_numpy(dtype=bool, copy=False)
    non_training_base_info = all_groups_base_info.loc[non_training_mask_array].reset_index(drop=True)
    expected_all_rows = len(all_groups_base_info)
    expected_non_training_rows = len(non_training_base_info)
    observed_all_rows = len(all_groups_probas_df)
    if observed_all_rows == expected_all_rows:
        non_training_probas = all_groups_probas_df.loc[non_training_mask_array].reset_index(drop=True)
    elif observed_all_rows == expected_non_training_rows:
        non_training_probas = all_groups_probas_df.reset_index(drop=True)
    else:
        if observed_all_rows not in (0,):
            print(
                f"WARNING: {export_label} all-groups prediction matrix has {observed_all_rows} rows; "
                f"expected either {expected_all_rows} (all groups) or {expected_non_training_rows} (non-training only). "
                "Filling non-training predictions with NaN."
            )
        non_training_probas = pd.DataFrame(np.nan, index=range(expected_non_training_rows), columns=prob_cols)

    non_training_predictions = pd.concat([
        non_training_base_info,
        pd.DataFrame(np.nan, index=non_training_probas.index, columns=[f"{col}_insample" for col in prob_cols]),
        pd.DataFrame(np.nan, index=non_training_probas.index, columns=[f"{col}_outofsample" for col in prob_cols]),
        non_training_probas,
    ], axis=1)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=FutureWarning)
        all_predictions = pd.concat([training_predictions, non_training_predictions], axis=0, ignore_index=True)
    all_predictions = all_predictions.sort_values("group").reset_index(drop=True)
    all_predictions = insert_group_separator_rows(all_predictions)
    return training_predictions, non_training_predictions, all_predictions


def _compute_binary_metrics_from_prob_matrix(y_true: np.ndarray, probas: np.ndarray) -> Dict[str, float]:
    """Calculate binary metrics from class-0/class-1 probability columns."""
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, brier_score_loss, f1_score, log_loss, roc_auc_score

    y = np.asarray(y_true, dtype=np.int32)
    arr = np.asarray(probas, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(f"Expected probability matrix of shape (n_samples, 2), got {arr.shape!r}.")
    prob1 = np.clip(arr[:, 1], 1e-15, 1.0 - 1e-15)
    preds = (prob1 >= 0.5).astype(np.int32)
    two_col = np.column_stack([1.0 - prob1, prob1])
    auc_val = float('nan')
    try:
        if len(np.unique(y)) == 2:
            auc_val = float(roc_auc_score(y, prob1))
    except Exception:
        auc_val = float('nan')
    return {
        "accuracy": float(accuracy_score(y, preds)),
        "balanced_accuracy": float(balanced_accuracy_score(y, preds)),
        "f1": float(f1_score(y, preds, zero_division=0)),
        "auc": auc_val,
        "log_loss": float(log_loss(y, two_col, labels=[0, 1])),
        "brier": float(brier_score_loss(y, prob1)),
        "mean_predicted_probability_group_b": float(np.mean(prob1)),
    }


def _write_committee_pooled_performance(
    *,
    analysis_folder: str,
    y_true: np.ndarray,
    insample_probas: np.ndarray,
    oof_probas: np.ndarray,
    H_hits_per_feature: int,
    aggregated_performance_results: Sequence[Mapping[str, Any]],
) -> str:
    """Write pooled training and out-of-fold committee metrics."""
    unique_specialists = set()
    for perf in aggregated_performance_results:
        if bool(perf.get("is_permuted", False)):
            continue
        h_hit = perf.get("h_hit")
        subset_idx = perf.get("subset_idx")
        if h_hit is not None and subset_idx is not None:
            unique_specialists.add((int(h_hit), int(subset_idx)))
    ins_metrics = _compute_binary_metrics_from_prob_matrix(np.asarray(y_true, dtype=np.int32), np.asarray(insample_probas, dtype=np.float64))
    oof_metrics = _compute_binary_metrics_from_prob_matrix(np.asarray(y_true, dtype=np.int32), np.asarray(oof_probas, dtype=np.float64))
    df = pd.DataFrame([{
        "n_h_hits": int(H_hits_per_feature),
        "n_specialists": int(len(unique_specialists)),
        "n_real_models": int(len([r for r in aggregated_performance_results if not bool(r.get('is_permuted', False))])),
        **{f"oof_{k}": v for k, v in oof_metrics.items()},
        **{f"insample_{k}": v for k, v in ins_metrics.items()},
    }])
    out_path = os.path.join(analysis_folder, COMMITTEE_POOLED_PERFORMANCE_FILENAME)
    df.to_csv(out_path, index=False)
    return out_path


def _write_ensemble_metrics(
    *,
    analysis_folder: str,
    y_true: np.ndarray,
    insample_probas: np.ndarray,
    oof_probas: np.ndarray,
    num_classes: int,
    le: FixedBinaryLabelEncoder,
    H_hits_per_feature: int,
    aggregated_performance_results: Sequence[Mapping[str, Any]],
) -> None:
    """Write binary ensemble metrics and specialist-count metadata."""
    unique_specialists = set()
    for perf in aggregated_performance_results:
        if bool(perf.get("is_permuted", False)):
            continue
        h_hit = perf.get("h_hit")
        subset_idx = perf.get("subset_idx")
        if h_hit is not None and subset_idx is not None:
            unique_specialists.add((int(h_hit), int(subset_idx)))
    rounds_seen = {hh for (hh, _) in unique_specialists}
    avg_num_specialists = float(len(unique_specialists)) / float(len(rounds_seen)) if len(rounds_seen) > 0 else 0.0
    metrics_dict = calculate_ensemble_metrics(
        y_true=y_true,
        insample_probas=insample_probas,
        oof_probas=oof_probas,
        num_classes=num_classes,
        le=le,
    )
    metrics_summary = pd.DataFrame({
        "Metric": ["F1 Score", "AUC"],
        "In-Sample (Training)": [f"{metrics_dict['f1_insample']:.4f}", f"{metrics_dict['auc_insample']:.4f}"],
        "Out-of-Sample (Validation)": [f"{metrics_dict['f1_outofsample']:.4f}", f"{metrics_dict['auc_outofsample']:.4f}"],
    })
    metadata_rows = pd.DataFrame({
        "Metric": ["Number of Classes", "Target H-Hits per Feature", "Average Number of Specialists in Ensemble"],
        "In-Sample (Training)": [str(num_classes), str(H_hits_per_feature), f"{avg_num_specialists:.1f}"],
        "Out-of-Sample (Validation)": ["", "", ""],
    })
    out_path = os.path.join(analysis_folder, "ENSEMBLE_PERFORMANCE_METRICS.xlsx")
    with pd.ExcelWriter(out_path) as writer:
        metrics_summary.to_excel(writer, sheet_name="Metrics", index=False)
        metadata_rows.to_excel(writer, sheet_name="Metadata", index=False)
