"""Process frozen FeatureImp subset tasks and write shard-local numerical results and
completion receipts.

Final reports are assembled by the merge stage.
"""

from __future__ import annotations

import csv
import json
import os
import shutil
import traceback
from contextlib import contextmanager, ExitStack
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
from sklearn.preprocessing import LabelEncoder

from autobehaver.atomic_io import sha256_file
from autobehaver.atomic_io import atomic_write_json
from .integrity import seal_receipt, validate_receipt, flatten_paths

from .committee import FixedBinaryLabelEncoder
from .config import DEFAULT_SETTINGS
from .committee_fast import _init_fast_pool_worker, _train_subset_worker
from .hhit import make_hhit_folds, validate_stratified_cv_labels
from .matrix import compute_feature_schema_sha256
from .mp_utils import iter_parallel_results
from .bounded_pool import BoundedProcessRunner
from .archive_io import TaskArchiveWriter
from autobehaver.scratch import scratch_workspace
from autobehaver.completion import verified_run
from .committee_fast import _clear_fast_pool_worker
from autobehaver.resource_monitor import record_work
from .permutation import _generate_sex_class_stratified_permutation
from .plan import (
    ALL_GROUPS_ROW_INDICES_FILENAME,
    HHIT_FOLD_PLAN_FILENAME,
    HHIT_SUBSET_PLAN_FILENAME,
    OTHER_GROUP_ROW_INDICES_FILENAME,
    PERMUTATION_SEEDS_FILENAME,
    PLAN_DIRNAME,
    SELECTED_FEATURE_COLUMNS_FILENAME,
    SHARD_PLAN_DIRNAME,
    TRAINING_ROW_INDICES_FILENAME,
    load_featureimp_plan_manifest,
    validate_hhit_fold_plan,
)
from .margin_tasks import (
    PERM_MARGIN_TASKS_FILENAME_TEMPLATE,
    REAL_MARGIN_TASKS_FILENAME,
    save_margin_task_archive,
)
from .shap_tasks import (
    PERM_SHAP_TASKS_FILENAME_TEMPLATE,
    REAL_SHAP_TASKS_FILENAME,
    save_shap_task_archive,
)


SHARD_OUTPUT_DIRNAME = "_featureimp_shards"
SHARD_RUNNING_FILENAME = "SHARD_RUNNING.json"
SHARD_DONE_FILENAME = "SHARD_DONE.json"
SHARD_FAILED_FILENAME = "SHARD_FAILED.json"
SHARD_ERRORS_FILENAME = "SHARD_ERRORS.jsonl"
REAL_PARTIALS_FILENAME = "REAL_PARTIALS.npz"
PERM_PARTIALS_FILENAME_TEMPLATE = "PERM_PARTIALS_perm{perm_idx:03d}.npz"
PERFORMANCE_RECORDS_REAL_FILENAME = "PERFORMANCE_RECORDS_REAL.jsonl"
PERFORMANCE_RECORDS_PERM_FILENAME = "PERFORMANCE_RECORDS_PERM.jsonl"
SUBSET_PERFORMANCE_REAL_FILENAME = "SUBSET_PERFORMANCE_REAL_PART.csv"
SUBSET_PERFORMANCE_PERM_FILENAME = "SUBSET_PERFORMANCE_PERM_PART.csv"
LOCK_DIRNAME = "_featureimp_shard.lock"


def _json_load(path: str) -> Any:
    with open(path, "r") as f:
        return json.load(f)


def _json_dump(obj: Any, path: str, *, verified_inputs=None) -> None:
    if os.path.basename(path) in {"SHARD_DONE.json", "WEIGHTS_DONE.json", "FINAL_MERGE_DONE.json", "WEIGHTED_REDUCTION_DONE.json"}:
        obj = seal_receipt(obj, path, verified_inputs=verified_inputs)
    atomic_write_json(_to_builtin(obj), path)


def _jsonl_load(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not os.path.exists(path):
        return rows
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _to_builtin(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return [_to_builtin(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _to_builtin(v) for k, v in value.items()}
    return value


def _write_jsonl(rows: Iterable[Dict[str, Any]], path: str) -> None:
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(_to_builtin(row), allow_nan=True))
            f.write("\n")


def _read_csv_rows(path: str) -> List[Dict[str, str]]:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _resolve_analysis_artifact(analysis_folder_path: str, manifest: Mapping[str, Any], *, artifact_key: str, fallback_filename: str) -> str:
    """Resolve a planned artifact inside the analysis folder."""
    artifacts = manifest.get("artifacts", {}) if isinstance(manifest, Mapping) else {}
    rel = artifacts.get(artifact_key)
    if rel:
        relative = Path(str(rel))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"Unsafe FeatureImp artifact path for {artifact_key}: {rel!r}")
        base = Path(analysis_folder_path).resolve()
        candidate = (base / relative).resolve()
        if not candidate.is_relative_to(base):
            raise ValueError(f"FeatureImp artifact escapes the analysis folder: {rel!r}")
        return str(candidate)
    return os.path.join(analysis_folder_path, PLAN_DIRNAME, fallback_filename)


def _validate_planned_artifact_hashes(
    analysis_folder_path: str,
    manifest: Mapping[str, Any],
) -> None:
    """Verify every frozen plan artifact before loading shard context."""
    expected_hashes = manifest.get("artifact_sha256")
    if not isinstance(expected_hashes, Mapping) or not expected_hashes:
        raise ValueError("FeatureImp plan is missing required artifact_sha256 fingerprints.")
    for artifact_key, expected_hash in expected_hashes.items():
        path = _resolve_analysis_artifact(
            analysis_folder_path,
            manifest,
            artifact_key=str(artifact_key),
            fallback_filename=str(artifact_key),
        )
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Frozen FeatureImp plan artifact is missing: {path}")
        if sha256_file(path) != str(expected_hash):
            raise ValueError(
                f"Frozen FeatureImp plan artifact changed after planning: {artifact_key}"
            )


def _flatten_artifact_paths(value: Any) -> List[str]:
    if isinstance(value, Mapping):
        paths: List[str] = []
        for nested in value.values():
            paths.extend(_flatten_artifact_paths(nested))
        return paths
    if isinstance(value, (list, tuple)):
        paths = []
        for nested in value:
            paths.extend(_flatten_artifact_paths(nested))
        return paths
    return [str(value)] if value else []


def _validate_completed_shard(
    done_path: str,
    *,
    analysis_folder_path: str,
    plan_manifest_path: str,
) -> tuple[bool, str]:
    """Verify a shard receipt and its declared artifacts against the frozen plan."""
    try:
        summary = _json_load(done_path)
        validate_receipt(done_path, analysis_folder_path)
        if str(summary.get("status", "")).lower() != "completed":
            return False, "completion marker status is not completed"
        if summary.get("plan_manifest_sha256") != sha256_file(plan_manifest_path):
            return False, "FeatureImp plan changed after this shard completed"
        expected_hashes = summary.get("artifact_sha256")
        if not isinstance(expected_hashes, Mapping) or not expected_hashes:
            return False, "completion marker has no artifact fingerprints"
        if set(flatten_paths(summary.get("artifacts", {}))) != set(expected_hashes):
            return False, "completion marker does not fingerprint every required artifact"
        base = Path(analysis_folder_path).resolve()
        for relative, expected_hash in expected_hashes.items():
            relative_path = Path(str(relative))
            if relative_path.is_absolute() or ".." in relative_path.parts:
                return False, f"unsafe shard artifact path: {relative!r}"
            artifact = (base / relative_path).resolve()
            if not artifact.is_relative_to(base) or not artifact.is_file():
                return False, f"shard artifact is missing: {relative}"
            # validate_receipt already checked every required artifact hash.
        return True, "ok"
    except Exception as exc:
        return False, f"could not validate completion marker: {exc}"


def _load_settings_snapshot(analysis_folder_path: str) -> Dict[str, Any]:
    path = os.path.join(analysis_folder_path, "FEATUREIMP_RUN_SETTINGS.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Required frozen FeatureImp settings not found: {path}")
    try:
        data = _json_load(path)
        from .core import filter_saved_settings_snapshot

        if not isinstance(data, dict):
            raise ValueError("Saved FeatureImp settings must be a JSON object")
        return filter_saved_settings_snapshot(data)
    except Exception as exc:
        raise ValueError(f"Saved FeatureImp settings are invalid: {path}") from exc


def _get_runtime_num_workers() -> int:
    from . import core as _core

    current_settings = dict(_core.get_current_settings() or {})
    raw = current_settings.get("num_workers", DEFAULT_SETTINGS.get("num_workers", 1))
    try:
        return max(1, int(raw) if raw is not None else int(DEFAULT_SETTINGS.get("num_workers", 1) or 1))
    except Exception:
        return max(1, int(DEFAULT_SETTINGS.get("num_workers", 1) or 1))


def _load_shard_plan_and_tasks(analysis_folder_path: str, *, shard_index: int) -> Tuple[Dict[str, Any], List[Dict[str, Any]], str]:
    """Load and validate one shard's assignments and task identities."""
    manifest = load_featureimp_plan_manifest(analysis_folder_path)
    active_num_shards = int(manifest.get("active_num_shards", 1) or 1)
    if shard_index < 0 or shard_index >= active_num_shards:
        raise ValueError(
            f"shard_index={shard_index} is out of range for active_num_shards={active_num_shards}."
        )
    width = max(3, len(str(active_num_shards - 1)))
    shard_label = f"shard_{shard_index:0{width}d}_of_{active_num_shards:0{width}d}"
    shard_dir = os.path.join(analysis_folder_path, PLAN_DIRNAME, SHARD_PLAN_DIRNAME, shard_label)
    shard_plan_path = os.path.join(shard_dir, "SHARD_PLAN.json")
    shard_tasks_path = os.path.join(shard_dir, "SHARD_TASKS.csv")
    if not os.path.exists(shard_plan_path):
        raise FileNotFoundError(f"Planned shard manifest not found: {shard_plan_path}")
    if not os.path.exists(shard_tasks_path):
        raise FileNotFoundError(f"Planned shard task CSV not found: {shard_tasks_path}")
    shard_hashes = manifest.get("shard_artifact_sha256")
    if not isinstance(shard_hashes, Mapping):
        raise ValueError("FeatureImp plan is missing shard_artifact_sha256 fingerprints.")
    for path in (shard_plan_path, shard_tasks_path):
        relative = Path(os.path.relpath(path, analysis_folder_path)).as_posix()
        expected_hash = shard_hashes.get(relative) or shard_hashes.get(os.path.relpath(path, analysis_folder_path))
        if not expected_hash:
            raise ValueError(f"FeatureImp plan has no fingerprint for shard artifact: {relative}")
        if sha256_file(path) != str(expected_hash):
            raise ValueError(f"FeatureImp shard plan artifact changed after planning: {relative}")
    shard_plan = _json_load(shard_plan_path)
    rows_raw = _read_csv_rows(shard_tasks_path)
    task_rows: List[Dict[str, Any]] = []
    for row in rows_raw:
        task_rows.append(
            {
                "task_id": int(row.get("task_id", 0) or 0),
                "task_key": str(row.get("task_key", "")),
                "shard_index": int(row.get("shard_index", 0) or 0),
                "shard_label": str(row.get("shard_label", shard_label)),
                "family": str(row.get("family", "")),
                "permutation_idx": None
                if str(row.get("permutation_idx", "")).strip() == ""
                else int(row.get("permutation_idx", 0)),
                "h_hit": int(row.get("h_hit", 0) or 0),
                "subset_idx": int(row.get("subset_idx", 0) or 0),
                "n_features": int(row.get("n_features", 0) or 0),
                "n_splits": int(row.get("n_splits", 0) or 0),
                "planned_model_count": int(row.get("planned_model_count", 0) or 0),
            }
        )
    if str(shard_plan.get("status", "")).lower() != "planned":
        raise ValueError(f"Shard plan status is not planned: {shard_plan.get('status')!r}")
    if int(shard_plan.get("shard_index", -1)) != int(shard_index):
        raise ValueError("Shard plan index does not match the requested shard.")
    if int(shard_plan.get("task_count", -1)) != len(task_rows):
        raise ValueError("Shard plan task_count does not match SHARD_TASKS.csv.")
    task_keys = [str(row.get("task_key") or "") for row in task_rows]
    if any(not key for key in task_keys) or len(task_keys) != len(set(task_keys)):
        raise ValueError("Shard task keys must be non-blank and unique.")
    if any(int(row["shard_index"]) != int(shard_index) for row in task_rows):
        raise ValueError("SHARD_TASKS.csv contains a task assigned to another shard.")
    return shard_plan, task_rows, shard_label


@contextmanager
def _directory_lock(lock_dir: str):
    """Hold process ownership; preserve unknown legacy directory locks."""
    from autobehaver.file_locks import FileLock
    os.makedirs(os.path.dirname(lock_dir), exist_ok=True)
    if os.path.isdir(lock_dir):
        raise RuntimeError(f"Legacy FeatureImp directory lock requires verified recovery: {lock_dir}")
    lock = FileLock(lock_dir)
    lock.acquire(metadata={"stage": "featureimp", "lock_path": os.path.abspath(lock_dir)})
    try:
        yield
    finally:
        lock.release()


def _load_featureimp_context(analysis_folder_path: str, manifest: Mapping[str, Any]) -> Dict[str, Any]:
    """Restore frozen sample and feature order and prepare validated model inputs."""
    _validate_planned_artifact_hashes(analysis_folder_path, manifest)
    # The frozen relative artifact is authoritative after moving a local plan to HPC.
    normalized_matrix_path = _resolve_analysis_artifact(
        analysis_folder_path, manifest, artifact_key="normalized_matrix_csv",
        fallback_filename="NORMALIZED_FEATURE_MATRIX.csv")
    if not os.path.exists(normalized_matrix_path):
        raise FileNotFoundError(f"Normalized feature matrix not found: {normalized_matrix_path}")
    expected_matrix_hash = str(manifest.get("normalized_matrix_sha256") or "")
    if not expected_matrix_hash:
        raise ValueError("FeatureImp plan is missing normalized_matrix_sha256.")
    if sha256_file(normalized_matrix_path) != expected_matrix_hash:
        raise ValueError("Normalized feature matrix changed after the FeatureImp plan was created.")

    from .matrix_input import read_feature_matrix
    df_raw = read_feature_matrix(normalized_matrix_path)

    selected_features_path = _resolve_analysis_artifact(
        analysis_folder_path,
        manifest,
        artifact_key="selected_feature_columns_json",
        fallback_filename=SELECTED_FEATURE_COLUMNS_FILENAME,
    )
    training_rows_path = _resolve_analysis_artifact(
        analysis_folder_path,
        manifest,
        artifact_key="training_row_indices_json",
        fallback_filename=TRAINING_ROW_INDICES_FILENAME,
    )
    all_groups_rows_path = _resolve_analysis_artifact(
        analysis_folder_path,
        manifest,
        artifact_key="all_groups_row_indices_json",
        fallback_filename=ALL_GROUPS_ROW_INDICES_FILENAME,
    )
    other_rows_path = _resolve_analysis_artifact(
        analysis_folder_path,
        manifest,
        artifact_key="other_group_row_indices_json",
        fallback_filename=OTHER_GROUP_ROW_INDICES_FILENAME,
    )
    fold_plan_path = _resolve_analysis_artifact(
        analysis_folder_path,
        manifest,
        artifact_key="hhit_fold_plan_jsonl",
        fallback_filename=HHIT_FOLD_PLAN_FILENAME,
    )
    subset_plan_path = _resolve_analysis_artifact(
        analysis_folder_path,
        manifest,
        artifact_key="hhit_subset_plan_jsonl",
        fallback_filename=HHIT_SUBSET_PLAN_FILENAME,
    )
    permutation_seeds_path = _resolve_analysis_artifact(
        analysis_folder_path,
        manifest,
        artifact_key="permutation_seeds_csv",
        fallback_filename=PERMUTATION_SEEDS_FILENAME,
    )

    selected_features = [str(c) for c in _json_load(selected_features_path)]
    feature_schema = compute_feature_schema_sha256(selected_features)
    expected_schema = str(manifest.get("selected_feature_schema_sha256") or "")
    if expected_schema and feature_schema != expected_schema:
        raise ValueError(
            "Selected feature schema mismatch between plan manifest and selected feature list. "
            f"Expected {expected_schema}, observed {feature_schema}."
        )

    training_row_indices = [int(v) for v in _json_load(training_rows_path)]
    all_groups_row_indices = [int(v) for v in _json_load(all_groups_rows_path)]
    other_group_row_indices = [int(v) for v in _json_load(other_rows_path)]

    if len(training_row_indices) != int(manifest.get("training_row_count", len(training_row_indices))):
        raise ValueError("Training row-count mismatch between plan manifest and training-row index file.")
    if len(all_groups_row_indices) != int(manifest.get("all_groups_row_count", len(all_groups_row_indices))):
        raise ValueError("All-groups row-count mismatch between plan manifest and all-groups row index file.")
    if len(training_row_indices) != len(set(training_row_indices)):
        raise ValueError("Training row-index plan contains duplicate rows.")
    if len(other_group_row_indices) != len(set(other_group_row_indices)):
        raise ValueError("Other-group row-index plan contains duplicate rows.")
    if set(training_row_indices) & set(other_group_row_indices):
        raise ValueError("Training and other-group row-index plans overlap.")
    if all_groups_row_indices != training_row_indices + other_group_row_indices:
        raise ValueError("All-groups row order does not equal the frozen training + other row order.")
    invalid_indices = [
        index for index in all_groups_row_indices if index < 0 or index >= len(df_raw)
    ]
    if invalid_indices:
        raise ValueError(f"Frozen row indices fall outside the normalized matrix: {invalid_indices[:20]}")
    missing_features = [feature for feature in selected_features if feature not in df_raw.columns]
    if missing_features:
        raise ValueError(f"Selected features are missing from normalized matrix: {missing_features[:20]}")

    df_for_training = df_raw.loc[training_row_indices].copy()
    df_for_other_groups = df_raw.loc[other_group_row_indices].copy()
    df_for_shap_and_agg = df_raw.loc[all_groups_row_indices].copy()

    if "group" not in df_for_training.columns:
        raise ValueError("Normalized feature matrix does not contain required 'group' column.")
    if "sex" not in df_for_training.columns:
        raise ValueError("Normalized feature matrix does not contain required 'sex' column for shard execution.")

    training_groups = list(manifest.get("training_groups") or [])
    if len(training_groups) != 2:
        raise ValueError(
            f"Plan manifest expected exactly two training groups, got: {training_groups!r}"
        )
    training_group_a = str(training_groups[0])
    training_group_b = str(training_groups[1])

    le = FixedBinaryLabelEncoder(class0=training_group_a, class1=training_group_b)
    y_encoded_train = le.transform(df_for_training["group"].values)
    num_classes = len(le.classes_)
    if num_classes != 2:
        raise ValueError(f"Expected binary classification for shard execution, got {num_classes}")

    df_for_training["sex_group"] = (
        df_for_training["sex"].astype(str) + "_" + df_for_training["group"].astype(str)
    )
    n_splits = int(manifest.get("n_splits", 0) or 0)
    sex_group_counts = validate_stratified_cv_labels(
        df_for_training["sex_group"].values,
        n_splits=n_splits,
        context="FeatureImp shard cross-validation",
    )
    le_sex_group = LabelEncoder()
    sex_group_encoded = le_sex_group.fit_transform(df_for_training["sex_group"])

    X_train_full = df_for_training[selected_features].copy()
    if int(X_train_full.isna().sum().sum()) > 0:
        X_train_full = X_train_full.fillna(0)

    X_shap_full = df_for_shap_and_agg[selected_features].copy()
    if int(X_shap_full.isna().sum().sum()) > 0:
        X_shap_full = X_shap_full.fillna(0)

    fold_plan_records = {
        int(rec["h_hit"]): rec for rec in _jsonl_load(fold_plan_path)
    }
    validate_hhit_fold_plan(
        list(fold_plan_records.values()),
        n_training_rows=len(training_row_indices),
        n_splits=n_splits,
        expected_h_hits=int(manifest.get("H_hits_per_feature", 0) or 0),
    )
    subset_plan_records = {
        int(rec["h_hit"]): rec for rec in _jsonl_load(subset_plan_path)
    }

    permutation_seed_map: Dict[Tuple[int, int, int], int] = {}
    if os.path.exists(permutation_seeds_path):
        for row in _read_csv_rows(permutation_seeds_path):
            permutation_seed_map[
                (
                    int(row.get("permutation_idx", 0) or 0),
                    int(row.get("h_hit", 0) or 0),
                    int(row.get("fold_idx", 0) or 0),
                )
            ] = int(row.get("permutation_seed", 0) or 0)

    return {
        "manifest": dict(manifest),
        "normalized_matrix_path": normalized_matrix_path,
        "df_for_training": df_for_training,
        "df_for_other_groups": df_for_other_groups,
        "df_for_shap_and_agg": df_for_shap_and_agg,
        "X_train_full": X_train_full,
        "X_shap_full": X_shap_full,
        "y_encoded_train": y_encoded_train,
        "le": le,
        "num_classes": int(num_classes),
        "selected_features": selected_features,
        "feature_to_idx": {feat: idx for idx, feat in enumerate(selected_features)},
        "sex_group_encoded": sex_group_encoded,
        "sex_group_counts": sex_group_counts,
        "fold_plan_records": fold_plan_records,
        "subset_plan_records": subset_plan_records,
        "permutation_seed_map": permutation_seed_map,
    }


def _build_real_fold_indices(fold_plan_record: Mapping[str, Any]) -> Tuple[Tuple[np.ndarray, np.ndarray], ...]:
    folds: List[Tuple[np.ndarray, np.ndarray]] = []
    for rec in list(fold_plan_record.get("folds") or []):
        train_idx = np.asarray(rec.get("train_indices") or [], dtype=np.int64)
        test_idx = np.asarray(rec.get("test_indices") or [], dtype=np.int64)
        folds.append((train_idx, test_idx))
    return tuple(folds)


def _baseline_fold_logloss_sums(
    *,
    y_encoded_train: np.ndarray,
    fold_indices_round: Sequence[Tuple[np.ndarray, np.ndarray]],
    num_classes: int,
) -> List[float]:
    """Calculate constant-predictor log-loss sums using each fold's training prevalence."""
    baseline_sums: List[float] = []
    for train_indices, test_indices in fold_indices_round:
        y_train_fold = y_encoded_train[np.asarray(train_indices, dtype=np.int64)]
        pi_fold = float(np.mean(y_train_fold)) if len(y_train_fold) > 0 else 0.5
        y_test_fold = y_encoded_train[np.asarray(test_indices, dtype=np.int64)]
        p_baseline = np.full(len(y_test_fold), pi_fold, dtype=np.float64)
        baseline_sums.append(
            float(
                log_loss(
                    y_test_fold,
                    p_baseline,
                    labels=list(range(num_classes)),
                    normalize=False,
                )
            )
        )
    return baseline_sums


def _init_partial_accumulators(*, n_train: int, n_all: int, n_feats: int, num_classes: int) -> Dict[str, Any]:
    return {
        "oof_probas_sum": np.zeros((n_train, num_classes), dtype=np.float32),
        "oof_shap_sum": np.zeros((n_train, n_feats), dtype=np.float32),
        "insample_probas_sum": np.zeros((n_train, num_classes), dtype=np.float32),
        "insample_shap_sum": np.zeros((n_train, n_feats), dtype=np.float32),
        "train_pred_count": np.zeros(n_train, dtype=np.int32),
        "test_pred_count": np.zeros(n_train, dtype=np.int32),
        "all_groups_probas_sum": np.zeros((n_all, num_classes), dtype=np.float32),
        "all_groups_probas_count": 0,
        "all_groups_shap_sum": np.zeros((n_all, n_feats), dtype=np.float32),
        "feature_subset_hit_counts": np.zeros(n_feats, dtype=np.int32),
        "feature_model_hit_counts": np.zeros(n_feats, dtype=np.int32),
    }


def _subset_performance_row_from_result(
    *,
    result: Mapping[str, Any],
    baseline_fold_logloss_sums: Sequence[float],
    n_splits: int,
    n_train: int,
    is_permuted: bool,
    permutation_idx: Optional[int],
) -> Dict[str, Any]:
    """Validate fold metrics and summarize one subset's out-of-fold skill."""
    h_hit = int(result.get("h_hit", 0) or 0)
    subset_idx = int(result.get("subset_idx", 0) or 0)
    n_features = int(result.get("n_features", len(result.get("features", []) or [])) or 0)

    folds_seen: set[int] = set()
    folds_with_metrics: set[int] = set()
    oof_logloss_sum = 0.0
    n_test_seen_total = 0
    auc_weighted_sum = 0.0
    auc_weighted_n = 0

    for fold_metric in list(result.get("fold_metrics") or []):
        fold_idx_raw = fold_metric.get("fold_idx", -1)
        fold_idx = int(fold_idx_raw) if fold_idx_raw not in (None, "") else -1
        if fold_idx < 0:
            continue
        folds_seen.add(fold_idx)
        n_test_seen = int(fold_metric.get("n_test_seen_total", 0) or 0)
        if n_test_seen > 0:
            oof_logloss_sum += float(fold_metric.get("oof_logloss_sum", 0.0) or 0.0)
            n_test_seen_total += n_test_seen
            folds_with_metrics.add(fold_idx)
            auc_weighted_sum += float(fold_metric.get("auc_weighted_sum", 0.0) or 0.0)
            auc_weighted_n += int(fold_metric.get("auc_weighted_n", 0) or 0)

    baseline_logloss_sum_full = float(np.sum(np.asarray(list(baseline_fold_logloss_sums), dtype=np.float64)))
    baseline_logloss_full = baseline_logloss_sum_full / float(n_train) if n_train > 0 else float("nan")
    baseline_sum_seen = float(
        np.sum([baseline_fold_logloss_sums[i] for i in sorted(folds_with_metrics)])
    ) if folds_with_metrics else 0.0
    baseline_logloss_seen = (
        baseline_sum_seen / float(n_test_seen_total) if n_test_seen_total > 0 else float("nan")
    )
    oof_logloss = (
        oof_logloss_sum / float(n_test_seen_total) if n_test_seen_total > 0 else float("nan")
    )
    delta_logloss = baseline_logloss_seen - oof_logloss if np.isfinite(baseline_logloss_seen) else float("nan")
    delta_norm = (
        delta_logloss / baseline_logloss_seen
        if np.isfinite(delta_logloss) and np.isfinite(baseline_logloss_seen) and baseline_logloss_seen > 0
        else float("nan")
    )
    oof_auc = auc_weighted_sum / float(auc_weighted_n) if auc_weighted_n > 0 else float("nan")

    return {
        "subset_id": f"h{h_hit:03d}_s{subset_idx:04d}",
        "h_hit": int(h_hit),
        "subset_idx": int(subset_idx),
        "n_features": int(n_features),
        "n_splits": int(n_splits),
        "n_train_total": int(n_train),
        "baseline_logloss_full": float(baseline_logloss_full),
        "baseline_logloss_sum_full": float(baseline_logloss_sum_full),
        "folds_seen_count": int(len(folds_seen)),
        "folds_with_metrics_count": int(len(folds_with_metrics)),
        "n_test_seen_total": int(n_test_seen_total),
        "subset_complete": bool(len(folds_seen) == int(n_splits) and n_test_seen_total == int(n_train)),
        "baseline_logloss_seen": float(baseline_logloss_seen),
        "oof_logloss": float(oof_logloss),
        "delta_logloss": float(delta_logloss),
        "delta_norm": float(delta_norm),
        "oof_auc": float(oof_auc),
        "auc_weighted_n": int(auc_weighted_n),
        "is_permuted": bool(is_permuted),
        "permutation_idx": None if permutation_idx is None else int(permutation_idx),
    }


def _performance_records_from_result(result: Mapping[str, Any], *, is_permuted: bool, permutation_idx: Optional[int]) -> List[Dict[str, Any]]:
    """Attach real or permutation identity to fold-performance records."""
    specialist_features = result.get("features", [])
    if isinstance(specialist_features, (pd.Index, np.ndarray, tuple)):
        specialist_features = list(specialist_features)
    elif not isinstance(specialist_features, list):
        specialist_features = [str(specialist_features)]
    return [
        {
            "h_hit": int(perf["h_hit"]),
            "subset_idx": int(perf["subset_idx"]),
            "fold_idx": int(perf["fold_idx"]),
            "n_features": int(result.get("n_features", len(specialist_features)) or len(specialist_features)),
            "accuracy_test": float(perf.get("accuracy_test", np.nan)),
            "accuracy_train": float(perf.get("accuracy_train", np.nan)),
            "specialist_features": [str(f) for f in specialist_features],
            "features": [str(f) for f in specialist_features],
            "model_path": perf.get("model_path", None),
            "is_permuted": bool(is_permuted),
            "permutation_idx": None if permutation_idx is None else int(permutation_idx),
        }
        for perf in list(result.get("fold_performance") or [])
    ]


def _accumulate_result(
    *,
    accum: MutableMapping[str, Any],
    result: Mapping[str, Any],
    feature_to_idx: Mapping[str, int],
) -> None:
    """Add one subset result to ordered feature-level and sample-level accumulators."""
    feature_indices = np.asarray([feature_to_idx[f] for f in list(result.get("features") or [])], dtype=np.int32)

    accum["feature_subset_hit_counts"][feature_indices] += 1

    if result.get("oof_shap_sum") is not None:
        accum["oof_shap_sum"][:, feature_indices] += np.asarray(result["oof_shap_sum"], dtype=np.float32)
    if result.get("oof_probas_sum") is not None:
        accum["oof_probas_sum"] += np.asarray(result["oof_probas_sum"], dtype=np.float32)
    if result.get("test_pred_count") is not None:
        accum["test_pred_count"] += np.asarray(result["test_pred_count"], dtype=np.int32)

    if result.get("insample_shap_sum") is not None:
        accum["insample_shap_sum"][:, feature_indices] += np.asarray(result["insample_shap_sum"], dtype=np.float32)
    if result.get("insample_probas_sum") is not None:
        accum["insample_probas_sum"] += np.asarray(result["insample_probas_sum"], dtype=np.float32)
    if result.get("train_pred_count") is not None:
        accum["train_pred_count"] += np.asarray(result["train_pred_count"], dtype=np.int32)

    model_count = int(result.get("all_groups_model_count", 0) or 0)
    if model_count > 0 and result.get("all_groups_shap_sum") is not None:
        accum["all_groups_shap_sum"][:, feature_indices] += np.asarray(result["all_groups_shap_sum"], dtype=np.float32)
        accum["feature_model_hit_counts"][feature_indices] += int(model_count)
        if result.get("all_groups_probas_sum") is not None:
            accum["all_groups_probas_sum"] += np.asarray(result["all_groups_probas_sum"], dtype=np.float32)
            accum["all_groups_probas_count"] = int(accum["all_groups_probas_count"]) + int(model_count)


def _save_partial_npz(path: str, *, partial: Mapping[str, Any], metadata: Mapping[str, Any]) -> None:
    arrays = {k: np.asarray(v) if not np.isscalar(v) else np.asarray(v) for k, v in partial.items()}
    for mk, mv in metadata.items():
        arrays[f"meta__{mk}"] = np.asarray(mv)
    # Internal shard scratch payloads favor faster write/read throughput over compression.
    np.savez(path, **arrays)


def _empty_subset_performance_frame() -> pd.DataFrame:
    """Create the declared subset-performance columns for an empty task group."""
    return pd.DataFrame(
        columns=[
            "subset_id",
            "h_hit",
            "subset_idx",
            "n_features",
            "n_splits",
            "n_train_total",
            "baseline_logloss_full",
            "baseline_logloss_sum_full",
            "folds_seen_count",
            "folds_with_metrics_count",
            "n_test_seen_total",
            "subset_complete",
            "baseline_logloss_seen",
            "oof_logloss",
            "delta_logloss",
            "delta_norm",
            "oof_auc",
            "auc_weighted_n",
            "is_permuted",
            "permutation_idx",
        ]
    )


def _margin_task_record_from_result(result: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "h_hit": int(result.get("h_hit", 0) or 0),
        "subset_idx": int(result.get("subset_idx", 0) or 0),
        "oof_margin_sum": np.asarray(result.get("oof_margin_sum"), dtype=np.float64),
        "insample_margin_sum": np.asarray(result.get("insample_margin_sum"), dtype=np.float64),
        "all_groups_margin_sum": np.asarray(result.get("all_groups_margin_sum"), dtype=np.float64),
        "test_pred_count": np.asarray(result.get("test_pred_count"), dtype=np.int32),
        "train_pred_count": np.asarray(result.get("train_pred_count"), dtype=np.int32),
        "all_groups_model_count": int(result.get("all_groups_model_count", 0) or 0),
    }


def _shap_task_record_from_result(
    result: Mapping[str, Any],
    *,
    feature_to_idx: Mapping[str, int],
    include_insample: bool,
    include_all_groups: bool,
) -> Dict[str, Any]:
    """Build an ordered subset SHAP record for the requested output views."""
    feature_indices = np.asarray(
        [feature_to_idx[f] for f in list(result.get("features") or [])],
        dtype=np.int32,
    )
    rec: Dict[str, Any] = {
        "h_hit": int(result.get("h_hit", 0) or 0),
        "subset_idx": int(result.get("subset_idx", 0) or 0),
        "feature_indices": feature_indices,
        "oof_shap_sum": np.asarray(result.get("oof_shap_sum"), dtype=np.float32),
    }
    if include_insample:
        rec["insample_shap_sum"] = np.asarray(result.get("insample_shap_sum"), dtype=np.float32)
    if include_all_groups:
        rec["all_groups_shap_sum"] = np.asarray(result.get("all_groups_shap_sum"), dtype=np.float32)
    return rec


def _run_task_group(
    *,
    group_label: str,
    task_payloads: Sequence[Dict[str, Any]],
    X_train_full: pd.DataFrame,
    X_shap_full: pd.DataFrame,
    y_encoded_train: np.ndarray,
    le: Any,
    num_workers: int,
    feature_to_idx: Mapping[str, int],
    accum: MutableMapping[str, Any],
    baseline_fold_logloss_sums: Sequence[float],
    n_splits: int,
    n_train: int,
    is_permuted: bool,
    permutation_idx: Optional[int],
    runner=None,
    archive_writer=None,
    error_path=None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Run and accumulate assigned subset tasks, recording every worker failure."""
    subset_rows: List[Dict[str, Any]] = []
    performance_rows: List[Dict[str, Any]] = []
    margin_task_rows: List[Dict[str, Any]] = []
    shap_task_rows: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []

    if not task_payloads:
        return subset_rows, performance_rows, margin_task_rows, shap_task_rows, errors

    result_iterator = (runner.run(_train_subset_worker, task_payloads, desc=group_label,
                                  stage_name=f"FeatureImp shard {group_label}")
                       if runner is not None else iter_parallel_results(
        _train_subset_worker,
        task_payloads,
        num_workers=num_workers,
        initializer=_init_fast_pool_worker,
        initargs=(X_train_full, X_shap_full, y_encoded_train, le),
        desc=group_label,
        stage_name=f"FeatureImp shard {group_label}",
    ))
    for result in result_iterator:
        if result is None:
            record_work("featureimp_subset_tasks", failed=1)
            errors.append(
                {
                    "error_type": "WorkerReturnedNone",
                    "error_message": f"Subset worker returned None during {group_label}.",
                    "is_permuted": bool(is_permuted),
                    "permutation_idx": permutation_idx,
                }
            )
            continue

        if isinstance(result, dict) and result.get("__worker_error__"):
            record_work("featureimp_subset_tasks", failed=1)
            errors.append(dict(result))
            continue

        failed_folds = list(result.get("failed_folds") or [])
        record_work("featureimp_subset_tasks", completed=int(not failed_folds), failed=int(bool(failed_folds)))
        if failed_folds:
            errors.extend([dict(err) for err in failed_folds])

        _accumulate_result(accum=accum, result=result, feature_to_idx=feature_to_idx)
        subset_rows.append(
            _subset_performance_row_from_result(
                result=result,
                baseline_fold_logloss_sums=baseline_fold_logloss_sums,
                n_splits=n_splits,
                n_train=n_train,
                is_permuted=is_permuted,
                permutation_idx=permutation_idx,
            )
        )
        performance_rows.extend(
            _performance_records_from_result(
                result,
                is_permuted=is_permuted,
                permutation_idx=permutation_idx,
            )
        )
        if archive_writer is not None:
            archive_writer.append(result, feature_to_idx)
        else:
            margin_task_rows.append(_margin_task_record_from_result(result))
            shap_task_rows.append(
                _shap_task_record_from_result(
                    result,
                    feature_to_idx=feature_to_idx,
                    include_insample=True,
                    include_all_groups=True,
                )
            )

    if errors and error_path is not None:
        _write_jsonl(errors, error_path)
        raise RuntimeError(f"FeatureImp shard {group_label} failed: {errors[0].get('error_message', errors[0])}; see {error_path}")
    return subset_rows, performance_rows, margin_task_rows, shap_task_rows, errors


def _process_featureimp_shard_owned(
    analysis_folder_path: str,
    *,
    shard_index: int,
) -> str:
    """Run one staged FeatureImp shard and write shard-local partial artifacts."""

    from .prereqs import check_shap_xgboost_treeexplainer_compatibility

    analysis_folder_path = os.path.abspath(str(analysis_folder_path))
    if not os.path.isdir(analysis_folder_path):
        raise FileNotFoundError(f"FeatureImp analysis folder not found: {analysis_folder_path}")

    manifest = load_featureimp_plan_manifest(analysis_folder_path)
    if str(manifest.get("status", "")).strip().lower() != "planned":
        raise ValueError(
            f"FeatureImp analysis folder does not have planned status: {manifest.get('status')!r}"
        )

    _shard_plan, shard_task_rows, shard_label = _load_shard_plan_and_tasks(
        analysis_folder_path,
        shard_index=int(shard_index),
    )

    shard_output_dir = os.path.join(analysis_folder_path, SHARD_OUTPUT_DIRNAME, shard_label)
    os.makedirs(shard_output_dir, exist_ok=True)

    done_path = os.path.join(shard_output_dir, SHARD_DONE_FILENAME)
    if os.path.exists(done_path):
        plan_manifest_path = os.path.join(
            analysis_folder_path,
            PLAN_DIRNAME,
            "PLAN_MANIFEST.json",
        )
        valid_done, reason = _validate_completed_shard(
            done_path,
            analysis_folder_path=analysis_folder_path,
            plan_manifest_path=plan_manifest_path,
        )
        if not valid_done:
            raise RuntimeError(
                "Existing FeatureImp shard completion marker cannot be safely reused: "
                f"{reason}. Remove or archive the shard output directory before rerunning."
            )
        print(f"Shard already completed successfully; reusing validated outputs: {done_path}")
        return shard_output_dir

    settings_snapshot = _load_settings_snapshot(analysis_folder_path)
    num_workers = _get_runtime_num_workers()
    model_subsample = float(settings_snapshot.get("model_subsample", 1.0))
    n_estimators = int(settings_snapshot.get("n_estimators", 200))
    learning_rate = float(settings_snapshot.get("learning_rate", 0.1))
    max_depth = int(settings_snapshot.get("max_depth", 6))
    resolved_global_seed = int(manifest.get("resolved_global_random_seed", 0) or 0)
    n_splits = int(manifest.get("n_splits", 0) or 0)

    check_shap_xgboost_treeexplainer_compatibility()

    running_path = os.path.join(shard_output_dir, SHARD_RUNNING_FILENAME)
    failed_path = os.path.join(shard_output_dir, SHARD_FAILED_FILENAME)
    errors_path = os.path.join(shard_output_dir, SHARD_ERRORS_FILENAME)
    lock_dir = os.path.join(shard_output_dir, LOCK_DIRNAME)

    for stale in (failed_path, errors_path):
        if os.path.exists(stale):
            try:
                os.remove(stale)
            except Exception:
                pass

    start_ts = datetime.now().isoformat(timespec="seconds")
    _json_dump(
        {
            "status": "running",
            "started_at": start_ts,
            "analysis_folder": analysis_folder_path,
            "shard_index": int(shard_index),
            "shard_label": shard_label,
            "task_count": int(len(shard_task_rows)),
        },
        running_path,
    )

    try:
        with ExitStack() as run_stack:
            context = _load_featureimp_context(analysis_folder_path, manifest)

            df_for_training = context["df_for_training"]
            X_train_full = context["X_train_full"]
            X_shap_full = context["X_shap_full"]
            y_encoded_train = context["y_encoded_train"]
            le = context["le"]
            num_classes = int(context["num_classes"])
            feature_to_idx = context["feature_to_idx"]
            selected_features = list(context["selected_features"])
            sex_group_encoded = np.asarray(context["sex_group_encoded"])
            fold_plan_records = dict(context["fold_plan_records"])
            subset_plan_records = dict(context["subset_plan_records"])
            permutation_seed_map = dict(context["permutation_seed_map"])
            n_train = int(len(df_for_training))
            n_all = int(len(X_shap_full))
            n_feats = int(len(selected_features))

            pipeline_folder = os.path.join(analysis_folder_path, "SAVED_PIPELINE")
            os.makedirs(pipeline_folder, exist_ok=True)
            models_folder = os.path.join(pipeline_folder, "models", shard_label)
            os.makedirs(models_folder, exist_ok=True)

            paths = {
                "pipeline_folder": pipeline_folder,
                "shap_values_folder": os.path.join(pipeline_folder, "temp_shap_values"),
                "models_folder": models_folder,
                "analysis_folder": analysis_folder_path,
                "file_suffix": "",
                "permuted_base_folder": os.path.join(analysis_folder_path, "PERMUTED_RESULTS")
                if bool(manifest.get("run_permutation_test", False))
                else None,
            }
            os.makedirs(paths["shap_values_folder"], exist_ok=True)

            real_tasks_by_hhit: Dict[int, List[Dict[str, Any]]] = {}
            perm_tasks_by_perm_hhit: Dict[int, Dict[int, List[Dict[str, Any]]]] = {}
            for row in shard_task_rows:
                family = str(row.get("family", ""))
                h_hit = int(row.get("h_hit", 0) or 0)
                if family == "real":
                    real_tasks_by_hhit.setdefault(h_hit, []).append(dict(row))
                elif family == "perm":
                    perm_idx = int(row.get("permutation_idx", 0) or 0)
                    perm_tasks_by_perm_hhit.setdefault(perm_idx, {}).setdefault(h_hit, []).append(dict(row))
                else:
                    raise ValueError(f"Unknown shard task family: {family!r}")

            real_partial = _init_partial_accumulators(
                n_train=n_train,
                n_all=n_all,
                n_feats=n_feats,
                num_classes=num_classes,
            )
            perm_partials: Dict[int, Dict[str, Any]] = {}
            real_subset_rows: List[Dict[str, Any]] = []
            perm_subset_rows: List[Dict[str, Any]] = []
            real_performance_rows: List[Dict[str, Any]] = []
            perm_performance_rows: List[Dict[str, Any]] = []
            errors: List[Dict[str, Any]] = []
            task_features = {str(row['task_key']): tuple(subset_plan_records[int(row['h_hit'])]['feature_subsets'][int(row['subset_idx'])]) for row in shard_task_rows}
            archive_bytes = sum(4 * (2 * n_train + n_all + 1) * len(task_features[str(row['task_key'])]) +
                                8 * (2 * n_train + n_all) + 8 * n_train + 128 for row in shard_task_rows)
            matrix_bytes = int(X_train_full.memory_usage(index=False).sum() + X_shap_full.memory_usage(index=False).sum())
            scratch = run_stack.enter_context(scratch_workspace('featureimp', required_bytes=archive_bytes + matrix_bytes + 1048576))
            def matrix_descriptor(frame, name):
                path = scratch / f'{name}.npy'
                np.save(path, frame.to_numpy(copy=False), allow_pickle=False)
                return {'memmap_path': str(path), 'feature_columns': list(frame.columns)}
            descriptors = matrix_descriptor(X_train_full, 'training'), matrix_descriptor(X_shap_full, 'all_groups')
            run_stack.callback(_clear_fast_pool_worker)
            from .execution import worker_capacity, GIB
            subset_width = max((len(v) for v in task_features.values()), default=n_feats)
            # Inputs, SHAP temporaries, tree workspace and interpreter/native overhead.
            worker_bytes = GIB + 96 * (n_train + n_all) * subset_width
            num_workers = worker_capacity(num_workers, worker_bytes, 'Training')
            runner = run_stack.enter_context(BoundedProcessRunner(num_workers=num_workers,
                estimated_worker_bytes=worker_bytes,
                initializer=_init_fast_pool_worker, initargs=(*descriptors, y_encoded_train, le)))
            def make_writer(family, permutation_idx=None):
                specs = {(int(row['h_hit']), int(row['subset_idx'])): task_features[str(row['task_key'])]
                         for row in shard_task_rows if row['family'] == family and row.get('permutation_idx') == permutation_idx}
                writer = TaskArchiveWriter(scratch / f'{family}_{permutation_idx}', specs, n_train=n_train, n_all=n_all)
                run_stack.callback(writer.close)
                return writer
            real_writer = make_writer('real')

            print("\n" + "=" * 80)
            print("PROCESSING FEATUREIMP SHARD")
            print("=" * 80)
            print(f"Analysis folder: {analysis_folder_path}")
            print(f"Shard: {shard_label} (index {shard_index})")
            print(f"Tasks assigned: {len(shard_task_rows)}")
            print(f"Using num_workers={num_workers}")
            print(f"Selected features: {n_feats}")
            print(f"Training rows: {n_train}; all-groups rows: {n_all}")
            print(f"Training sex_group counts: {context['sex_group_counts']}")

            for h_hit in sorted(real_tasks_by_hhit.keys()):
                task_rows_h = sorted(real_tasks_by_hhit[h_hit], key=lambda r: int(r["subset_idx"]))
                fold_record = fold_plan_records.get(int(h_hit))
                if fold_record is None:
                    raise KeyError(f"Missing fold plan for real h_hit={h_hit}")
                subset_record = subset_plan_records.get(int(h_hit))
                if subset_record is None:
                    raise KeyError(f"Missing subset plan for real h_hit={h_hit}")
                fold_indices_round = _build_real_fold_indices(fold_record)
                baseline_sums = _baseline_fold_logloss_sums(
                    y_encoded_train=y_encoded_train,
                    fold_indices_round=fold_indices_round,
                    num_classes=num_classes,
                )
                task_payloads: List[Dict[str, Any]] = []
                for row in task_rows_h:
                    subset_idx = int(row["subset_idx"])
                    feature_subsets = list(subset_record.get("feature_subsets") or [])
                    if subset_idx >= len(feature_subsets):
                        raise IndexError(
                            f"subset_idx={subset_idx} out of range for real h_hit={h_hit} subset plan"
                        )
                    selected_subset_features = tuple(str(f) for f in feature_subsets[subset_idx])
                    task_payloads.append(
                        {
                            "h_hit": int(h_hit),
                            "subset_idx": int(subset_idx),
                            "selected_features": selected_subset_features,
                            "fold_indices_round": fold_indices_round,
                            "num_classes": int(num_classes),
                            "model_subsample": float(model_subsample),
                            "n_estimators": int(n_estimators),
                            "learning_rate": float(learning_rate),
                            "max_depth": int(max_depth),
                            "global_random_seed": int(resolved_global_seed),
                            "paths": paths,
                            "is_permuted": False,
                            "permutation_idx": None,
                        }
                    )

                print(f"\nReal shard tasks for h_hit={h_hit}: {len(task_payloads)} subset tasks")
                subset_rows, performance_rows, margin_rows, shap_rows, group_errors = _run_task_group(
                    group_label=f"real h_hit {h_hit}", runner=runner, archive_writer=real_writer, error_path=errors_path,
                    task_payloads=task_payloads,
                    X_train_full=X_train_full,
                    X_shap_full=X_shap_full,
                    y_encoded_train=y_encoded_train,
                    le=le,
                    num_workers=num_workers,
                    feature_to_idx=feature_to_idx,
                    accum=real_partial,
                    baseline_fold_logloss_sums=baseline_sums,
                    n_splits=n_splits,
                    n_train=n_train,
                    is_permuted=False,
                    permutation_idx=None,
                )
                real_subset_rows.extend(subset_rows)
                real_performance_rows.extend(performance_rows)
                errors.extend(group_errors)

            for perm_idx in sorted(perm_tasks_by_perm_hhit.keys()):
                perm_writer = make_writer('perm', int(perm_idx))
                perm_partial = _init_partial_accumulators(
                    n_train=n_train,
                    n_all=n_all,
                    n_feats=n_feats,
                    num_classes=num_classes,
                )
                perm_partials[int(perm_idx)] = perm_partial

                for h_hit in sorted(perm_tasks_by_perm_hhit[perm_idx].keys()):
                    task_rows_h = sorted(
                        perm_tasks_by_perm_hhit[perm_idx][h_hit],
                        key=lambda r: int(r["subset_idx"]),
                    )
                    subset_record = subset_plan_records.get(int(h_hit))
                    if subset_record is None:
                        raise KeyError(f"Missing subset plan for permutation {perm_idx}, h_hit={h_hit}")

                    split_seed = int(resolved_global_seed + int(perm_idx) * 100000 + int(h_hit) * 1000)
                    fold_indices_round = tuple(
                        make_hhit_folds(
                            X_train_full,
                            sex_group_encoded,
                            n_splits=n_splits,
                            random_state=split_seed,
                        )
                    )
                    baseline_sums = _baseline_fold_logloss_sums(
                        y_encoded_train=y_encoded_train,
                        fold_indices_round=fold_indices_round,
                        num_classes=num_classes,
                    )
                    y_original_full = df_for_training["group"].to_numpy()
                    permuted_by_fold: List[np.ndarray] = []
                    for fold_idx, (train_indices, test_indices) in enumerate(fold_indices_round):
                        perm_seed = permutation_seed_map.get((int(perm_idx), int(h_hit), int(fold_idx)))
                        if perm_seed is None:
                            perm_seed = int(resolved_global_seed + int(perm_idx) * 100000 + int(h_hit) * 1000 + int(fold_idx))
                        y_permuted_full = _generate_sex_class_stratified_permutation(
                            y_original=y_original_full,
                            df_with_sex=df_for_training,
                            train_indices=train_indices,
                            test_indices=test_indices,
                            permutation_seed=int(perm_seed),
                        )
                        permuted_by_fold.append(np.asarray(y_permuted_full))

                    task_payloads = []
                    for row in task_rows_h:
                        subset_idx = int(row["subset_idx"])
                        feature_subsets = list(subset_record.get("feature_subsets") or [])
                        if subset_idx >= len(feature_subsets):
                            raise IndexError(
                                f"subset_idx={subset_idx} out of range for permutation {perm_idx}, h_hit={h_hit} subset plan"
                            )
                        selected_subset_features = tuple(str(f) for f in feature_subsets[subset_idx])
                        task_payloads.append(
                            {
                                "h_hit": int(h_hit),
                                "subset_idx": int(subset_idx),
                                "selected_features": selected_subset_features,
                                "fold_indices_round": fold_indices_round,
                                "num_classes": int(num_classes),
                                "model_subsample": float(model_subsample),
                                "n_estimators": int(n_estimators),
                                "learning_rate": float(learning_rate),
                                "max_depth": int(max_depth),
                                "global_random_seed": int(resolved_global_seed),
                                "paths": paths,
                                "is_permuted": True,
                                "permutation_idx": int(perm_idx),
                                "y_permuted_by_fold": tuple(permuted_by_fold),
                                "y_original_full": y_original_full,
                            }
                        )

                    print(
                        f"\nPermuted shard tasks for perm={perm_idx}, h_hit={h_hit}: {len(task_payloads)} subset tasks"
                    )
                    subset_rows, performance_rows, margin_rows, shap_rows, group_errors = _run_task_group(
                        group_label=f"perm {perm_idx} h_hit {h_hit}", runner=runner, archive_writer=perm_writer, error_path=errors_path,
                        task_payloads=task_payloads,
                        X_train_full=X_train_full,
                        X_shap_full=X_shap_full,
                        y_encoded_train=y_encoded_train,
                        le=le,
                        num_workers=num_workers,
                        feature_to_idx=feature_to_idx,
                        accum=perm_partial,
                        baseline_fold_logloss_sums=baseline_sums,
                        n_splits=n_splits,
                        n_train=n_train,
                        is_permuted=True,
                        permutation_idx=int(perm_idx),
                    )
                    perm_subset_rows.extend(subset_rows)
                    perm_performance_rows.extend(performance_rows)
                    errors.extend(group_errors)

                perm_margin_tasks_path = os.path.join(
                    shard_output_dir,
                    PERM_MARGIN_TASKS_FILENAME_TEMPLATE.format(perm_idx=int(perm_idx)),
                )
                perm_shap_tasks_path = os.path.join(shard_output_dir, PERM_SHAP_TASKS_FILENAME_TEMPLATE.format(perm_idx=int(perm_idx)))
                perm_writer.publish(margin_path=perm_margin_tasks_path, shap_path=perm_shap_tasks_path,
                    metadata={'family': 'perm_tasks', 'permutation_idx': int(perm_idx), 'shard_index': int(shard_index)})
                perm_writer.close()
                perm_partials_path = os.path.join(
                    shard_output_dir,
                    PERM_PARTIALS_FILENAME_TEMPLATE.format(perm_idx=int(perm_idx)),
                )
                _save_partial_npz(
                    perm_partials_path,
                    partial=perm_partial,
                    metadata={
                        "family": "perm",
                        "permutation_idx": int(perm_idx),
                        "n_train": n_train,
                        "n_all": n_all,
                        "n_features": n_feats,
                        "num_classes": num_classes,
                        "shard_index": int(shard_index),
                    },
                )

                perm_partials[int(perm_idx)] = {}
                del perm_partial

            real_margin_tasks_path = os.path.join(shard_output_dir, REAL_MARGIN_TASKS_FILENAME)
            real_shap_tasks_path = os.path.join(shard_output_dir, REAL_SHAP_TASKS_FILENAME)
            real_writer.publish(margin_path=real_margin_tasks_path, shap_path=real_shap_tasks_path,
                metadata={'family': 'real_tasks', 'permutation_idx': -1, 'shard_index': int(shard_index)})
            real_writer.close()
            real_partials_path = os.path.join(shard_output_dir, REAL_PARTIALS_FILENAME)
            _save_partial_npz(
                real_partials_path,
                partial=real_partial,
                metadata={
                    "family": "real",
                    "permutation_idx": -1,
                    "n_train": n_train,
                    "n_all": n_all,
                    "n_features": n_feats,
                    "num_classes": num_classes,
                    "shard_index": int(shard_index),
                },
            )

            performance_real_path = os.path.join(shard_output_dir, PERFORMANCE_RECORDS_REAL_FILENAME)
            performance_perm_path = os.path.join(shard_output_dir, PERFORMANCE_RECORDS_PERM_FILENAME)
            subset_real_path = os.path.join(shard_output_dir, SUBSET_PERFORMANCE_REAL_FILENAME)
            subset_perm_path = os.path.join(shard_output_dir, SUBSET_PERFORMANCE_PERM_FILENAME)
            _write_jsonl(real_performance_rows, performance_real_path)
            _write_jsonl(perm_performance_rows, performance_perm_path)
            if real_subset_rows:
                pd.DataFrame(real_subset_rows).to_csv(subset_real_path, index=False)
            else:
                _empty_subset_performance_frame().to_csv(subset_real_path, index=False)
            if perm_subset_rows:
                pd.DataFrame(perm_subset_rows).to_csv(subset_perm_path, index=False)
            else:
                _empty_subset_performance_frame().to_csv(subset_perm_path, index=False)

            if errors:
                _write_jsonl(errors, errors_path)
                failure_summary = {
                    "status": "failed",
                    "started_at": start_ts,
                    "failed_at": datetime.now().isoformat(timespec="seconds"),
                    "analysis_folder": analysis_folder_path,
                    "shard_index": int(shard_index),
                    "shard_label": shard_label,
                    "task_count": int(len(shard_task_rows)),
                    "error_count": int(len(errors)),
                    "errors_path": os.path.relpath(errors_path, analysis_folder_path),
                }
                _json_dump(failure_summary, failed_path)
                raise RuntimeError(
                    f"Shard {shard_label} completed with {len(errors)} worker/fold errors. "
                    f"See {errors_path} for details."
                )

            summary = {
                "status": "completed",
                "started_at": start_ts,
                "completed_at": datetime.now().isoformat(timespec="seconds"),
                "analysis_folder": analysis_folder_path,
                "shard_index": int(shard_index),
                "shard_label": shard_label,
                "task_count": int(len(shard_task_rows)),
                "real_task_count": int(sum(1 for r in shard_task_rows if str(r.get("family")) == "real")),
                "perm_task_count": int(sum(1 for r in shard_task_rows if str(r.get("family")) == "perm")),
                "real_performance_record_count": int(len(real_performance_rows)),
                "perm_performance_record_count": int(len(perm_performance_rows)),
                "real_subset_row_count": int(len(real_subset_rows)),
                "perm_subset_row_count": int(len(perm_subset_rows)),
                "written_perm_indices": sorted(int(k) for k in perm_partials.keys()),
                "num_workers": int(num_workers),
                "selected_feature_count": int(n_feats),
                "n_train": int(n_train),
                "n_all": int(n_all),
                "artifacts": {
                    "real_partials_npz": os.path.relpath(real_partials_path, analysis_folder_path),
                    "real_margin_tasks_npz": os.path.relpath(real_margin_tasks_path, analysis_folder_path),
                    "real_shap_tasks_npz": os.path.relpath(real_shap_tasks_path, analysis_folder_path),
                    "performance_records_real_jsonl": os.path.relpath(performance_real_path, analysis_folder_path),
                    "performance_records_perm_jsonl": os.path.relpath(performance_perm_path, analysis_folder_path),
                    "subset_performance_real_csv": os.path.relpath(subset_real_path, analysis_folder_path),
                    "subset_performance_perm_csv": os.path.relpath(subset_perm_path, analysis_folder_path),
                    "perm_partials": {
                        str(perm_idx): os.path.relpath(
                            os.path.join(
                                shard_output_dir,
                                PERM_PARTIALS_FILENAME_TEMPLATE.format(perm_idx=int(perm_idx)),
                            ),
                            analysis_folder_path,
                        )
                        for perm_idx in sorted(perm_partials.keys())
                    },
                    "perm_margin_tasks": {
                        str(perm_idx): os.path.relpath(
                            os.path.join(
                                shard_output_dir,
                                PERM_MARGIN_TASKS_FILENAME_TEMPLATE.format(perm_idx=int(perm_idx)),
                            ),
                            analysis_folder_path,
                        )
                        for perm_idx in sorted(perm_partials.keys())
                    },
                    "perm_shap_tasks": {
                        str(perm_idx): os.path.relpath(
                            os.path.join(
                                shard_output_dir,
                                PERM_SHAP_TASKS_FILENAME_TEMPLATE.format(perm_idx=int(perm_idx)),
                            ),
                            analysis_folder_path,
                        )
                        for perm_idx in sorted(perm_partials.keys())
                    },
                    "model_files": [
                        os.path.relpath(str(path), analysis_folder_path)
                        for path in sorted(Path(models_folder).rglob("*.json"))
                    ],
                },
            }
            summary["plan_manifest_sha256"] = sha256_file(
                os.path.join(analysis_folder_path, PLAN_DIRNAME, "PLAN_MANIFEST.json")
            )
            artifact_hashes: Dict[str, str] = {}
            for relative in _flatten_artifact_paths(summary["artifacts"]):
                artifact_path = os.path.abspath(os.path.join(analysis_folder_path, relative))
                if not os.path.isfile(artifact_path):
                    raise FileNotFoundError(
                        f"Shard declared an output artifact that was not written: {relative}"
                    )
                artifact_hashes[relative] = sha256_file(artifact_path)
            summary["artifact_sha256"] = artifact_hashes
            _json_dump(summary, done_path)
            print("\nShard processing complete.")
            print(f"  - Done manifest: {done_path}")
            print(f"  - Real partials: {real_partials_path}")
            print(f"  - Real margin tasks: {real_margin_tasks_path}")
            print(f"  - Real SHAP tasks: {real_shap_tasks_path}")
            if perm_partials:
                print(f"  - Perm partials written: {sorted(perm_partials.keys())}")
                print(f"  - Perm margin task archives written: {sorted(perm_partials.keys())}")
                print(f"  - Perm SHAP task archives written: {sorted(perm_partials.keys())}")

    except Exception:
        if not os.path.exists(failed_path):
            _json_dump(
                {
                    "status": "failed",
                    "started_at": start_ts,
                    "failed_at": datetime.now().isoformat(timespec="seconds"),
                    "analysis_folder": analysis_folder_path,
                    "shard_index": int(shard_index),
                    "shard_label": shard_label,
                    "traceback": traceback.format_exc(),
                },
                failed_path,
            )
        raise
    finally:
        try:
            if os.path.exists(running_path):
                os.remove(running_path)
        except Exception:
            pass

    return shard_output_dir



@verified_run
def process_featureimp_shard_run(analysis_folder_path: str, *, shard_index: int) -> str:
    """Own the shard before inspecting or changing completion/status records."""
    analysis_folder_path = os.path.abspath(str(analysis_folder_path))
    _plan, _tasks, label = _load_shard_plan_and_tasks(analysis_folder_path, shard_index=int(shard_index))
    output = os.path.join(analysis_folder_path, SHARD_OUTPUT_DIRNAME, label)
    with _directory_lock(os.path.join(output, LOCK_DIRNAME)):
        return _process_featureimp_shard_owned(analysis_folder_path, shard_index=shard_index)

__all__ = [
    "SHARD_OUTPUT_DIRNAME",
    "SHARD_RUNNING_FILENAME",
    "SHARD_DONE_FILENAME",
    "SHARD_FAILED_FILENAME",
    "SHARD_ERRORS_FILENAME",
    "REAL_PARTIALS_FILENAME",
    "REAL_MARGIN_TASKS_FILENAME",
    "PERM_MARGIN_TASKS_FILENAME_TEMPLATE",
    "PERFORMANCE_RECORDS_REAL_FILENAME",
    "PERFORMANCE_RECORDS_PERM_FILENAME",
    "SUBSET_PERFORMANCE_REAL_FILENAME",
    "SUBSET_PERFORMANCE_PERM_FILENAME",
    "process_featureimp_shard_run",
]
