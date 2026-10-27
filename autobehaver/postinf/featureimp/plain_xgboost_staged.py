"""Plan, process, and merge sharded full-feature XGBoost comparison runs.

Each task processes one frozen coverage round or one permutation of that round.
"""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
import shutil
import time
import traceback
from datetime import datetime
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

from .integrity import strict_bool, seal_baseline_receipt, validate_baseline_receipt, baseline_inputs
from autobehaver.atomic_io import atomic_write_json
import numpy as np
import pandas as pd

from .baseline_outputs import (
    PLAIN_XGBOOST_STAGE_NAME,
    collect_plain_xgboost_artifact_paths,
    load_featureimp_baseline_context,
    upsert_postanalysis_stage_record,
    write_baseline_outputs_summary,
)
from .committee import FixedBinaryLabelEncoder
from .config import DEFAULT_SETTINGS
from .baseline_common import (
    _build_predictions_dataframe,
    _build_shap_summary_frames,
    _build_training_to_all_positions,
    _compute_binary_metrics,
    _make_group_labels_binary,
    _probabilities_from_logits,
    _resolve_feature_matrix,
    _write_shap_outputs,
)
from .finalize import _aggregate_and_summarize_permuted_results
from .io_utils import save_to_excel_sheets
from .matrix import compute_feature_schema_sha256
from .permutation import _generate_sex_class_stratified_permutation
from .plain_xgboost_baseline import (
    CONFIG_FILENAME,
    FOLD_PERFORMANCE_FILENAME,
    GLOBAL_IMPORTANCE_RANKING_FILENAME,
    MANIFEST_FILENAME,
    OUTPUT_DIRNAME,
    POOLED_PERFORMANCE_FILENAME,
    PREDICTIONS_FULL_FILENAME,
    PREDICTIONS_GRAPHPAD_FILENAME,
    REPEAT_PERFORMANCE_FILENAME,
    SHAP_COMBINED_FILENAME,
    SHAP_MALE_FEMALE_FILENAME,
    SHAP_PER_INDIVIDUAL_FILENAME,
    SHAP_STABILITY_FILENAME,
    SHAP_SUMMARY_FILENAME,
    _baseline_log,
    _build_real_fold_indices_local,
    _cleanup_stale_permutation_outputs,
    _compute_across_repeat_shap_stability,
    _finalize_global_importance_columns,
    _fit_plain_xgboost_model,
    _merge_prediction_views,
    _positive_class_shap,
    _run_optional_plain_xgboost_fsea,
    _write_global_importance_ranking,
)
from .postanalysis_logging import append_postanalysis_error_log
from .reporting import create_formatted_predictions_for_graphpad, save_predictions_with_formatting
from .shap_compat import make_tree_explainer
from .shard_runner import _get_runtime_num_workers, _directory_lock

STAGED_PLAN_DIRNAME = "_STAGED_PLAN"
STAGED_SHARDS_DIRNAME = "_STAGED_SHARDS"
STAGED_PLAN_MANIFEST_FILENAME = "PLAIN_XGBOOST_STAGED_PLAN_MANIFEST.json"
STAGED_TASK_MANIFEST_FILENAME = "PLAIN_XGBOOST_STAGED_TASK_MANIFEST.csv"
STAGED_PLAN_DONE_FILENAME = "PLAIN_XGBOOST_STAGED_PLAN_DONE.json"
SHARD_TASKS_FILENAME = "SHARD_TASKS.csv"
SHARD_RUNNING_FILENAME = "SHARD_RUNNING.json"
SHARD_DONE_FILENAME = "SHARD_DONE.json"
SHARD_FAILED_FILENAME = "SHARD_FAILED.json"
SHARD_ERRORS_FILENAME = "SHARD_ERRORS.jsonl"
REAL_PARTIALS_FILENAME = "REAL_PARTIALS.npz"
REAL_FOLD_ROWS_FILENAME = "REAL_FOLD_ROWS.jsonl"
REAL_REPEAT_ROWS_FILENAME = "REAL_REPEAT_ROWS.jsonl"
REAL_STABILITY_FILENAME = "REAL_STABILITY.npz"
PERM_PARTIALS_FILENAME_TEMPLATE = "PERM_PARTIALS_perm{perm_idx:03d}.npz"
PERM_FOLD_ROWS_FILENAME_TEMPLATE = "PERM_FOLD_ROWS_perm{perm_idx:03d}.jsonl"
MERGE_DONE_FILENAME = "PLAIN_XGBOOST_STAGED_MERGE_DONE.json"


def _json_dump(obj: Any, path: str) -> None:
    if os.path.basename(path) in {SHARD_DONE_FILENAME, MERGE_DONE_FILENAME}:
        obj = seal_baseline_receipt(obj, path)
    atomic_write_json(obj, path)


def _required_shard_artifacts(analysis_dir: str, shard_dir: str, tasks: Sequence[Mapping[str, Any]]) -> List[str]:
    """List the artifacts required by the assigned real and permutation tasks."""
    names = [REAL_PARTIALS_FILENAME, REAL_FOLD_ROWS_FILENAME, REAL_REPEAT_ROWS_FILENAME, REAL_STABILITY_FILENAME]
    permutations = sorted({int(t["permutation_idx"]) for t in tasks if str(t["family"]).lower() == "perm"})
    for index in permutations:
        names.extend([PERM_PARTIALS_FILENAME_TEMPLATE.format(perm_idx=index), PERM_FOLD_ROWS_FILENAME_TEMPLATE.format(perm_idx=index)])
    return [os.path.relpath(os.path.join(shard_dir, name), analysis_dir) for name in names]


def _validate_shard_tasks(shard_dir: str, tasks: Sequence[Mapping[str, Any]], receipt: Mapping[str, Any]) -> None:
    """Check identities of completed folds, rather than accepting counts alone."""
    if int(receipt.get("task_count", -1)) != len(tasks):
        raise ValueError("Baseline shard task count differs from its frozen plan")
    expected = {}
    for task in tasks:
        family = str(task["family"]).lower()
        permutation = int(task["permutation_idx"]) if family == "perm" else -1
        key = (family, permutation)
        folds = expected.setdefault(key, [])
        folds.extend((int(task["h_hit"]), i) for i in range(1, int(task["n_splits"]) + 1))
    expected.setdefault(("real", -1), [])
    for (family, permutation), planned in expected.items():
        filename = REAL_FOLD_ROWS_FILENAME if family == "real" else PERM_FOLD_ROWS_FILENAME_TEMPLATE.format(perm_idx=permutation)
        rows = _read_jsonl(os.path.join(shard_dir, filename))
        observed = [(int(row["h_hit"]), int(row["fold_idx"])) for row in rows]
        if sorted(observed) != sorted(planned) or len(set(observed)) != len(observed):
            raise ValueError("Baseline shard fold identities differ from its frozen tasks")
    repeats = _read_jsonl(os.path.join(shard_dir, REAL_REPEAT_ROWS_FILENAME))
    if sorted(int(row["h_hit"]) for row in repeats) != sorted(int(t["h_hit"]) for t in tasks if str(t["family"]).lower() == "real"):
        raise ValueError("Baseline completed repeat identities differ from its frozen tasks")


def _analysis_dir(path: str) -> str:
    return os.path.abspath(str(path))


def _plain_xgb_root(analysis_folder_path: str) -> str:
    return os.path.join(_analysis_dir(analysis_folder_path), OUTPUT_DIRNAME)


def _plan_root(analysis_folder_path: str) -> str:
    return os.path.join(_plain_xgb_root(analysis_folder_path), STAGED_PLAN_DIRNAME)


def _shards_root(analysis_folder_path: str) -> str:
    return os.path.join(_plain_xgb_root(analysis_folder_path), STAGED_SHARDS_DIRNAME)


def _plan_manifest_path(analysis_folder_path: str) -> str:
    return os.path.join(_plan_root(analysis_folder_path), STAGED_PLAN_MANIFEST_FILENAME)


def _task_manifest_path(analysis_folder_path: str) -> str:
    return os.path.join(_plan_root(analysis_folder_path), STAGED_TASK_MANIFEST_FILENAME)


def _plan_done_path(analysis_folder_path: str) -> str:
    return os.path.join(_plan_root(analysis_folder_path), STAGED_PLAN_DONE_FILENAME)


def _format_shard_label(shard_index: int, active_num_shards: int) -> str:
    width = max(3, len(str(max(0, int(active_num_shards) - 1))))
    return f"shard_{int(shard_index):0{width}d}_of_{int(active_num_shards):0{width}d}"


def _shard_plan_dir(analysis_folder_path: str, shard_index: int, active_num_shards: int) -> str:
    return os.path.join(_plan_root(analysis_folder_path), "shards", _format_shard_label(shard_index, active_num_shards))


def _shard_output_dir(analysis_folder_path: str, shard_index: int, active_num_shards: int) -> str:
    return os.path.join(_shards_root(analysis_folder_path), _format_shard_label(shard_index, active_num_shards))


def _json_load(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_jsonl(rows: Iterable[Mapping[str, Any]], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(dict(row), allow_nan=True, default=str))
            f.write("\n")


def _read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not os.path.exists(path):
        return rows
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _write_task_rows(path: str, task_rows: Sequence[Mapping[str, Any]]) -> None:
    """Write typed repeat assignments in the baseline task manifest."""
    fieldnames = [
        "task_id",
        "task_key",
        "shard_index",
        "shard_label",
        "family",
        "permutation_idx",
        "h_hit",
        "repeat_idx",
        "n_splits",
        "planned_model_count",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in task_rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def _read_task_rows(path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _iter_explicit_settings(settings_override: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    if settings_override is not None:
        return dict(settings_override)
    from . import core as _core

    return dict(_core.get_explicit_settings() or {})


def _current_num_shards(num_shards: Optional[int]) -> int:
    if num_shards is not None:
        return max(1, int(num_shards))
    from . import core as _core

    current = dict(_core.get_current_settings() or {})
    raw = current.get("num_shards", DEFAULT_SETTINGS.get("num_shards", 1))
    try:
        return max(1, int(raw) if raw is not None else int(DEFAULT_SETTINGS.get("num_shards", 1) or 1))
    except Exception:
        return max(1, int(DEFAULT_SETTINGS.get("num_shards", 1) or 1))


def _current_shard_index(shard_index: Optional[int]) -> int:
    if shard_index is not None:
        return int(shard_index)
    from . import core as _core

    current = dict(_core.get_current_settings() or {})
    raw = current.get("shard_index", DEFAULT_SETTINGS.get("shard_index", 0))
    try:
        return int(raw) if raw is not None else int(DEFAULT_SETTINGS.get("shard_index", 0) or 0)
    except Exception:
        return int(DEFAULT_SETTINGS.get("shard_index", 0) or 0)


def _collect_baseline_settings(
    loaded: Mapping[str, Any],
    *,
    settings_override: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Resolve baseline settings from frozen context and explicit overrides."""
    saved_snapshot = dict(loaded.get("settings_snapshot") or {})
    explicit = _iter_explicit_settings(settings_override)
    effective = dict(saved_snapshot)
    effective.update(explicit)
    return {
        **effective,
        "GLOBAL_RANDOM_SEED": int(effective.get("GLOBAL_RANDOM_SEED", loaded.get("global_random_seed", DEFAULT_SETTINGS.get("GLOBAL_RANDOM_SEED", 7)))),
        "n_estimators": int(effective.get("n_estimators", DEFAULT_SETTINGS.get("n_estimators", 1000))),
        "learning_rate": float(effective.get("learning_rate", DEFAULT_SETTINGS.get("learning_rate", 0.01))),
        "max_depth": int(effective.get("max_depth", DEFAULT_SETTINGS.get("max_depth", 6))),
        "model_subsample": float(effective.get("model_subsample", DEFAULT_SETTINGS.get("model_subsample", 0.8))),
        "run_permutation_test": strict_bool(effective.get("run_permutation_test", False)),
        "n_permutations": int(effective.get("n_permutations", 0) or 0) if strict_bool(effective.get("run_permutation_test", False)) else 0,
    }


def _build_plan_tasks(*, h_hits: Sequence[int], n_splits: int, n_permutations: int) -> List[Dict[str, Any]]:
    """Create one task per real coverage round and requested permutation round."""
    task_rows: List[Dict[str, Any]] = []
    task_id = 0
    for repeat_zero_idx, h_hit in enumerate(h_hits):
        repeat_idx = int(repeat_zero_idx + 1)
        task_rows.append(
            {
                "task_id": int(task_id),
                "task_key": f"real_h{int(h_hit):03d}",
                "family": "real",
                "permutation_idx": "",
                "h_hit": int(h_hit),
                "repeat_idx": int(repeat_idx),
                "n_splits": int(n_splits),
                "planned_model_count": int(n_splits),
            }
        )
        task_id += 1
        for perm_idx in range(int(max(0, n_permutations))):
            task_rows.append(
                {
                    "task_id": int(task_id),
                    "task_key": f"perm_p{int(perm_idx):03d}_h{int(h_hit):03d}",
                    "family": "perm",
                    "permutation_idx": int(perm_idx),
                    "h_hit": int(h_hit),
                    "repeat_idx": int(repeat_idx),
                    "n_splits": int(n_splits),
                    "planned_model_count": int(n_splits),
                }
            )
            task_id += 1
    return task_rows


def _assign_tasks_to_shards(task_rows: Sequence[Mapping[str, Any]], num_shards: int) -> Tuple[List[Dict[str, Any]], int]:
    """Distribute planned repeat tasks across no more shards than tasks."""
    requested = max(1, int(num_shards))
    task_count = int(len(task_rows))
    active_num_shards = max(1, min(requested, task_count if task_count > 0 else 1))
    assigned: List[Dict[str, Any]] = []
    for idx, row in enumerate(task_rows):
        shard_index = int(idx % active_num_shards)
        new_row = dict(row)
        new_row["shard_index"] = int(shard_index)
        new_row["shard_label"] = _format_shard_label(shard_index, active_num_shards)
        assigned.append(new_row)
    return assigned, active_num_shards


def _prepare_context(
    *,
    loaded: Mapping[str, Any],
    baseline_settings: Mapping[str, Any],
    runtime_num_workers: int,
    run_fsea_analysis: Optional[bool],
) -> Dict[str, Any]:
    """Validate ordered rows and features and prepare binary baseline arrays."""
    context_loaded = dict(loaded["context"])
    training_group_a = str(loaded["training_group_a"])
    training_group_b = str(loaded["training_group_b"])
    n_splits = int(loaded["n_splits"])
    X_train_full = context_loaded["X_train_full"]
    df_train = context_loaded["df_for_training"].copy()
    df_all = context_loaded["df_for_shap_and_agg"].copy()
    X_shap_full = context_loaded["X_shap_full"]
    fold_plan_records = dict(context_loaded["fold_plan_records"])

    feature_names = [str(c) for c in list(X_train_full.columns)]
    X_np = _resolve_feature_matrix(X_train_full, feature_names)
    X_all_np = _resolve_feature_matrix(X_shap_full, feature_names)
    X_np32 = np.ascontiguousarray(np.asarray(X_np, dtype=np.float32))
    X_all_np32 = np.ascontiguousarray(np.asarray(X_all_np, dtype=np.float32))

    y = _make_group_labels_binary(df_train, training_group_a, training_group_b)
    y_original_labels = df_train["group"].astype(str).to_numpy(copy=False)
    le = context_loaded.get("le")
    if le is None:
        le = FixedBinaryLabelEncoder(class0=str(training_group_a), class1=str(training_group_b))
    resolved_num_classes = int(context_loaded.get("num_classes", 2) or 2)
    if resolved_num_classes != 2:
        raise ValueError(f"Plain XGBoost staged baseline expects binary classification, got num_classes={resolved_num_classes!r}.")

    train_to_all_positions = _build_training_to_all_positions(df_train, df_all)
    all_groups = [str(g) for g in pd.unique(df_all["group"].astype(str))]
    training_groups = {str(training_group_a), str(training_group_b)}
    ordered_stability_groups = [str(training_group_a), str(training_group_b)] + [g for g in all_groups if g not in training_groups]

    settings_map = dict(baseline_settings)
    settings_map["num_workers"] = int(max(1, runtime_num_workers))
    if run_fsea_analysis is not None:
        settings_map["run_fsea_analysis"] = bool(run_fsea_analysis)

    return {
        "analysis_folder_path": str(loaded["analysis_folder_path"]),
        "training_group_a": str(training_group_a),
        "training_group_b": str(training_group_b),
        "n_splits": int(n_splits),
        "random_state": int(baseline_settings["GLOBAL_RANDOM_SEED"]),
        "settings_map": settings_map,
        "run_permutation_test": bool(baseline_settings["run_permutation_test"]),
        "n_permutations": int(baseline_settings["n_permutations"]),
        "num_workers": int(max(1, runtime_num_workers)),
        "n_estimators": int(baseline_settings["n_estimators"]),
        "learning_rate": float(baseline_settings["learning_rate"]),
        "max_depth": int(baseline_settings["max_depth"]),
        "model_subsample": float(baseline_settings["model_subsample"]),
        "feature_names": feature_names,
        "X_np32": X_np32,
        "X_all_np32": X_all_np32,
        "y": np.asarray(y, dtype=np.int32),
        "y_original_labels": np.asarray(y_original_labels),
        "df_train": df_train,
        "df_all": df_all,
        "le": le,
        "num_classes": int(resolved_num_classes),
        "train_to_all_positions": np.asarray(train_to_all_positions, dtype=np.int64),
        "n_train": int(len(df_train)),
        "n_all": int(len(df_all)),
        "n_feats": int(len(feature_names)),
        "out_dir": _plain_xgb_root(str(loaded["analysis_folder_path"])),
        "fold_plan_records": fold_plan_records,
        "sorted_h_hits": sorted(int(h) for h in list(fold_plan_records.keys())),
        "all_groups": all_groups,
        "training_groups": training_groups,
        "ordered_stability_groups": ordered_stability_groups,
    }


def _init_real_partial(context: Mapping[str, Any]) -> Dict[str, Any]:
    """Allocate real-repeat prediction and SHAP accumulators with frozen dimensions."""
    n_train = int(context["n_train"])
    n_all = int(context["n_all"])
    n_feats = int(context["n_feats"])
    return {
        "oof_margin_sum": np.zeros(n_train, dtype=np.float64),
        "oof_count": np.zeros(n_train, dtype=np.int32),
        "insample_margin_sum": np.zeros(n_train, dtype=np.float64),
        "insample_count": np.zeros(n_train, dtype=np.int32),
        "all_margin_sum": np.zeros(n_all, dtype=np.float64),
        "all_model_count": 0,
        "oof_shap_sum": np.zeros((n_train, n_feats), dtype=np.float32),
        "insample_shap_sum": np.zeros((n_train, n_feats), dtype=np.float32),
        "all_shap_sum": np.zeros((n_all, n_feats), dtype=np.float32),
        "fold_rows": [],
        "repeat_rows": [],
        "repeat_global_abs": [],
        "repeat_group_signed": {g: [] for g in list(context["ordered_stability_groups"])},
        "repeat_group_abs": {g: [] for g in list(context["ordered_stability_groups"])},
        "actual_outer_splits": None,
    }


def _init_perm_partial(context: Mapping[str, Any], *, permutation_idx: int) -> Dict[str, Any]:
    """Allocate prediction and SHAP accumulators for one permutation."""
    n_train = int(context["n_train"])
    n_all = int(context["n_all"])
    n_feats = int(context["n_feats"])
    return {
        "permutation_idx": int(permutation_idx),
        "oof_margin_sum": np.zeros(n_train, dtype=np.float64),
        "oof_count": np.zeros(n_train, dtype=np.int32),
        "insample_margin_sum": np.zeros(n_train, dtype=np.float64),
        "insample_count": np.zeros(n_train, dtype=np.int32),
        "all_margin_sum": np.zeros(n_all, dtype=np.float64),
        "all_model_count": 0,
        "oof_shap_sum": np.zeros((n_train, n_feats), dtype=np.float32),
        "insample_shap_sum": np.zeros((n_train, n_feats), dtype=np.float32),
        "all_shap_sum": np.zeros((n_all, n_feats), dtype=np.float32),
        "fold_rows": [],
        "actual_outer_splits": None,
    }


def _merge_real_partial(total: MutableMapping[str, Any], part: Mapping[str, Any]) -> None:
    """Accumulate compatible real-repeat sums, counts, and performance records."""
    for key in [
        "oof_margin_sum",
        "oof_count",
        "insample_margin_sum",
        "insample_count",
        "all_margin_sum",
        "oof_shap_sum",
        "insample_shap_sum",
        "all_shap_sum",
    ]:
        total[key] += part[key]
    total["all_model_count"] = int(total.get("all_model_count", 0)) + int(part.get("all_model_count", 0))
    actual = part.get("actual_outer_splits")
    if actual is not None and int(actual) > 0:
        if total.get("actual_outer_splits") is None:
            total["actual_outer_splits"] = int(actual)
        elif int(total["actual_outer_splits"]) != int(actual):
            raise ValueError(
                f"Inconsistent actual_outer_splits across plain-XGBoost shards: {total['actual_outer_splits']} vs {actual}."
            )
    total["fold_rows"].extend(list(part.get("fold_rows") or []))
    total["repeat_rows"].extend(list(part.get("repeat_rows") or []))
    total["repeat_global_abs"].extend(list(part.get("repeat_global_abs") or []))
    for group_name in list(total["repeat_group_signed"].keys()):
        total["repeat_group_signed"][group_name].extend(list((part.get("repeat_group_signed") or {}).get(group_name) or []))
        total["repeat_group_abs"][group_name].extend(list((part.get("repeat_group_abs") or {}).get(group_name) or []))


def _merge_perm_partial(total: MutableMapping[str, Any], part: Mapping[str, Any]) -> None:
    """Accumulate compatible permutation sums, counts, and performance records."""
    for key in [
        "oof_margin_sum",
        "oof_count",
        "insample_margin_sum",
        "insample_count",
        "all_margin_sum",
        "oof_shap_sum",
        "insample_shap_sum",
        "all_shap_sum",
    ]:
        total[key] += part[key]
    total["all_model_count"] = int(total.get("all_model_count", 0)) + int(part.get("all_model_count", 0))
    actual = part.get("actual_outer_splits")
    if actual is not None and int(actual) > 0:
        if total.get("actual_outer_splits") is None:
            total["actual_outer_splits"] = int(actual)
        elif int(total["actual_outer_splits"]) != int(actual):
            raise ValueError(
                f"Inconsistent actual_outer_splits across plain-XGBoost permuted shards: {total['actual_outer_splits']} vs {actual}."
            )
    total["fold_rows"].extend(list(part.get("fold_rows") or []))


def _run_real_repeat_task(
    *,
    context: Mapping[str, Any],
    h_hit: int,
    repeat_idx: int,
    started_at: float,
    log_prefix: str,
) -> Dict[str, Any]:
    """Train every frozen fold in one real repeat and collect numerical partials."""
    fold_plan_records = dict(context["fold_plan_records"])
    fold_indices_round = _build_real_fold_indices_local(fold_plan_records[int(h_hit)])
    n_splits = int(context["n_splits"])
    if int(len(fold_indices_round)) != int(n_splits):
        raise ValueError(
            f"Plain XGBoost staged repeat task h_hit={h_hit} expected n_splits={n_splits}, observed {len(fold_indices_round)}."
        )

    n_train = int(context["n_train"])
    n_all = int(context["n_all"])
    n_feats = int(context["n_feats"])
    X_np32 = np.asarray(context["X_np32"], dtype=np.float32)
    X_all_np32 = np.asarray(context["X_all_np32"], dtype=np.float32)
    y = np.asarray(context["y"], dtype=np.int32)
    df_train = context["df_train"]
    df_all = context["df_all"]
    training_groups = set(context["training_groups"])
    ordered_stability_groups = list(context["ordered_stability_groups"])
    train_to_all_positions = np.asarray(context["train_to_all_positions"], dtype=np.int64)

    _baseline_log(started_at, f"{log_prefix}: starting real h_hit={int(h_hit)} ({int(repeat_idx)}/{int(len(context['sorted_h_hits']))})")

    payload = {
        "oof_margin_sum": np.zeros(n_train, dtype=np.float64),
        "oof_count": np.zeros(n_train, dtype=np.int32),
        "insample_margin_sum": np.zeros(n_train, dtype=np.float64),
        "insample_count": np.zeros(n_train, dtype=np.int32),
        "all_margin_sum": np.zeros(n_all, dtype=np.float64),
        "all_model_count": 0,
        "oof_shap_sum": np.zeros((n_train, n_feats), dtype=np.float32),
        "insample_shap_sum": np.zeros((n_train, n_feats), dtype=np.float32),
        "all_shap_sum": np.zeros((n_all, n_feats), dtype=np.float32),
        "fold_rows": [],
        "repeat_rows": [],
        "repeat_global_abs": [],
        "repeat_group_signed": {g: [] for g in ordered_stability_groups},
        "repeat_group_abs": {g: [] for g in ordered_stability_groups},
        "actual_outer_splits": int(len(fold_indices_round)),
    }

    repeat_oof_margin = np.full(n_train, np.nan, dtype=np.float64)
    repeat_oof_count = np.zeros(n_train, dtype=np.int32)
    repeat_oof_shap_sum = np.zeros((n_train, n_feats), dtype=np.float32)
    repeat_oof_shap_count = np.zeros(n_train, dtype=np.int32)
    repeat_all_shap_sum = np.zeros((n_all, n_feats), dtype=np.float32)
    repeat_all_model_count = 0

    for fold_zero_idx, (train_idx, test_idx) in enumerate(fold_indices_round):
        fold_idx = int(fold_zero_idx + 1)
        train_idx = np.asarray(train_idx, dtype=np.int64)
        test_idx = np.asarray(test_idx, dtype=np.int64)
        train_all_pos = train_to_all_positions[train_idx]
        test_all_pos = train_to_all_positions[test_idx]

        model = _fit_plain_xgboost_model(
            X_np32[train_idx],
            y[train_idx],
            n_estimators=int(context["n_estimators"]),
            learning_rate=float(context["learning_rate"]),
            max_depth=int(context["max_depth"]),
            subsample=float(context["model_subsample"]),
            random_state=int(context["random_state"] + int(h_hit) * 10000 + fold_idx),
            n_jobs=int(context["num_workers"]),
        )
        explainer = make_tree_explainer(model)
        margins_all = np.asarray(model.predict(X_all_np32, output_margin=True), dtype=np.float64)
        shap_all = _positive_class_shap(explainer, X_all_np32)

        oof_margin = margins_all[test_all_pos]
        insample_margin = margins_all[train_all_pos]
        payload["oof_margin_sum"][test_idx] += oof_margin
        payload["oof_count"][test_idx] += 1
        payload["insample_margin_sum"][train_idx] += insample_margin
        payload["insample_count"][train_idx] += 1
        payload["all_margin_sum"] += margins_all
        payload["all_model_count"] += 1
        payload["oof_shap_sum"][test_idx, :] += shap_all[test_all_pos, :]
        payload["insample_shap_sum"][train_idx, :] += shap_all[train_all_pos, :]
        payload["all_shap_sum"] += shap_all

        repeat_oof_margin[test_idx] = oof_margin
        repeat_oof_count[test_idx] += 1
        repeat_oof_shap_sum[test_idx, :] += shap_all[test_all_pos, :]
        repeat_oof_shap_count[test_idx] += 1
        repeat_all_shap_sum += shap_all
        repeat_all_model_count += 1

        test_metrics = _compute_binary_metrics(y[test_idx], _probabilities_from_logits(oof_margin))
        payload["fold_rows"].append(
            {
                "repeat_idx": int(repeat_idx),
                "h_hit": int(h_hit),
                "fold_idx": int(fold_idx),
                "n_train": int(len(train_idx)),
                "n_test": int(len(test_idx)),
                "n_estimators": int(context["n_estimators"]),
                "learning_rate": float(context["learning_rate"]),
                "max_depth": int(context["max_depth"]),
                "subsample": float(context["model_subsample"]),
                "accuracy_test": float(test_metrics.get("accuracy", np.nan)),
                "accuracy_train": np.nan,
                **test_metrics,
            }
        )
        del explainer
        del model

    if np.any(repeat_oof_count != 1):
        raise ValueError(
            f"Plain XGBoost staged repeat task OOF bookkeeping failed in h_hit={h_hit}."
        )
    if np.any(repeat_oof_shap_count != 1):
        raise ValueError(
            f"Plain XGBoost staged repeat task OOF SHAP bookkeeping failed in h_hit={h_hit}."
        )
    if int(repeat_all_model_count) != int(len(fold_indices_round)):
        raise ValueError(
            f"Plain XGBoost staged repeat task all-groups SHAP bookkeeping failed in h_hit={h_hit}."
        )

    repeat_oof_probs = _probabilities_from_logits(repeat_oof_margin)
    repeat_metrics = _compute_binary_metrics(y, repeat_oof_probs)
    payload["repeat_rows"].append(
        {
            "repeat_idx": int(repeat_idx),
            "h_hit": int(h_hit),
            "n_outer_folds": int(len(fold_indices_round)),
            **{f"oof_{k}": v for k, v in repeat_metrics.items()},
        }
    )

    repeat_oof_shap = (repeat_oof_shap_sum / np.maximum(repeat_oof_shap_count[:, None], 1)).astype(np.float32)
    repeat_all_shap = (repeat_all_shap_sum / float(max(repeat_all_model_count, 1))).astype(np.float32)
    payload["repeat_global_abs"].append(np.mean(np.abs(repeat_oof_shap), axis=0).astype(np.float32))
    for group_name in ordered_stability_groups:
        if group_name in training_groups:
            mask = df_train["group"].astype(str).to_numpy(copy=False) == str(group_name)
            group_vals = repeat_oof_shap[mask, :]
        else:
            mask = df_all["group"].astype(str).to_numpy(copy=False) == str(group_name)
            group_vals = repeat_all_shap[mask, :]
        if group_vals.shape[0] == 0:
            continue
        payload["repeat_group_signed"][group_name].append(np.mean(group_vals, axis=0).astype(np.float32))
        payload["repeat_group_abs"][group_name].append(np.mean(np.abs(group_vals), axis=0).astype(np.float32))

    return payload


def _run_permuted_repeat_task(
    *,
    context: Mapping[str, Any],
    permutation_idx: int,
    h_hit: int,
    repeat_idx: int,
    started_at: float,
    log_prefix: str,
) -> Dict[str, Any]:
    """Train one permutation repeat and collect results in the original label frame."""
    fold_plan_records = dict(context["fold_plan_records"])
    fold_indices_round = _build_real_fold_indices_local(fold_plan_records[int(h_hit)])
    n_splits = int(context["n_splits"])
    if int(len(fold_indices_round)) != int(n_splits):
        raise ValueError(
            f"Plain XGBoost staged permutation task h_hit={h_hit} expected n_splits={n_splits}, observed {len(fold_indices_round)}."
        )

    n_train = int(context["n_train"])
    n_all = int(context["n_all"])
    n_feats = int(context["n_feats"])
    X_np32 = np.asarray(context["X_np32"], dtype=np.float32)
    X_all_np32 = np.asarray(context["X_all_np32"], dtype=np.float32)
    y = np.asarray(context["y"], dtype=np.int32)
    y_original_labels = np.asarray(context["y_original_labels"])
    df_train = context["df_train"]
    train_to_all_positions = np.asarray(context["train_to_all_positions"], dtype=np.int64)

    _baseline_log(started_at, f"{log_prefix}: starting perm={int(permutation_idx)} h_hit={int(h_hit)} ({int(repeat_idx)}/{int(len(context['sorted_h_hits']))})")

    payload = _init_perm_partial(context, permutation_idx=int(permutation_idx))
    payload["actual_outer_splits"] = int(len(fold_indices_round))

    for fold_zero_idx, (train_idx, test_idx) in enumerate(fold_indices_round):
        fold_idx = int(fold_zero_idx + 1)
        train_idx = np.asarray(train_idx, dtype=np.int64)
        test_idx = np.asarray(test_idx, dtype=np.int64)
        train_all_pos = train_to_all_positions[train_idx]
        test_all_pos = train_to_all_positions[test_idx]
        y_permuted_full = np.asarray(
            _generate_sex_class_stratified_permutation(
                y_original=y_original_labels,
                df_with_sex=df_train,
                train_indices=train_idx,
                test_indices=test_idx,
                permutation_seed=int(context["random_state"] + (int(permutation_idx) + 1) * 1_000_000 + int(h_hit) * 10_000 + fold_idx),
            ),
            dtype=object,
        )
        y_train_permuted = np.asarray(np.asarray(y_permuted_full)[train_idx] == str(context["training_group_b"]), dtype=np.int32)
        model = _fit_plain_xgboost_model(
            X_np32[train_idx],
            y_train_permuted,
            n_estimators=int(context["n_estimators"]),
            learning_rate=float(context["learning_rate"]),
            max_depth=int(context["max_depth"]),
            subsample=float(context["model_subsample"]),
            random_state=int(context["random_state"] + (int(permutation_idx) + 1) * 1_000_000 + int(h_hit) * 10_000 + fold_idx),
            n_jobs=int(context["num_workers"]),
        )
        explainer = make_tree_explainer(model)
        margins_all = np.asarray(model.predict(X_all_np32, output_margin=True), dtype=np.float64)
        shap_all = _positive_class_shap(explainer, X_all_np32)
        margins_all_for_agg = np.asarray(margins_all, dtype=np.float64).copy()
        flip_mask_train = np.asarray(y_original_labels[train_idx] != y_permuted_full[train_idx], dtype=bool)
        if np.any(flip_mask_train):
            margins_all_for_agg[train_all_pos[flip_mask_train]] *= -1.0

        perm_oof_margin = margins_all_for_agg[test_all_pos]
        perm_insample_margin = margins_all_for_agg[train_all_pos]
        payload["oof_margin_sum"][test_idx] += perm_oof_margin
        payload["oof_count"][test_idx] += 1
        payload["insample_margin_sum"][train_idx] += perm_insample_margin
        payload["insample_count"][train_idx] += 1
        payload["all_margin_sum"] += margins_all_for_agg
        payload["all_model_count"] += 1
        payload["oof_shap_sum"][test_idx, :] += shap_all[test_all_pos, :]
        payload["insample_shap_sum"][train_idx, :] += shap_all[train_all_pos, :]
        payload["all_shap_sum"] += shap_all

        fold_metrics = _compute_binary_metrics(y[test_idx], _probabilities_from_logits(perm_oof_margin))
        payload["fold_rows"].append(
            {
                "permutation_idx": int(permutation_idx),
                "repeat_idx": int(repeat_idx),
                "h_hit": int(h_hit),
                "fold_idx": int(fold_idx),
                "n_train": int(len(train_idx)),
                "n_test": int(len(test_idx)),
                "accuracy_test": float(fold_metrics.get("accuracy", np.nan)),
                "accuracy_train": np.nan,
                **fold_metrics,
            }
        )
        del explainer
        del model

    return payload


def _write_real_partial_files(shard_dir: str, partial: Mapping[str, Any], ordered_groups: Sequence[str], n_feats: int) -> None:
    """Persist real-repeat sums, counts, stability arrays, and performance records."""
    np.savez_compressed(
        os.path.join(shard_dir, REAL_PARTIALS_FILENAME),
        oof_margin_sum=np.asarray(partial["oof_margin_sum"], dtype=np.float64),
        oof_count=np.asarray(partial["oof_count"], dtype=np.int32),
        insample_margin_sum=np.asarray(partial["insample_margin_sum"], dtype=np.float64),
        insample_count=np.asarray(partial["insample_count"], dtype=np.int32),
        all_margin_sum=np.asarray(partial["all_margin_sum"], dtype=np.float64),
        all_model_count=np.asarray([int(partial.get("all_model_count", 0))], dtype=np.int32),
        oof_shap_sum=np.asarray(partial["oof_shap_sum"], dtype=np.float32),
        insample_shap_sum=np.asarray(partial["insample_shap_sum"], dtype=np.float32),
        all_shap_sum=np.asarray(partial["all_shap_sum"], dtype=np.float32),
        actual_outer_splits=np.asarray([int(partial.get("actual_outer_splits") or 0)], dtype=np.int32),
    )
    _write_jsonl(list(partial.get("fold_rows") or []), os.path.join(shard_dir, REAL_FOLD_ROWS_FILENAME))
    _write_jsonl(list(partial.get("repeat_rows") or []), os.path.join(shard_dir, REAL_REPEAT_ROWS_FILENAME))

    stability_payload: Dict[str, Any] = {
        "group_names": np.asarray(list(ordered_groups), dtype=object),
        "repeat_global_abs": (
            np.vstack([np.asarray(v, dtype=np.float32).reshape(1, -1) for v in list(partial.get("repeat_global_abs") or [])])
            if list(partial.get("repeat_global_abs") or [])
            else np.zeros((0, int(n_feats)), dtype=np.float32)
        ),
    }
    signed_map = dict(partial.get("repeat_group_signed") or {})
    abs_map = dict(partial.get("repeat_group_abs") or {})
    for idx, group_name in enumerate(list(ordered_groups)):
        signed_vals = list(signed_map.get(group_name) or [])
        abs_vals = list(abs_map.get(group_name) or [])
        stability_payload[f"group_signed_{idx}"] = (
            np.vstack([np.asarray(v, dtype=np.float32).reshape(1, -1) for v in signed_vals])
            if signed_vals else np.zeros((0, int(n_feats)), dtype=np.float32)
        )
        stability_payload[f"group_abs_{idx}"] = (
            np.vstack([np.asarray(v, dtype=np.float32).reshape(1, -1) for v in abs_vals])
            if abs_vals else np.zeros((0, int(n_feats)), dtype=np.float32)
        )
    np.savez_compressed(os.path.join(shard_dir, REAL_STABILITY_FILENAME), **stability_payload)


def _load_real_partial_files(shard_dir: str, ordered_groups: Sequence[str]) -> Dict[str, Any]:
    """Load real-repeat numerical partials and ordered group stability arrays."""
    npz = np.load(os.path.join(shard_dir, REAL_PARTIALS_FILENAME), allow_pickle=True)
    payload = {
        "oof_margin_sum": np.asarray(npz["oof_margin_sum"], dtype=np.float64),
        "oof_count": np.asarray(npz["oof_count"], dtype=np.int32),
        "insample_margin_sum": np.asarray(npz["insample_margin_sum"], dtype=np.float64),
        "insample_count": np.asarray(npz["insample_count"], dtype=np.int32),
        "all_margin_sum": np.asarray(npz["all_margin_sum"], dtype=np.float64),
        "all_model_count": int(np.asarray(npz["all_model_count"]).reshape(-1)[0]),
        "oof_shap_sum": np.asarray(npz["oof_shap_sum"], dtype=np.float32),
        "insample_shap_sum": np.asarray(npz["insample_shap_sum"], dtype=np.float32),
        "all_shap_sum": np.asarray(npz["all_shap_sum"], dtype=np.float32),
        "actual_outer_splits": int(np.asarray(npz["actual_outer_splits"]).reshape(-1)[0]),
        "fold_rows": _read_jsonl(os.path.join(shard_dir, REAL_FOLD_ROWS_FILENAME)),
        "repeat_rows": _read_jsonl(os.path.join(shard_dir, REAL_REPEAT_ROWS_FILENAME)),
        "repeat_global_abs": [],
        "repeat_group_signed": {g: [] for g in list(ordered_groups)},
        "repeat_group_abs": {g: [] for g in list(ordered_groups)},
    }
    stability = np.load(os.path.join(shard_dir, REAL_STABILITY_FILENAME), allow_pickle=True)
    repeat_global_abs = np.asarray(stability["repeat_global_abs"], dtype=np.float32)
    payload["repeat_global_abs"] = [np.asarray(row, dtype=np.float32) for row in repeat_global_abs]
    group_names = [str(x) for x in list(np.asarray(stability["group_names"], dtype=object))]
    for idx, group_name in enumerate(group_names):
        signed = np.asarray(stability[f"group_signed_{idx}"], dtype=np.float32)
        abs_vals = np.asarray(stability[f"group_abs_{idx}"], dtype=np.float32)
        payload["repeat_group_signed"][group_name] = [np.asarray(row, dtype=np.float32) for row in signed]
        payload["repeat_group_abs"][group_name] = [np.asarray(row, dtype=np.float32) for row in abs_vals]
    return payload


def _write_perm_partial_files(shard_dir: str, partial: Mapping[str, Any]) -> None:
    """Persist one permutation's numerical partials and fold-performance records."""
    perm_idx = int(partial["permutation_idx"])
    np.savez_compressed(
        os.path.join(shard_dir, PERM_PARTIALS_FILENAME_TEMPLATE.format(perm_idx=perm_idx)),
        oof_margin_sum=np.asarray(partial["oof_margin_sum"], dtype=np.float64),
        oof_count=np.asarray(partial["oof_count"], dtype=np.int32),
        insample_margin_sum=np.asarray(partial["insample_margin_sum"], dtype=np.float64),
        insample_count=np.asarray(partial["insample_count"], dtype=np.int32),
        all_margin_sum=np.asarray(partial["all_margin_sum"], dtype=np.float64),
        all_model_count=np.asarray([int(partial.get("all_model_count", 0))], dtype=np.int32),
        oof_shap_sum=np.asarray(partial["oof_shap_sum"], dtype=np.float32),
        insample_shap_sum=np.asarray(partial["insample_shap_sum"], dtype=np.float32),
        all_shap_sum=np.asarray(partial["all_shap_sum"], dtype=np.float32),
        actual_outer_splits=np.asarray([int(partial.get("actual_outer_splits") or 0)], dtype=np.int32),
        permutation_idx=np.asarray([perm_idx], dtype=np.int32),
    )
    _write_jsonl(list(partial.get("fold_rows") or []), os.path.join(shard_dir, PERM_FOLD_ROWS_FILENAME_TEMPLATE.format(perm_idx=perm_idx)))


def _load_perm_partial_files(shard_dir: str, perm_idx: int) -> Dict[str, Any]:
    """Load one permutation's numerical partials and fold-performance records."""
    npz = np.load(os.path.join(shard_dir, PERM_PARTIALS_FILENAME_TEMPLATE.format(perm_idx=perm_idx)), allow_pickle=True)
    return {
        "permutation_idx": int(np.asarray(npz["permutation_idx"]).reshape(-1)[0]),
        "oof_margin_sum": np.asarray(npz["oof_margin_sum"], dtype=np.float64),
        "oof_count": np.asarray(npz["oof_count"], dtype=np.int32),
        "insample_margin_sum": np.asarray(npz["insample_margin_sum"], dtype=np.float64),
        "insample_count": np.asarray(npz["insample_count"], dtype=np.int32),
        "all_margin_sum": np.asarray(npz["all_margin_sum"], dtype=np.float64),
        "all_model_count": int(np.asarray(npz["all_model_count"]).reshape(-1)[0]),
        "oof_shap_sum": np.asarray(npz["oof_shap_sum"], dtype=np.float32),
        "insample_shap_sum": np.asarray(npz["insample_shap_sum"], dtype=np.float32),
        "all_shap_sum": np.asarray(npz["all_shap_sum"], dtype=np.float32),
        "actual_outer_splits": int(np.asarray(npz["actual_outer_splits"]).reshape(-1)[0]),
        "fold_rows": _read_jsonl(os.path.join(shard_dir, PERM_FOLD_ROWS_FILENAME_TEMPLATE.format(perm_idx=perm_idx))),
    }


def _finalize_outputs_from_aggregates(
    *,
    context: Mapping[str, Any],
    real_partial: Mapping[str, Any],
    perm_partials: Mapping[int, Mapping[str, Any]],
    run_fsea_analysis: bool,
    started_at: float,
) -> str:
    """Convert merged repeat partials into baseline predictions, SHAP reports, and
    optional FSEA.
    """
    out_dir = str(context["out_dir"])
    os.makedirs(out_dir, exist_ok=True)
    _cleanup_stale_permutation_outputs(out_dir)

    df_train = context["df_train"]
    df_all = context["df_all"]
    feature_names = list(context["feature_names"])
    y = np.asarray(context["y"], dtype=np.int32)
    train_to_all_positions = np.asarray(context["train_to_all_positions"], dtype=np.int64)
    training_group_a = str(context["training_group_a"])
    training_group_b = str(context["training_group_b"])
    le = context["le"]
    resolved_num_classes = int(context["num_classes"])
    n_repeats = int(len(context["sorted_h_hits"]))
    actual_outer_splits = int(real_partial.get("actual_outer_splits") or 0)
    if actual_outer_splits <= 0:
        raise ValueError("Plain XGBoost staged merge did not recover a valid outer-fold count.")

    oof_count = np.asarray(real_partial["oof_count"], dtype=np.int32)
    insample_count = np.asarray(real_partial["insample_count"], dtype=np.int32)
    all_model_count = int(real_partial["all_model_count"])
    expected_insample_count = int(n_repeats) * max(int(actual_outer_splits) - 1, 0)
    if np.any(oof_count != int(n_repeats)):
        raise ValueError(
            f"Plain XGBoost staged merge OOF bookkeeping failed. Expected {n_repeats} OOF predictions per training row."
        )
    if np.any(insample_count != expected_insample_count):
        raise ValueError(
            f"Plain XGBoost staged merge in-sample bookkeeping failed. Expected {expected_insample_count} in-sample predictions per training row."
        )
    if int(all_model_count) != int(n_repeats) * int(actual_outer_splits):
        raise ValueError(
            f"Plain XGBoost staged merge all-model accumulator mismatch. Expected {int(n_repeats) * int(actual_outer_splits)}, observed {all_model_count}."
        )

    oof_margin = np.asarray(real_partial["oof_margin_sum"], dtype=np.float64) / np.maximum(oof_count, 1)
    insample_margin = np.asarray(real_partial["insample_margin_sum"], dtype=np.float64) / np.maximum(insample_count, 1)
    all_margin = np.asarray(real_partial["all_margin_sum"], dtype=np.float64) / float(max(all_model_count, 1))
    oof_prob_group_b = _probabilities_from_logits(oof_margin)
    insample_prob_group_b = _probabilities_from_logits(insample_margin)
    all_prob_group_b = _probabilities_from_logits(all_margin)
    oof_shap_values = (np.asarray(real_partial["oof_shap_sum"], dtype=np.float64) / np.maximum(oof_count[:, None], 1)).astype(np.float64)
    insample_shap_values = (np.asarray(real_partial["insample_shap_sum"], dtype=np.float64) / np.maximum(insample_count[:, None], 1)).astype(np.float64)
    all_groups_avg_shap_values = (np.asarray(real_partial["all_shap_sum"], dtype=np.float64) / float(max(all_model_count, 1))).astype(np.float64)

    fold_rows = sorted(
        [dict(r) for r in list(real_partial.get("fold_rows") or [])],
        key=lambda row: (int(row.get("repeat_idx", 0) or 0), int(row.get("fold_idx", 0) or 0)),
    )
    repeat_rows = sorted(
        [dict(r) for r in list(real_partial.get("repeat_rows") or [])],
        key=lambda row: (int(row.get("repeat_idx", 0) or 0), int(row.get("h_hit", 0) or 0)),
    )

    permuted_global_importances: List[np.ndarray] = []
    permuted_predictions_views: List[Tuple[int, pd.DataFrame]] = []
    for perm_idx in sorted(int(k) for k in perm_partials.keys()):
        perm_partial = perm_partials[int(perm_idx)]
        perm_oof_count = np.asarray(perm_partial["oof_count"], dtype=np.int32)
        perm_insample_count = np.asarray(perm_partial["insample_count"], dtype=np.int32)
        perm_all_model_count = int(perm_partial["all_model_count"])
        if np.any(perm_oof_count != int(n_repeats)):
            raise ValueError(
                f"Plain XGBoost staged permuted merge OOF bookkeeping failed for permutation {perm_idx}."
            )
        if np.any(perm_insample_count != expected_insample_count):
            raise ValueError(
                f"Plain XGBoost staged permuted merge in-sample bookkeeping failed for permutation {perm_idx}."
            )
        if int(perm_all_model_count) != int(n_repeats) * int(actual_outer_splits):
            raise ValueError(
                f"Plain XGBoost staged permuted merge all-model count mismatch for permutation {perm_idx}."
            )
        perm_oof_margin = np.asarray(perm_partial["oof_margin_sum"], dtype=np.float64) / np.maximum(perm_oof_count, 1)
        perm_insample_margin = np.asarray(perm_partial["insample_margin_sum"], dtype=np.float64) / np.maximum(perm_insample_count, 1)
        perm_all_margin = np.asarray(perm_partial["all_margin_sum"], dtype=np.float64) / float(max(perm_all_model_count, 1))
        perm_oof_prob_group_b = _probabilities_from_logits(perm_oof_margin)
        perm_insample_prob_group_b = _probabilities_from_logits(perm_insample_margin)
        perm_all_prob_group_b = _probabilities_from_logits(perm_all_margin)
        perm_oof_shap = (np.asarray(perm_partial["oof_shap_sum"], dtype=np.float64) / np.maximum(perm_oof_count[:, None], 1)).astype(np.float64)
        perm_insample_shap = (np.asarray(perm_partial["insample_shap_sum"], dtype=np.float64) / np.maximum(perm_insample_count[:, None], 1)).astype(np.float64)
        perm_all_shap = (np.asarray(perm_partial["all_shap_sum"], dtype=np.float64) / float(max(perm_all_model_count, 1))).astype(np.float64)
        permuted_global_importances.append(np.mean(np.abs(perm_oof_shap), axis=0).astype(np.float64))
        perm_fold_rows = sorted(
            [dict(r) for r in list(perm_partial.get("fold_rows") or [])],
            key=lambda row: (
                int(row.get("permutation_idx", 0) or 0),
                int(row.get("repeat_idx", 0) or 0),
                int(row.get("fold_idx", 0) or 0),
            ),
        )
        perm_predictions_df = _build_predictions_dataframe(
            df_for_training=df_train,
            df_for_shap_and_agg=df_all,
            train_to_all_positions=train_to_all_positions,
            training_group_a=training_group_a,
            training_group_b=training_group_b,
            oof_prob_group_b=perm_oof_prob_group_b,
            insample_prob_group_b=perm_insample_prob_group_b,
            all_prob_group_b=perm_all_prob_group_b,
        )
        permuted_predictions_views.append((int(perm_idx), perm_predictions_df))
        _aggregate_and_summarize_permuted_results(
            df=df_all,
            selected_features=feature_names,
            all_permuted_performance=perm_fold_rows,
            all_permuted_shap_values=None,
            le=le,
            paths={"analysis_folder": out_dir, "file_suffix": ""},
            permutation_idx=int(perm_idx),
            permuted_insample_shap=perm_insample_shap,
            permuted_oof_shap=perm_oof_shap,
            df_for_training=df_train,
            train_groups_list=[training_group_a, training_group_b],
            n_splits=int(context["n_splits"]),
            permuted_all_groups_avg_probas=np.column_stack([1.0 - perm_all_prob_group_b, perm_all_prob_group_b]).astype(np.float32),
            permuted_all_groups_avg_shap_values=perm_all_shap,
            df_for_shap_and_agg=df_all,
            permuted_insample_probas=np.column_stack([1.0 - perm_insample_prob_group_b, perm_insample_prob_group_b]).astype(np.float32),
            permuted_oof_probas=np.column_stack([1.0 - perm_oof_prob_group_b, perm_oof_prob_group_b]).astype(np.float32),
            all_fold_indices=None,
        )
    perm_baseline = (
        np.mean(np.stack(permuted_global_importances, axis=0), axis=0).astype(np.float64)
        if permuted_global_importances
        else np.zeros((int(context["n_feats"]),), dtype=np.float64)
    )

    _baseline_log(started_at, "writing plain-XGBoost staged baseline outputs")

    fold_perf_df = pd.DataFrame(fold_rows)
    repeat_perf_df = pd.DataFrame(repeat_rows)
    pooled_perf_df = pd.DataFrame([
        {
            "n_repeats": int(n_repeats),
            "outer_n_splits_requested": int(context["n_splits"]),
            "outer_n_splits_actual": int(actual_outer_splits),
            "n_outer_models": int(all_model_count),
            "oof_prediction_count_per_training_row": int(np.min(oof_count)) if len(oof_count) > 0 else 0,
            "insample_prediction_count_per_training_row": int(np.min(insample_count)) if len(insample_count) > 0 else 0,
            **{f"oof_{k}": v for k, v in _compute_binary_metrics(y, oof_prob_group_b).items()},
            **{f"insample_{k}": v for k, v in _compute_binary_metrics(y, insample_prob_group_b).items()},
        }
    ])

    base_predictions_df = _build_predictions_dataframe(
        df_for_training=df_train,
        df_for_shap_and_agg=df_all,
        train_to_all_positions=train_to_all_positions,
        training_group_a=training_group_a,
        training_group_b=training_group_b,
        oof_prob_group_b=oof_prob_group_b,
        insample_prob_group_b=insample_prob_group_b,
        all_prob_group_b=all_prob_group_b,
    )
    all_predictions_df = _merge_prediction_views(base_df=base_predictions_df, permuted_dfs=permuted_predictions_views)
    save_predictions_with_formatting(all_predictions_df, os.path.join(out_dir, PREDICTIONS_FULL_FILENAME))
    create_formatted_predictions_for_graphpad(
        base_predictions_df,
        training_group_a=training_group_a,
        training_group_b=training_group_b,
        analysis_folder=out_dir,
        file_suffix="",
        perm_idx=0,
    )

    summary_df, combined_df = _build_shap_summary_frames(
        feature_names=feature_names,
        df_for_training=df_train,
        df_for_shap_and_agg=df_all,
        training_group_a=training_group_a,
        training_group_b=training_group_b,
        oof_shap_values=oof_shap_values,
        insample_shap_values=insample_shap_values,
        all_groups_avg_shap_values=all_groups_avg_shap_values,
    )
    perm_map = {str(name): float(val) for name, val in zip(feature_names, perm_baseline.tolist())}
    if "Global_Importance" in summary_df.columns:
        summary_df["Global_Importance_Raw"] = summary_df["Global_Importance"].astype(float)
        summary_df["Global_Importance_PermBaseline"] = summary_df["feature_name"].map(perm_map).astype(float)
        summary_df["Global_Importance"] = np.clip(
            summary_df["Global_Importance_Raw"].astype(float) - summary_df["Global_Importance_PermBaseline"].astype(float),
            a_min=0.0,
            a_max=None,
        )
        summary_df["Global_Importance_Normalized"] = np.nan
        summary_df["Global_Importance_Cumulative"] = np.nan
    if "Global_Importance_OUTOFSAMPLE" in combined_df.columns:
        combined_df["Global_Importance_OUTOFSAMPLE_Raw"] = combined_df["Global_Importance_OUTOFSAMPLE"].astype(float)
        combined_df["Global_Importance_OUTOFSAMPLE_PermBaseline"] = combined_df["feature_name"].map(perm_map).astype(float)
        combined_df["Global_Importance_OUTOFSAMPLE"] = np.clip(
            combined_df["Global_Importance_OUTOFSAMPLE_Raw"].astype(float) - combined_df["Global_Importance_OUTOFSAMPLE_PermBaseline"].astype(float),
            a_min=0.0,
            a_max=None,
        )

    stability_df = _compute_across_repeat_shap_stability(
        feature_names=feature_names,
        repeat_global_abs=list(real_partial.get("repeat_global_abs") or []),
        repeat_group_signed=dict(real_partial.get("repeat_group_signed") or {}),
        repeat_group_abs=dict(real_partial.get("repeat_group_abs") or {}),
    )
    stability_path = os.path.join(out_dir, SHAP_STABILITY_FILENAME)
    stability_df.to_csv(stability_path, index=False)
    summary_df = pd.merge(summary_df, stability_df, on="feature_name", how="left")
    summary_df = _finalize_global_importance_columns(summary_df)
    if len(combined_df) > 0 and "Global_Importance_OUTOFSAMPLE" in combined_df.columns:
        combined_df = combined_df.sort_values(by=["Global_Importance_OUTOFSAMPLE", "feature_name"], ascending=[False, True], kind="mergesort").reset_index(drop=True)

    summary_path, _combined_path = _write_shap_outputs(out_dir=out_dir, summary_df=summary_df, combined_df=combined_df)
    _write_global_importance_ranking(out_dir=out_dir, summary_df=summary_df)

    out_paths = {"analysis_folder": out_dir, "file_suffix": ""}
    from .compat import _generate_kfold_shap_summaries, _generate_male_female_comparison

    outofsample_result = _generate_kfold_shap_summaries(
        shap_values=oof_shap_values,
        df=df_train,
        selected_features=feature_names,
        le=le,
        paths=out_paths,
        suffix="outofsample",
        train_groups=[training_group_a, training_group_b],
    )
    insample_result = _generate_kfold_shap_summaries(
        shap_values=insample_shap_values,
        df=df_train,
        selected_features=feature_names,
        le=le,
        paths=out_paths,
        suffix="insample",
        train_groups=[training_group_a, training_group_b],
    )
    if outofsample_result is not None and insample_result is not None:
        out_sheets, _ = outofsample_result
        in_sheets, _ = insample_result
        if out_sheets is not None and in_sheets is not None:
            combined_individual_sheets: Dict[str, pd.DataFrame] = {}
            for sheet_name, sheet_df in out_sheets.items():
                combined_individual_sheets[f"OutOfSample_{sheet_name}"[:31]] = sheet_df
            for sheet_name, sheet_df in in_sheets.items():
                combined_individual_sheets[f"InSample_{sheet_name}"[:31]] = sheet_df
            save_to_excel_sheets(combined_individual_sheets, os.path.join(out_dir, SHAP_PER_INDIVIDUAL_FILENAME))

    hybrid_shap = np.asarray(all_groups_avg_shap_values, dtype=np.float32).copy()
    hybrid_shap[np.asarray(train_to_all_positions, dtype=np.int64), :] = np.asarray(oof_shap_values, dtype=np.float32)
    _generate_male_female_comparison(
        df=df_all,
        selected_features=feature_names,
        all_shap_file_paths=[],
        le=le,
        paths=out_paths,
        avg_shap_values_override=hybrid_shap,
    )

    fold_perf_df.to_csv(os.path.join(out_dir, FOLD_PERFORMANCE_FILENAME), index=False)
    repeat_perf_df.to_csv(os.path.join(out_dir, REPEAT_PERFORMANCE_FILENAME), index=False)
    pooled_perf_df.to_csv(os.path.join(out_dir, POOLED_PERFORMANCE_FILENAME), index=False)

    config_payload = {
        "training_group_a": training_group_a,
        "training_group_b": training_group_b,
        "n_training_rows": int(len(df_train)),
        "n_all_group_rows": int(len(df_all)),
        "n_features": int(len(feature_names)),
        "outer_n_splits_requested": int(context["n_splits"]),
        "outer_n_splits_actual": int(actual_outer_splits),
        "n_repeats": int(n_repeats),
        "n_outer_models": int(all_model_count),
        "reuse_frozen_hhit_fold_plan": True,
        "n_estimators": int(context["n_estimators"]),
        "learning_rate": float(context["learning_rate"]),
        "max_depth": int(context["max_depth"]),
        "subsample": float(context["model_subsample"]),
        "colsample_bytree": 1.0,
        "random_state": int(context["random_state"]),
        "num_workers": int(context["num_workers"]),
        "run_permutation_test": bool(context["run_permutation_test"]),
        "n_permutations": int(context["n_permutations"]),
        "execution_mode": "staged_sharded",
        "outputs": {
            "fold_performance": FOLD_PERFORMANCE_FILENAME,
            "repeat_performance": REPEAT_PERFORMANCE_FILENAME,
            "pooled_performance": POOLED_PERFORMANCE_FILENAME,
            "predictions_full": PREDICTIONS_FULL_FILENAME,
            "predictions_graphpad": PREDICTIONS_GRAPHPAD_FILENAME,
            "shap_summary": SHAP_SUMMARY_FILENAME,
            "shap_insample_vs_outofsample": SHAP_COMBINED_FILENAME,
            "shap_per_individual": SHAP_PER_INDIVIDUAL_FILENAME,
            "shap_male_vs_female": SHAP_MALE_FEMALE_FILENAME,
            "shap_stability": SHAP_STABILITY_FILENAME,
            "global_importance_ranking": GLOBAL_IMPORTANCE_RANKING_FILENAME,
        },
    }
    with open(os.path.join(out_dir, CONFIG_FILENAME), "w", encoding="utf-8") as f:
        json.dump(config_payload, f, indent=2)

    training_groups = set(context["training_groups"])
    all_groups = list(context["all_groups"])
    with open(os.path.join(out_dir, MANIFEST_FILENAME), "w", encoding="utf-8") as f:
        json.dump(
            {
                "n_repeats": int(n_repeats),
                "n_features": int(len(feature_names)),
                "global_importance_formula": "max(0, Global_Importance_Raw - Global_Importance_PermBaseline)",
                "permutation_baseline_used": bool(perm_partials),
                "stability_source": "across_repeats",
                "training_groups": [training_group_a, training_group_b],
                "nontraining_groups": [g for g in all_groups if g not in training_groups],
                "execution_mode": "staged_sharded",
            },
            f,
            indent=2,
        )

    if bool(run_fsea_analysis):
        _baseline_log(started_at, "starting optional FSEA")
        _run_optional_plain_xgboost_fsea(
            out_dir=out_dir,
            settings_snapshot=context["settings_map"],
            oof_shap_values=oof_shap_values,
            all_groups_avg_shap_values=all_groups_avg_shap_values,
            df_for_training=df_train,
            df_for_shap_and_agg=df_all,
            feature_names=feature_names,
            training_group_a=training_group_a,
            training_group_b=training_group_b,
            le=le,
            num_classes=resolved_num_classes,
        )
        _baseline_log(started_at, "optional FSEA completed")
    _baseline_log(started_at, "plain-XGBoost staged merge complete")
    return out_dir


def _plan_plain_xgboost_baseline_shards(
    *,
    analysis_folder_path: str,
    num_shards: Optional[int] = None,
    settings_override: Optional[Mapping[str, Any]] = None,
) -> str:
    """Freeze the comparison-baseline settings and repeat assignments."""
    analysis_dir = _analysis_dir(analysis_folder_path)
    loaded = load_featureimp_baseline_context(analysis_dir)
    baseline_settings = _collect_baseline_settings(loaded, settings_override=settings_override)
    context = dict(loaded["context"])
    sorted_h_hits = sorted(int(h) for h in list((context.get("fold_plan_records") or {}).keys()))
    if not sorted_h_hits:
        raise ValueError("Plain XGBoost staged plan received an empty fold plan.")

    requested_num_shards = _current_num_shards(num_shards)
    task_rows = _build_plan_tasks(
        h_hits=sorted_h_hits,
        n_splits=int(loaded["n_splits"]),
        n_permutations=int(baseline_settings["n_permutations"] if baseline_settings["run_permutation_test"] else 0),
    )
    task_rows, active_num_shards = _assign_tasks_to_shards(task_rows, requested_num_shards)

    out_dir = _plain_xgb_root(analysis_dir)
    plan_dir = _plan_root(analysis_dir)
    shards_root = _shards_root(analysis_dir)
    if os.path.exists(_plan_manifest_path(analysis_dir)):
        raise FileExistsError("Baseline plans are immutable; use a new analysis folder.")
    if os.path.exists(plan_dir) and any(Path(plan_dir).iterdir()):
        raise FileExistsError("Incomplete baseline plan retained; use a new analysis folder.")
    os.makedirs(plan_dir, exist_ok=True)
    os.makedirs(shards_root, exist_ok=True)

    task_manifest = _task_manifest_path(analysis_dir)
    _write_task_rows(task_manifest, task_rows)
    for shard_index in range(int(active_num_shards)):
        shard_plan_dir = _shard_plan_dir(analysis_dir, shard_index, active_num_shards)
        os.makedirs(shard_plan_dir, exist_ok=True)
        shard_rows = [dict(row) for row in task_rows if int(row["shard_index"]) == int(shard_index)]
        _write_task_rows(os.path.join(shard_plan_dir, SHARD_TASKS_FILENAME), shard_rows)

    feature_names = [str(c) for c in list(context["X_train_full"].columns)]
    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "status": "planned",
        "analysis_folder": analysis_dir,
        "output_dir": out_dir,
        "requested_num_shards": int(requested_num_shards),
        "active_num_shards": int(active_num_shards),
        "training_groups": [str(loaded["training_group_a"]), str(loaded["training_group_b"])],
        "n_splits": int(loaded["n_splits"]),
        "resolved_global_random_seed": int(baseline_settings["GLOBAL_RANDOM_SEED"]),
        "n_repeats": int(len(sorted_h_hits)),
        "run_permutation_test": bool(baseline_settings["run_permutation_test"]),
        "n_permutations": int(baseline_settings["n_permutations"]),
        "selected_feature_count": int(len(feature_names)),
        "selected_feature_schema_sha256": compute_feature_schema_sha256(feature_names),
        "settings": dict(baseline_settings),
        "planned_task_count": int(len(task_rows)),
        "planned_family_counts": {
            "real": int(sum(1 for row in task_rows if str(row.get("family")) == "real")),
            "perm": int(sum(1 for row in task_rows if str(row.get("family")) == "perm")),
        },
        "planned_total_model_count": int(len(task_rows) * int(loaded["n_splits"])),
        "artifacts": {
            "task_manifest_csv": os.path.relpath(task_manifest, analysis_dir),
            "plan_dir": os.path.relpath(plan_dir, analysis_dir),
            "shards_root": os.path.relpath(shards_root, analysis_dir),
        },
    }
    _json_dump(manifest, _plan_manifest_path(analysis_dir))
    _json_dump(
        {
            "status": "planned",
            "created_at": manifest["created_at"],
            "analysis_folder": analysis_dir,
            "planned_task_count": int(len(task_rows)),
            "active_num_shards": int(active_num_shards),
            "manifest_path": os.path.relpath(_plan_manifest_path(analysis_dir), analysis_dir),
        },
        _plan_done_path(analysis_dir),
    )
    return plan_dir


def _load_staged_manifest(analysis_folder_path: str) -> Dict[str, Any]:
    manifest_path = _plan_manifest_path(analysis_folder_path)
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(
            f"Plain XGBoost staged plan manifest not found. Run stage=plain_xgboost_baseline_plan first: {manifest_path}"
        )
    payload = _json_load(manifest_path)
    if not isinstance(payload, dict) or str(payload.get("status", "")).strip().lower() != "planned":
        raise RuntimeError(f"Invalid plain XGBoost staged plan manifest: {manifest_path}")
    return payload


def _process_plain_xgboost_baseline_shard(
    *,
    analysis_folder_path: str,
    shard_index: Optional[int] = None,
) -> str:
    """Verify and process the assigned baseline repeat tasks, then publish a completion
    receipt.
    """
    analysis_dir = _analysis_dir(analysis_folder_path)
    manifest = _load_staged_manifest(analysis_dir)
    active_num_shards = int(manifest.get("active_num_shards", 1) or 1)
    resolved_shard_index = _current_shard_index(shard_index)
    if resolved_shard_index < 0 or resolved_shard_index >= active_num_shards:
        raise ValueError(
            f"plain_xgboost_baseline_process_shard received shard_index={resolved_shard_index}, but active_num_shards={active_num_shards}."
        )

    shard_dir = _shard_output_dir(analysis_dir, resolved_shard_index, active_num_shards)
    done_path = os.path.join(shard_dir, SHARD_DONE_FILENAME)
    if os.path.exists(done_path):
        validate_baseline_receipt(done_path, analysis_dir)
        return shard_dir

    shard_plan_dir = _shard_plan_dir(analysis_dir, resolved_shard_index, active_num_shards)
    shard_tasks_path = os.path.join(shard_plan_dir, SHARD_TASKS_FILENAME)
    task_rows = _read_task_rows(shard_tasks_path)
    frozen_inputs = baseline_inputs(analysis_dir, include_shards=False)

    loaded = load_featureimp_baseline_context(analysis_dir)
    baseline_settings = dict(manifest.get("settings") or {})
    runtime_num_workers = _get_runtime_num_workers()
    context = _prepare_context(
        loaded=loaded,
        baseline_settings=baseline_settings,
        runtime_num_workers=runtime_num_workers,
        run_fsea_analysis=False,
    )

    if os.path.isdir(shard_dir):
        from uuid import uuid4
        attempts = os.path.join(_shards_root(analysis_dir), "_attempts")
        os.makedirs(attempts, exist_ok=True)
        os.replace(shard_dir, os.path.join(attempts, os.path.basename(shard_dir) + "_" + uuid4().hex))
    os.makedirs(shard_dir, exist_ok=True)

    started_at = float(time.time())
    running_path = os.path.join(shard_dir, SHARD_RUNNING_FILENAME)
    failed_path = os.path.join(shard_dir, SHARD_FAILED_FILENAME)
    done_path = os.path.join(shard_dir, SHARD_DONE_FILENAME)
    errors_path = os.path.join(shard_dir, SHARD_ERRORS_FILENAME)
    for stale in [running_path, failed_path, done_path, errors_path]:
        if os.path.exists(stale):
            os.remove(stale)
    _json_dump(
        {
            "status": "running",
            "started_at": datetime.now().isoformat(timespec="seconds"),
            "analysis_folder": analysis_dir,
            "shard_index": int(resolved_shard_index),
            "active_num_shards": int(active_num_shards),
            "task_count": int(len(task_rows)),
        },
        running_path,
    )

    real_partial = _init_real_partial(context)
    perm_partials: Dict[int, Dict[str, Any]] = {}
    try:
        for task in task_rows:
            family = str(task.get("family", "")).strip().lower()
            h_hit = int(task.get("h_hit", 0) or 0)
            repeat_idx = int(task.get("repeat_idx", 0) or 0)
            if family == "real":
                payload = _run_real_repeat_task(
                    context=context,
                    h_hit=h_hit,
                    repeat_idx=repeat_idx,
                    started_at=started_at,
                    log_prefix=f"shard {resolved_shard_index}",
                )
                _merge_real_partial(real_partial, payload)
            elif family == "perm":
                perm_idx = int(task.get("permutation_idx", 0) or 0)
                payload = _run_permuted_repeat_task(
                    context=context,
                    permutation_idx=perm_idx,
                    h_hit=h_hit,
                    repeat_idx=repeat_idx,
                    started_at=started_at,
                    log_prefix=f"shard {resolved_shard_index}",
                )
                if int(perm_idx) not in perm_partials:
                    perm_partials[int(perm_idx)] = _init_perm_partial(context, permutation_idx=int(perm_idx))
                target = perm_partials[int(perm_idx)]
                _merge_perm_partial(target, payload)
            else:
                raise ValueError(f"Unsupported plain XGBoost staged task family: {family!r}")

        _write_real_partial_files(shard_dir, real_partial, list(context["ordered_stability_groups"]), int(context["n_feats"]))
        for perm_idx, perm_partial in sorted(perm_partials.items()):
            _write_perm_partial_files(shard_dir, perm_partial)

        if baseline_inputs(analysis_dir, include_shards=False) != frozen_inputs:
            raise RuntimeError("Baseline inputs or code changed while processing")

        _json_dump(
            {
                "status": "completed",
                "started_at": _json_load(running_path).get("started_at"),
                "completed_at": datetime.now().isoformat(timespec="seconds"),
                "analysis_folder": analysis_dir,
                "shard_index": int(resolved_shard_index),
                "active_num_shards": int(active_num_shards),
                "task_count": int(len(task_rows)),
                "artifacts": _required_shard_artifacts(analysis_dir, shard_dir, task_rows),
                "real_task_count": int(sum(1 for row in task_rows if str(row.get("family", "")).strip().lower() == "real")),
                "perm_task_counts": {
                    str(perm_idx): int(sum(1 for row in task_rows if str(row.get("family", "")).strip().lower() == "perm" and int(row.get("permutation_idx", 0) or 0) == int(perm_idx)))
                    for perm_idx in sorted(perm_partials.keys())
                },
            },
            done_path,
        )
        if os.path.exists(running_path):
            os.remove(running_path)
        return shard_dir
    except Exception as e:
        tb = traceback.format_exc()
        with open(errors_path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"error_type": type(e).__name__, "error": str(e), "traceback": tb}, default=str))
            f.write("\n")
        _json_dump(
            {
                "status": "failed",
                "started_at": _json_load(running_path).get("started_at") if os.path.exists(running_path) else datetime.now().isoformat(timespec="seconds"),
                "failed_at": datetime.now().isoformat(timespec="seconds"),
                "analysis_folder": analysis_dir,
                "shard_index": int(resolved_shard_index),
                "active_num_shards": int(active_num_shards),
                "error_type": type(e).__name__,
                "error": str(e),
                "error_log": os.path.basename(errors_path),
            },
            failed_path,
        )
        if os.path.exists(running_path):
            os.remove(running_path)
        raise


def _merge_plain_xgboost_baseline_shards(
    *,
    analysis_folder_path: str,
    settings_override: Optional[Mapping[str, Any]] = None,
    run_fsea_analysis: Optional[bool] = None,
) -> str:
    """Verify exact fold coverage and combine completed baseline shards."""
    analysis_dir = _analysis_dir(analysis_folder_path)
    manifest = _load_staged_manifest(analysis_dir)
    active_num_shards = int(manifest.get("active_num_shards", 1) or 1)

    loaded = load_featureimp_baseline_context(analysis_dir)
    baseline_settings = dict(manifest.get("settings") or {})
    explicit = _iter_explicit_settings(settings_override)
    # Model settings remain frozen by the baseline plan. Postanalysis settings
    # may be explicitly requested at merge, and must reach FSEA unchanged.
    for name, value in explicit.items():
        if name.startswith("fsea_") or name == "run_fsea_analysis":
            baseline_settings[name] = value
    runtime_num_workers = _get_runtime_num_workers()
    resolved_run_fsea = strict_bool(
        run_fsea_analysis if run_fsea_analysis is not None else explicit.get("run_fsea_analysis", baseline_settings.get("run_fsea_analysis", True))
    )
    context = _prepare_context(
        loaded=loaded,
        baseline_settings=baseline_settings,
        runtime_num_workers=runtime_num_workers,
        run_fsea_analysis=resolved_run_fsea,
    )

    started_at = float(time.time())
    real_total = _init_real_partial(context)
    perm_totals: Dict[int, Dict[str, Any]] = {}
    expected_perm_count = int(manifest.get("n_permutations", 0) or 0) if strict_bool(manifest.get("run_permutation_test", False)) else 0

    for shard_index in range(int(active_num_shards)):
        shard_dir = _shard_output_dir(analysis_dir, shard_index, active_num_shards)
        done_path = os.path.join(shard_dir, SHARD_DONE_FILENAME)
        failed_path = os.path.join(shard_dir, SHARD_FAILED_FILENAME)
        if os.path.exists(failed_path):
            failed = _json_load(failed_path)
            raise RuntimeError(
                f"Plain XGBoost staged merge cannot continue because shard {shard_index} failed: {failed.get('error', 'unknown error')}"
            )
        if not os.path.exists(done_path):
            raise RuntimeError(
                f"Plain XGBoost staged merge expected shard completion marker but did not find it: {done_path}"
            )
        done_payload = validate_baseline_receipt(done_path, analysis_dir)
        tasks = _read_task_rows(os.path.join(_shard_plan_dir(analysis_dir, shard_index, active_num_shards), SHARD_TASKS_FILENAME))
        _validate_shard_tasks(shard_dir, tasks, done_payload)
        real_part = _load_real_partial_files(shard_dir, list(context["ordered_stability_groups"]))
        _merge_real_partial(real_total, real_part)
        perm_task_counts = dict(done_payload.get("perm_task_counts") or {})
        for perm_idx in range(expected_perm_count):
            expected_rows = int(perm_task_counts.get(str(perm_idx), 0) or 0)
            perm_path = os.path.join(shard_dir, PERM_PARTIALS_FILENAME_TEMPLATE.format(perm_idx=perm_idx))
            if expected_rows <= 0:
                continue
            if not os.path.exists(perm_path):
                raise RuntimeError(
                    f"Plain XGBoost staged merge expected permutation partial for shard {shard_index}, perm {perm_idx}, but none was found: {perm_path}"
                )
            perm_part = _load_perm_partial_files(shard_dir, perm_idx)
            target = perm_totals.setdefault(int(perm_idx), _init_perm_partial(context, permutation_idx=int(perm_idx)))
            _merge_perm_partial(target, perm_part)

    frozen_inputs = baseline_inputs(analysis_dir, include_shards=True)
    out_dir = _finalize_outputs_from_aggregates(
        context=context,
        real_partial=real_total,
        perm_partials=perm_totals,
        run_fsea_analysis=resolved_run_fsea,
        started_at=started_at,
    )

    if baseline_inputs(analysis_dir, include_shards=True) != frozen_inputs:
        raise RuntimeError("Baseline inputs or code changed during merge")
    merge_done_path = os.path.join(_plan_root(analysis_dir), MERGE_DONE_FILENAME)
    artifact_paths = collect_plain_xgboost_artifact_paths(analysis_dir)
    required_files = [path for path in artifact_paths if os.path.isfile(path)]
    for path in artifact_paths:
        if not os.path.exists(path):
            raise RuntimeError(f"Required baseline output missing: {path}")
    _json_dump(
        {
            "status": "completed",
            "completed_at": datetime.now().isoformat(timespec="seconds"),
            "analysis_folder": analysis_dir,
            "output_dir": os.path.relpath(out_dir, analysis_dir),
            "active_num_shards": int(active_num_shards),
            "run_fsea_analysis": bool(resolved_run_fsea),
            "artifacts": [os.path.relpath(path, analysis_dir) for path in required_files],
        },
        merge_done_path,
    )

    artifact_paths = collect_plain_xgboost_artifact_paths(analysis_dir)
    stage_record = {
        "stage": PLAIN_XGBOOST_STAGE_NAME,
        "status": "completed",
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "completed_at": datetime.now().isoformat(timespec="seconds"),
        "output_dir": os.path.relpath(out_dir, analysis_dir),
        "result_path": os.path.relpath(out_dir, analysis_dir),
        "artifact_paths": [os.path.relpath(path, analysis_dir) for path in artifact_paths],
        "context": {
            "invoked_by": "staged_merge",
            "active_num_shards": int(active_num_shards),
            "training_group_a": str(loaded["training_group_a"]),
            "training_group_b": str(loaded["training_group_b"]),
            "feature_count": int(len(context["feature_names"])),
            "n_splits": int(context["n_splits"]),
            "n_repeats": int(len(context["sorted_h_hits"])),
        },
        "summary": "Plain XGBoost comparison baseline outputs updated via staged shard merge.",
    }
    upsert_postanalysis_stage_record(analysis_dir, stage_record)
    write_baseline_outputs_summary(analysis_dir, settings_snapshot=context["settings_map"])
    return out_dir


from contextlib import contextmanager

@contextmanager
def _baseline_lifecycle_lock(analysis_dir: str, operation: str, index=None):
    """Serialize lifecycle admission while allowing distinct shards to run together."""
    from autobehaver.file_locks import FileLock, file_lock_active
    admission = Path(analysis_dir) / "_plain_xgboost_admission.lock"
    own = Path(analysis_dir) / (f"_plain_xgboost_shard_{index}.lock" if operation == "shard"
                       else f"_plain_xgboost_{operation}.lock")
    owner = None
    with _directory_lock(str(admission)):
        candidates = list(Path(analysis_dir).glob("_plain_xgboost_*.lock"))
        conflicts = [p for p in candidates if p != admission and
                     (operation != "shard" or "_shard_" not in p.name or p == own)]
        if any(p.is_dir() or file_lock_active(p) for p in conflicts):
            raise RuntimeError("Conflicting baseline lifecycle operation is active")
        if os.path.isdir(own):
            raise RuntimeError(f"Legacy baseline lock requires verified recovery: {own}")
        owner = FileLock(own)
        owner.acquire(metadata={"stage": "featureimp_baseline", "operation": operation, "shard": index})
    try:
        yield
    finally:
        owner.release()


def plan_plain_xgboost_baseline_shards(*, analysis_folder_path: str, **kwargs) -> str:
    """Create an immutable baseline plan, excluding live shards and merge."""
    with _baseline_lifecycle_lock(_analysis_dir(analysis_folder_path), "plan"):
        return _plan_plain_xgboost_baseline_shards(analysis_folder_path=analysis_folder_path, **kwargs)


def process_plain_xgboost_baseline_shard(*, analysis_folder_path: str, shard_index: Optional[int] = None) -> str:
    """Own one shard while allowing other independently owned shards to run."""
    index = _current_shard_index(shard_index)
    with _baseline_lifecycle_lock(_analysis_dir(analysis_folder_path), "shard", index):
        return _process_plain_xgboost_baseline_shard(analysis_folder_path=analysis_folder_path, shard_index=index)


def merge_plain_xgboost_baseline_shards(*, analysis_folder_path: str, **kwargs) -> str:
    """Merge only when no baseline shard or plan mutation is active."""
    with _baseline_lifecycle_lock(_analysis_dir(analysis_folder_path), "merge"):
        return _merge_plain_xgboost_baseline_shards(analysis_folder_path=analysis_folder_path, **kwargs)


__all__ = [
    "plan_plain_xgboost_baseline_shards",
    "process_plain_xgboost_baseline_shard",
    "merge_plain_xgboost_baseline_shards",
]
