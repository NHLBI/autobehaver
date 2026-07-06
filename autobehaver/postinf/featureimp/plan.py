"""Planning utilities for staged/sharded FeatureImp execution.

This module implements the *plan* stage for FeatureImp sharding while keeping
heavy ML dependencies out of the import path.  The planner freezes the exact
training/all-groups row order, selected feature columns, fold definitions,
feature-subset plan, and mixed real/permutation task assignments that later
shard jobs will consume.
"""

from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from .config import RECOGNIZED_FEATURE_CATEGORY_PREFIXES
from .hhit import generate_feature_subsets_for_hhit, make_hhit_folds, validate_stratified_cv_labels
from .io_utils import write_analysis_log
from .matrix import compute_feature_schema_sha256
from .prep import (
    _filter_low_variance_features,
    _filter_sparse_features_by_group,
    filter_data_by_sex,
)

PLAN_DIRNAME = "_featureimp_plan"
SHARD_PLAN_DIRNAME = "shards"
PLAN_MANIFEST_FILENAME = "PLAN_MANIFEST.json"
PLAN_DONE_FILENAME = "PLAN_DONE.json"
TASK_MANIFEST_FILENAME = "TASK_MANIFEST.csv"
HHIT_SUBSET_PLAN_FILENAME = "HHIT_SUBSET_PLAN.jsonl"
HHIT_FOLD_PLAN_FILENAME = "HHIT_FOLD_PLAN.jsonl"
PERMUTATION_SEEDS_FILENAME = "PERMUTATION_SEEDS.csv"
SELECTED_FEATURE_COLUMNS_FILENAME = "SELECTED_FEATURE_COLUMNS.json"
TRAINING_ROW_INDICES_FILENAME = "TRAINING_ROW_INDICES.json"
ALL_GROUPS_ROW_INDICES_FILENAME = "ALL_GROUPS_ROW_INDICES.json"
OTHER_GROUP_ROW_INDICES_FILENAME = "OTHER_GROUP_ROW_INDICES.json"


@dataclass(frozen=True)
class FeatureImpPlanArtifacts:
    """Paths to the primary artifacts emitted by :func:`plan_featureimp_run`."""

    analysis_folder: str
    plan_folder: str
    manifest_path: str
    task_manifest_path: str
    subset_plan_path: str
    fold_plan_path: str
    permutation_seeds_path: str



def _normalize_include_feature_categories(val: Any) -> Any:
    """Normalize include_feature_categories into canonical form."""

    if val is None:
        return None

    if isinstance(val, str):
        tokens = [val]
    else:
        try:
            tokens = list(val)
        except Exception:
            tokens = [str(val)]

    if len(tokens) == 0:
        return None

    if len(tokens) == 1:
        t0 = str(tokens[0]).strip().lower()
        if t0 in ("none", "all", "*", "null"):
            return None

    return [str(t) for t in tokens]



def _ensure_int_list(values: Iterable[Any]) -> List[int]:
    return [int(v) for v in values]



def _json_dump(obj: Any, path: str) -> None:
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)



def _jsonl_dump(rows: Iterable[Dict[str, Any]], path: str) -> None:
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row, default=str))
            f.write("\n")



def _relpath(base_dir: str, path: str) -> str:
    return os.path.relpath(os.path.abspath(path), os.path.abspath(base_dir))



def _sanitize_non_numeric_feature_list(df: pd.DataFrame, feature_names: Sequence[str]) -> List[str]:
    numeric_feature_cols = df[list(feature_names)].select_dtypes(include=[np.number]).columns.tolist()
    non_numeric_cols = [col for col in feature_names if col not in numeric_feature_cols]
    if non_numeric_cols:
        print(f"\nWARNING: Excluding {len(non_numeric_cols)} non-numeric columns from plan:")
        for col in non_numeric_cols[:10]:
            print(f"  - {col}")
        if len(non_numeric_cols) > 10:
            print(f"  ... and {len(non_numeric_cols) - 10} more")
    return [str(c) for c in numeric_feature_cols]



def _validate_no_duplicate_features(feature_names: Sequence[str]) -> None:
    dup_mask = pd.Index(feature_names).duplicated()
    if bool(np.any(dup_mask)):
        dup_names = pd.Index(feature_names)[dup_mask].unique().tolist()
        preview = dup_names[:20]
        more = len(dup_names) - len(preview)
        msg = (
            f"Duplicate feature names detected in the feature matrix (n={len(dup_names)}). "
            f"Examples: {preview}" + (f" (+{more} more)" if more > 0 else "")
        )
        raise ValueError(msg)



def _resolve_global_seed(global_random_seed: Optional[int]) -> int:
    """Resolve a concrete global seed for staged/sharded planning.

    Planned shard execution must be reproducible across independent jobs.  If the
    caller did not provide a seed, we freeze one here and record it in the plan.
    """

    if global_random_seed is None:
        resolved = int(np.random.SeedSequence().generate_state(1, dtype=np.uint32)[0])
        print(
            "GLOBAL_RANDOM_SEED was None; resolved a concrete seed for staged planning: "
            f"{resolved}"
        )
        return resolved
    return int(global_random_seed)



def _compute_hhit_fold_plan(
    *,
    X_train_full: pd.DataFrame,
    stratify_labels: np.ndarray,
    n_splits: int,
    H_hits_per_feature: int,
    resolved_global_seed: int,
) -> List[Dict[str, Any]]:
    fold_plan_records: List[Dict[str, Any]] = []
    for h_hit in range(1, int(H_hits_per_feature) + 1):
        split_seed = int(resolved_global_seed + h_hit * 1000)
        folds = make_hhit_folds(
            X_train_full,
            stratify_labels,
            n_splits=int(n_splits),
            random_state=split_seed,
        )
        fold_plan_records.append(
            {
                "h_hit": int(h_hit),
                "split_seed": int(split_seed),
                "folds": [
                    {
                        "fold_idx": int(fold_idx),
                        "train_indices": _ensure_int_list(train_idx.tolist()),
                        "test_indices": _ensure_int_list(test_idx.tolist()),
                    }
                    for fold_idx, (train_idx, test_idx) in enumerate(folds)
                ],
            }
        )
    return fold_plan_records



def _compute_hhit_subset_plan(
    *,
    all_feature_names: Sequence[str],
    H_hits_per_feature: int,
    n_features_per_model: int,
    resolved_global_seed: int,
) -> List[Dict[str, Any]]:
    subset_plan_records: List[Dict[str, Any]] = []
    padding_extra_rep_count: Dict[str, int] = {str(feat): 0 for feat in all_feature_names}
    total_features = int(len(all_feature_names))

    for h_hit in range(1, int(H_hits_per_feature) + 1):
        subset_seed = int(resolved_global_seed + h_hit * 10000)
        plan = generate_feature_subsets_for_hhit(
            list(all_feature_names),
            n_features_per_model=int(n_features_per_model),
            seed=subset_seed,
            log_every=0,
            total_features=total_features,
            extra_rep_count=padding_extra_rep_count,
        )
        for feat, count in plan.feature_usage_count.items():
            if int(count) > 1:
                padding_extra_rep_count[str(feat)] = int(padding_extra_rep_count.get(str(feat), 0) + 1)

        subset_plan_records.append(
            {
                "h_hit": int(h_hit),
                "subset_seed": int(subset_seed),
                "n_subsets": int(len(plan.feature_subsets)),
                "reuse_count": int(plan.reuse_count),
                "feature_subsets": [[str(f) for f in subset] for subset in plan.feature_subsets],
                "feature_usage_count": {str(k): int(v) for k, v in plan.feature_usage_count.items()},
            }
        )

    return subset_plan_records



def _build_task_rows(
    *,
    subset_plan_records: Sequence[Dict[str, Any]],
    n_splits: int,
    run_permutation_test: bool,
    n_permutations: int,
) -> List[Dict[str, Any]]:
    task_rows: List[Dict[str, Any]] = []
    task_id = 0
    perm_count = int(n_permutations) if run_permutation_test else 0

    for record in subset_plan_records:
        h_hit = int(record["h_hit"])
        feature_subsets = record["feature_subsets"]
        for subset_idx, subset in enumerate(feature_subsets):
            task_rows.append(
                {
                    "task_id": int(task_id),
                    "task_key": f"real_h{h_hit:03d}_s{subset_idx:04d}",
                    "family": "real",
                    "permutation_idx": "",
                    "h_hit": int(h_hit),
                    "subset_idx": int(subset_idx),
                    "n_features": int(len(subset)),
                    "n_splits": int(n_splits),
                    "planned_model_count": int(n_splits),
                }
            )
            task_id += 1

            for perm_idx in range(perm_count):
                task_rows.append(
                    {
                        "task_id": int(task_id),
                        "task_key": f"perm_p{perm_idx:03d}_h{h_hit:03d}_s{subset_idx:04d}",
                        "family": "perm",
                        "permutation_idx": int(perm_idx),
                        "h_hit": int(h_hit),
                        "subset_idx": int(subset_idx),
                        "n_features": int(len(subset)),
                        "n_splits": int(n_splits),
                        "planned_model_count": int(n_splits),
                    }
                )
                task_id += 1

    return task_rows



def _assign_tasks_to_shards(task_rows: Sequence[Dict[str, Any]], num_shards: int) -> tuple[List[Dict[str, Any]], int]:
    requested_shards = int(num_shards)
    if requested_shards < 1:
        raise ValueError(f"num_shards must be >= 1, got {num_shards!r}")

    task_count = int(len(task_rows))
    active_num_shards = max(1, min(requested_shards, task_count if task_count > 0 else 1))

    assigned_rows: List[Dict[str, Any]] = []
    width = max(3, len(str(active_num_shards - 1)))
    for idx, row in enumerate(task_rows):
        shard_index = int(idx % active_num_shards)
        shard_label = f"shard_{shard_index:0{width}d}_of_{active_num_shards:0{width}d}"
        new_row = dict(row)
        new_row["shard_index"] = int(shard_index)
        new_row["shard_label"] = shard_label
        assigned_rows.append(new_row)
    return assigned_rows, active_num_shards



def _write_task_manifest_csv(task_rows: Sequence[Dict[str, Any]], path: str) -> None:
    fieldnames = [
        "task_id",
        "task_key",
        "shard_index",
        "shard_label",
        "family",
        "permutation_idx",
        "h_hit",
        "subset_idx",
        "n_features",
        "n_splits",
        "planned_model_count",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in task_rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})



def _write_per_shard_task_manifests(
    *,
    task_rows: Sequence[Dict[str, Any]],
    plan_folder: str,
    active_num_shards: int,
) -> None:
    shard_root = os.path.join(plan_folder, SHARD_PLAN_DIRNAME)
    os.makedirs(shard_root, exist_ok=True)
    width = max(3, len(str(active_num_shards - 1)))

    for shard_index in range(active_num_shards):
        shard_label = f"shard_{shard_index:0{width}d}_of_{active_num_shards:0{width}d}"
        shard_dir = os.path.join(shard_root, shard_label)
        os.makedirs(shard_dir, exist_ok=True)
        shard_rows = [row for row in task_rows if int(row["shard_index"]) == int(shard_index)]
        shard_tasks_path = os.path.join(shard_dir, "SHARD_TASKS.csv")
        _write_task_manifest_csv(shard_rows, shard_tasks_path)

        family_counts: Dict[str, int] = {}
        for row in shard_rows:
            family = str(row.get("family", ""))
            family_counts[family] = int(family_counts.get(family, 0) + 1)

        shard_manifest = {
            "shard_index": int(shard_index),
            "active_num_shards": int(active_num_shards),
            "shard_label": shard_label,
            "task_count": int(len(shard_rows)),
            "family_counts": family_counts,
            "tasks_csv": _relpath(shard_dir, shard_tasks_path),
            "status": "planned",
        }
        _json_dump(shard_manifest, os.path.join(shard_dir, "SHARD_PLAN.json"))



def _build_permutation_seed_rows(
    *,
    run_permutation_test: bool,
    n_permutations: int,
    H_hits_per_feature: int,
    n_splits: int,
    resolved_global_seed: int,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not run_permutation_test:
        return rows

    for perm_idx in range(int(n_permutations)):
        for h_hit in range(1, int(H_hits_per_feature) + 1):
            for fold_idx in range(int(n_splits)):
                seed = int(resolved_global_seed + perm_idx * 100000 + h_hit * 1000 + fold_idx)
                rows.append(
                    {
                        "permutation_idx": int(perm_idx),
                        "h_hit": int(h_hit),
                        "fold_idx": int(fold_idx),
                        "permutation_seed": int(seed),
                    }
                )
    return rows



def _write_csv_rows(rows: Sequence[Dict[str, Any]], path: str) -> None:
    if not rows:
        with open(path, "w", newline="") as f:
            f.write("")
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)



def load_featureimp_plan_manifest(analysis_folder_path: str) -> Dict[str, Any]:
    """Load ``PLAN_MANIFEST.json`` for a staged FeatureImp run."""

    manifest_path = os.path.join(analysis_folder_path, PLAN_DIRNAME, PLAN_MANIFEST_FILENAME)
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"FeatureImp plan manifest not found: {manifest_path}")
    with open(manifest_path, "r") as f:
        return json.load(f)



def plan_featureimp_run(
    normalized_matrix_path: str,
    results_dir: str,
    *,
    analysis_run_name: str = "",
    include_feature_categories: Optional[Sequence[str]] = None,
    analysis_folder_path: Optional[str] = None,
    control_group_tag: Optional[str] = None,
    training_group_a: Optional[str] = None,
    training_group_b: Optional[str] = None,
    sex_filter: str = "all",
    n_splits: int = 5,
    H_hits_per_feature: int = 100,
    n_features_per_model: int = 700,
    min_feature_completeness_per_group: float = 0.25,
    min_variance_threshold: float = 1e-6,
    run_permutation_test: bool = True,
    n_permutations: int = 1,
    num_shards: int = 1,
    GLOBAL_RANDOM_SEED: Optional[int] = 7,
) -> str:
    """Create and persist a frozen staged-execution plan for FeatureImp.

    The emitted plan freezes the exact row order, selected features, fold
    definitions, feature-subset plan, and mixed real/permutation task
    assignments needed for later shard execution.
    """

    if normalized_matrix_path is None:
        raise ValueError("normalized_matrix_path is None; plan stage requires a normalized matrix.")
    if not os.path.exists(normalized_matrix_path):
        raise FileNotFoundError(f"Normalized feature matrix not found: {normalized_matrix_path}")

    if not analysis_folder_path:
        raise ValueError("analysis_folder_path is required for plan_featureimp_run.")

    os.makedirs(analysis_folder_path, exist_ok=True)
    plan_folder = os.path.join(analysis_folder_path, PLAN_DIRNAME)
    os.makedirs(plan_folder, exist_ok=True)

    pipeline_folder = os.path.join(analysis_folder_path, "SAVED_PIPELINE")
    models_folder = os.path.join(pipeline_folder, "models")
    shap_values_folder = os.path.join(pipeline_folder, "temp_shap_values")
    os.makedirs(pipeline_folder, exist_ok=True)
    os.makedirs(models_folder, exist_ok=True)
    os.makedirs(shap_values_folder, exist_ok=True)

    include_feature_categories = _normalize_include_feature_categories(include_feature_categories)
    sex_filter_norm = str(sex_filter or "all").strip().lower()
    if sex_filter_norm not in {"all", "male", "female"}:
        raise ValueError(f"sex_filter must be one of 'all', 'male', or 'female'; got: {sex_filter!r}")
    if not isinstance(n_splits, int) or int(n_splits) < 2:
        raise ValueError(f"n_splits must be an integer >= 2, got: {n_splits!r}")
    if not isinstance(H_hits_per_feature, int) or int(H_hits_per_feature) < 1:
        raise ValueError(
            f"H_hits_per_feature must be an integer >= 1, got: {H_hits_per_feature!r}"
        )
    if not isinstance(n_features_per_model, int) or int(n_features_per_model) < 1:
        raise ValueError(
            f"n_features_per_model must be an integer >= 1, got: {n_features_per_model!r}"
        )
    if training_group_a is None or training_group_b is None:
        raise ValueError("training_group_a and training_group_b are required for plan stage.")

    resolved_global_seed = _resolve_global_seed(GLOBAL_RANDOM_SEED)

    print("\n" + "=" * 80)
    print("Creating staged FeatureImp plan")
    print("=" * 80)
    print(f"Analysis folder: {analysis_folder_path}")
    print(f"Normalized matrix: {normalized_matrix_path}")
    print(f"Requested num_shards: {num_shards}")
    print(f"Resolved GLOBAL_RANDOM_SEED: {resolved_global_seed}")

    df_raw = pd.read_csv(normalized_matrix_path)
    raw_row_count = int(len(df_raw))

    if sex_filter_norm in {"male", "female"}:
        df_all_groups = filter_data_by_sex(df_raw, sex_filter_norm)
        print(
            f"Applied sex filter '{sex_filter_norm}': kept {len(df_all_groups)} / {raw_row_count} rows"
        )
    else:
        df_all_groups = df_raw.copy()

    filtered_row_count = int(len(df_all_groups))

    training_groups = [str(training_group_a), str(training_group_b)]
    df_for_training = df_all_groups[df_all_groups["group"].isin(training_groups)].copy()
    df_for_other_groups = df_all_groups[~df_all_groups["group"].isin(training_groups)].copy()

    if len(df_for_training["group"].unique()) < 2:
        raise ValueError(
            f"Could not find both training groups '{training_group_a}' and '{training_group_b}' in the data."
        )

    id_cols = ["individual", "group", "sex", "source_files", "total_frames_analyzed"]
    all_feature_names = [col for col in df_all_groups.columns if col not in id_cols]
    features_before_numeric_filter = int(len(all_feature_names))
    all_feature_names = _sanitize_non_numeric_feature_list(df_all_groups, all_feature_names)
    _validate_no_duplicate_features(all_feature_names)

    effective_feature_categories = tuple(
        str(prefix)
        for prefix in (
            include_feature_categories
            if include_feature_categories is not None
            else RECOGNIZED_FEATURE_CATEGORY_PREFIXES
        )
    )
    original_count = len(all_feature_names)
    all_feature_names = [
        col
        for col in all_feature_names
        if any(str(col).startswith(prefix) for prefix in effective_feature_categories)
    ]
    if include_feature_categories is None:
        print("\nFeature category filtering applied using default recognized feature prefixes:")
    else:
        print("\nFeature category filtering applied:")
    print(f"  - Included categories: {list(effective_feature_categories)}")
    print(f"  - Features before filtering: {original_count}")
    print(f"  - Features after filtering: {len(all_feature_names)}")
    if len(all_feature_names) == 0:
        raise ValueError(
            "No features remain after category filtering. Check include_feature_categories / recognized prefixes."
        )

    all_kinematic_prefixes = ("Kinematic_", "KinematicBB_", "KinematicTW_")
    bb_like_kinematic_prefixes = ("Kinematic_", "KinematicBB_")
    kinematic_features_before = len([col for col in all_feature_names if str(col).startswith(all_kinematic_prefixes)])
    all_feature_names = [
        col
        for col in all_feature_names
        if not (
            str(col).startswith(bb_like_kinematic_prefixes)
            and ("_5_" in str(col) or "_9_" in str(col) or str(col).endswith("_5") or str(col).endswith("_9"))
        )
    ]
    kinematic_features_after = len([col for col in all_feature_names if str(col).startswith(all_kinematic_prefixes)])
    if kinematic_features_before > kinematic_features_after:
        print("\nKinematic feature redundancy filtering applied:")
        print(f"  - Kinematic features before filtering: {kinematic_features_before}")
        print(f"  - Kinematic features after filtering: {kinematic_features_after}")

    if float(min_feature_completeness_per_group) > 0 and training_group_a and training_group_b:
        print(
            f"\nFiltering sparse features (min {float(min_feature_completeness_per_group) * 100:.0f}% completeness per training group)..."
        )
        all_feature_names = _filter_sparse_features_by_group(
            df=df_all_groups,
            training_group_a=str(training_group_a),
            training_group_b=str(training_group_b),
            min_completeness=float(min_feature_completeness_per_group),
            feature_cols=all_feature_names,
        )
        if len(all_feature_names) == 0:
            raise ValueError(
                "All features were dropped by sparsity filtering. Lower min_feature_completeness_per_group."
            )

    y_train = df_for_training["group"]

    if "sex" not in df_for_training.columns:
        raise ValueError(
            "FeatureImp requires a 'sex' column in the normalized feature matrix to build sex-aware cross-validation folds."
        )

    df_for_training["sex_group"] = (
        df_for_training["sex"].astype(str) + "_" + df_for_training["group"].astype(str)
    )
    sex_group_counts = validate_stratified_cv_labels(
        df_for_training["sex_group"].values,
        n_splits=int(n_splits),
        context="FeatureImp training cross-validation",
    )
    le_sex_group = LabelEncoder()
    sex_group_encoded = le_sex_group.fit_transform(df_for_training["sex_group"])

    X_train_full = df_for_training[all_feature_names].copy()
    n_missing_train = int(X_train_full.isna().sum().sum())
    if n_missing_train > 0:
        print(
            f"Filling {n_missing_train} missing values in training feature matrix with 0 for planning"
        )
        X_train_full = X_train_full.fillna(0)

    all_feature_names, variance_filtering_stats = _filter_low_variance_features(
        X_train_full=X_train_full,
        y_train=y_train,
        training_group_a=str(training_group_a),
        training_group_b=str(training_group_b),
        feature_cols=all_feature_names,
        min_variance_threshold=float(min_variance_threshold),
    )
    if len(all_feature_names) == 0:
        raise ValueError(
            "All features were dropped by variance filtering. Adjust MIN_VARIANCE_THRESHOLD."
        )
    X_train_full = X_train_full[all_feature_names]

    training_row_indices = _ensure_int_list(df_for_training.index.tolist())
    other_group_row_indices = _ensure_int_list(df_for_other_groups.index.tolist())
    all_groups_row_indices = training_row_indices + other_group_row_indices
    X_all_groups = df_raw.loc[all_groups_row_indices, all_feature_names].copy()
    n_missing_all = int(X_all_groups.isna().sum().sum())
    if n_missing_all > 0:
        print(
            f"Filling {n_missing_all} missing values in all-groups feature matrix with 0 for planning"
        )
        X_all_groups = X_all_groups.fillna(0)

    total_features = int(len(all_feature_names))
    n_subsets_per_round = int(np.ceil(total_features / int(n_features_per_model)))

    with open(os.path.join(pipeline_folder, "GROUP_LABEL_MAPPING.json"), "w") as f:
        json.dump({0: str(training_group_a), 1: str(training_group_b)}, f, indent=2)

    write_analysis_log(
        analysis_folder_path=analysis_folder_path,
        normalized_matrix_path=normalized_matrix_path,
        results_dir=results_dir,
        analysis_run_name=analysis_run_name,
        control_group_tag=control_group_tag,
        training_group_a=training_group_a,
        training_group_b=training_group_b,
        include_feature_categories=include_feature_categories,
        n_splits=int(n_splits),
        variance_filtering_stats=variance_filtering_stats,
        min_variance_threshold=float(min_variance_threshold),
        sex_filter=sex_filter_norm,
        file_suffix="",
    )

    fold_plan_records = _compute_hhit_fold_plan(
        X_train_full=X_train_full,
        stratify_labels=sex_group_encoded,
        n_splits=int(n_splits),
        H_hits_per_feature=int(H_hits_per_feature),
        resolved_global_seed=resolved_global_seed,
    )
    subset_plan_records = _compute_hhit_subset_plan(
        all_feature_names=all_feature_names,
        H_hits_per_feature=int(H_hits_per_feature),
        n_features_per_model=int(n_features_per_model),
        resolved_global_seed=resolved_global_seed,
    )
    task_rows = _build_task_rows(
        subset_plan_records=subset_plan_records,
        n_splits=int(n_splits),
        run_permutation_test=bool(run_permutation_test),
        n_permutations=int(n_permutations),
    )
    task_rows, active_num_shards = _assign_tasks_to_shards(task_rows, int(num_shards))
    permutation_seed_rows = _build_permutation_seed_rows(
        run_permutation_test=bool(run_permutation_test),
        n_permutations=int(n_permutations),
        H_hits_per_feature=int(H_hits_per_feature),
        n_splits=int(n_splits),
        resolved_global_seed=resolved_global_seed,
    )

    selected_features_path = os.path.join(plan_folder, SELECTED_FEATURE_COLUMNS_FILENAME)
    training_rows_path = os.path.join(plan_folder, TRAINING_ROW_INDICES_FILENAME)
    all_groups_rows_path = os.path.join(plan_folder, ALL_GROUPS_ROW_INDICES_FILENAME)
    other_rows_path = os.path.join(plan_folder, OTHER_GROUP_ROW_INDICES_FILENAME)
    subset_plan_path = os.path.join(plan_folder, HHIT_SUBSET_PLAN_FILENAME)
    fold_plan_path = os.path.join(plan_folder, HHIT_FOLD_PLAN_FILENAME)
    task_manifest_path = os.path.join(plan_folder, TASK_MANIFEST_FILENAME)
    permutation_seed_path = os.path.join(plan_folder, PERMUTATION_SEEDS_FILENAME)
    manifest_path = os.path.join(plan_folder, PLAN_MANIFEST_FILENAME)
    plan_done_path = os.path.join(plan_folder, PLAN_DONE_FILENAME)

    _json_dump([str(c) for c in all_feature_names], selected_features_path)
    _json_dump(training_row_indices, training_rows_path)
    _json_dump(all_groups_row_indices, all_groups_rows_path)
    _json_dump(other_group_row_indices, other_rows_path)
    _jsonl_dump(fold_plan_records, fold_plan_path)
    _jsonl_dump(subset_plan_records, subset_plan_path)
    _write_task_manifest_csv(task_rows, task_manifest_path)
    _write_per_shard_task_manifests(
        task_rows=task_rows,
        plan_folder=plan_folder,
        active_num_shards=active_num_shards,
    )
    _write_csv_rows(permutation_seed_rows, permutation_seed_path)

    family_counts: Dict[str, int] = {}
    for row in task_rows:
        fam = str(row.get("family", ""))
        family_counts[fam] = int(family_counts.get(fam, 0) + 1)

    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "status": "planned",
        "analysis_folder": analysis_folder_path,
        "analysis_folder_name": os.path.basename(os.path.abspath(analysis_folder_path)),
        "normalized_matrix_path": normalized_matrix_path,
        "resolved_global_random_seed": int(resolved_global_seed),
        "requested_num_shards": int(num_shards),
        "active_num_shards": int(active_num_shards),
        "training_groups": [str(training_group_a), str(training_group_b)],
        "sex_filter": sex_filter_norm,
        "n_splits": int(n_splits),
        "H_hits_per_feature": int(H_hits_per_feature),
        "n_features_per_model": int(n_features_per_model),
        "run_permutation_test": bool(run_permutation_test),
        "n_permutations": int(n_permutations if run_permutation_test else 0),
        "raw_row_count": raw_row_count,
        "sex_filtered_row_count": filtered_row_count,
        "training_row_count": int(len(training_row_indices)),
        "other_group_row_count": int(len(other_group_row_indices)),
        "all_groups_row_count": int(len(all_groups_row_indices)),
        "features_before_numeric_filter": int(features_before_numeric_filter),
        "selected_feature_count": int(len(all_feature_names)),
        "selected_feature_schema_sha256": compute_feature_schema_sha256(all_feature_names),
        "n_subsets_per_round": int(n_subsets_per_round),
        "planned_task_count": int(len(task_rows)),
        "planned_family_counts": family_counts,
        "planned_total_model_count": int(len(task_rows) * int(n_splits)),
        "training_group_counts": {
            str(k): int(v)
            for k, v in df_for_training["group"].value_counts().sort_index().to_dict().items()
        },
        "other_group_counts": {
            str(k): int(v)
            for k, v in df_for_other_groups["group"].value_counts().sort_index().to_dict().items()
        },
        "sex_group_counts": {str(k): int(v) for k, v in sex_group_counts.items()},
        "artifacts": {
            "selected_feature_columns_json": _relpath(analysis_folder_path, selected_features_path),
            "training_row_indices_json": _relpath(analysis_folder_path, training_rows_path),
            "all_groups_row_indices_json": _relpath(analysis_folder_path, all_groups_rows_path),
            "other_group_row_indices_json": _relpath(analysis_folder_path, other_rows_path),
            "hhit_fold_plan_jsonl": _relpath(analysis_folder_path, fold_plan_path),
            "hhit_subset_plan_jsonl": _relpath(analysis_folder_path, subset_plan_path),
            "task_manifest_csv": _relpath(analysis_folder_path, task_manifest_path),
            "permutation_seeds_csv": _relpath(analysis_folder_path, permutation_seed_path),
            "normalized_matrix_csv": _relpath(analysis_folder_path, normalized_matrix_path),
            "normalization_params_json": _relpath(
                analysis_folder_path,
                os.path.join(analysis_folder_path, "NORMALIZATION_PARAMS.json"),
            ),
            "group_label_mapping_json": _relpath(
                analysis_folder_path,
                os.path.join(pipeline_folder, "GROUP_LABEL_MAPPING.json"),
            ),
        },
    }
    _json_dump(manifest, manifest_path)
    _json_dump(
        {
            "status": "planned",
            "created_at": manifest["created_at"],
            "analysis_folder": analysis_folder_path,
            "manifest_path": _relpath(analysis_folder_path, manifest_path),
            "planned_task_count": int(len(task_rows)),
            "active_num_shards": int(active_num_shards),
        },
        plan_done_path,
    )

    print("\nPlan artifacts written:")
    print(f"  - Manifest: {manifest_path}")
    print(f"  - Task manifest: {task_manifest_path}")
    print(f"  - Fold plan: {fold_plan_path}")
    print(f"  - Subset plan: {subset_plan_path}")
    print(f"  - Per-shard plans: {os.path.join(plan_folder, SHARD_PLAN_DIRNAME)}")
    print(f"  - Active shards: {active_num_shards}")

    return analysis_folder_path


__all__ = [
    "FeatureImpPlanArtifacts",
    "PLAN_DIRNAME",
    "SHARD_PLAN_DIRNAME",
    "PLAN_MANIFEST_FILENAME",
    "PLAN_DONE_FILENAME",
    "TASK_MANIFEST_FILENAME",
    "HHIT_SUBSET_PLAN_FILENAME",
    "HHIT_FOLD_PLAN_FILENAME",
    "PERMUTATION_SEEDS_FILENAME",
    "SELECTED_FEATURE_COLUMNS_FILENAME",
    "TRAINING_ROW_INDICES_FILENAME",
    "ALL_GROUPS_ROW_INDICES_FILENAME",
    "OTHER_GROUP_ROW_INDICES_FILENAME",
    "load_featureimp_plan_manifest",
    "plan_featureimp_run",
]
