"""Per-subset SHAP task artifacts for staged FeatureImp weighted aggregation.

This module supports a faster staged merge path for SHAP aggregation.
Instead of reopening every saved XGBoost model during the final merge just to
recompute SHAP values, shard jobs can persist per-subset SHAP sums while the
model and explainer are already in memory. Merge then only needs to apply
subset weights and aggregate those saved subset-local tensors.

Two use cases are supported:

* Real committee weighted SHAP
  - needs OOF, in-sample, and all-groups per-subset SHAP sums
* Permuted weighted baseline
  - only needs OOF per-subset SHAP sums
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


REAL_SHAP_TASKS_FILENAME = "REAL_SHAP_TASKS.npz"
PERM_SHAP_TASKS_FILENAME_TEMPLATE = "PERM_SHAP_TASKS_perm{perm_idx:03d}.npz"
WEIGHTED_REAL_SHAP_PARTIAL_FILENAME = "WEIGHTED_REAL_SHAP_PARTIAL.npz"
WEIGHTED_PERM_SHAP_PARTIAL_FILENAME_TEMPLATE = "WEIGHTED_PERM_SHAP_PARTIAL_perm{perm_idx:03d}.npz"

_REQUIRED_ARCHIVE_KEYS = (
    "h_hit",
    "subset_idx",
    "feature_count",
    "feature_indices_flat",
    "oof_shap_flat",
)


def save_shap_task_archive(
    path: str,
    *,
    task_records: Sequence[Mapping[str, Any]],
    n_train: int,
    n_all: int,
    include_insample: bool,
    include_all_groups: bool,
    metadata: Optional[Mapping[str, Any]] = None,
) -> None:
    """Write one shard-local stacked archive of per-subset SHAP outputs."""

    arrays: Dict[str, np.ndarray] = {
        "h_hit": np.asarray([int(r["h_hit"]) for r in task_records], dtype=np.int32),
        "subset_idx": np.asarray([int(r["subset_idx"]) for r in task_records], dtype=np.int32),
        "feature_count": np.asarray(
            [int(len(np.asarray(r["feature_indices"], dtype=np.int32))) for r in task_records],
            dtype=np.int32,
        ),
        "feature_indices_flat": np.concatenate(
            [np.asarray(r["feature_indices"], dtype=np.int32).reshape(-1) for r in task_records],
            axis=0,
        ) if task_records else np.zeros((0,), dtype=np.int32),
        "oof_shap_flat": np.concatenate(
            [np.asarray(r["oof_shap_sum"], dtype=np.float32).reshape(-1) for r in task_records],
            axis=0,
        ) if task_records else np.zeros((0,), dtype=np.float32),
        "meta__n_train": np.asarray(int(n_train), dtype=np.int32),
        "meta__n_all": np.asarray(int(n_all), dtype=np.int32),
        "meta__include_insample": np.asarray(bool(include_insample)),
        "meta__include_all_groups": np.asarray(bool(include_all_groups)),
    }
    if include_insample:
        arrays["insample_shap_flat"] = np.concatenate(
            [np.asarray(r["insample_shap_sum"], dtype=np.float32).reshape(-1) for r in task_records],
            axis=0,
        ) if task_records else np.zeros((0,), dtype=np.float32)
    if include_all_groups:
        arrays["all_groups_shap_flat"] = np.concatenate(
            [np.asarray(r["all_groups_shap_sum"], dtype=np.float32).reshape(-1) for r in task_records],
            axis=0,
        ) if task_records else np.zeros((0,), dtype=np.float32)
    if metadata:
        for key, value in metadata.items():
            arrays[f"meta__{key}"] = np.asarray(value)
    np.savez(path, **arrays)


def load_shap_task_archive(path: str) -> Dict[str, np.ndarray]:
    if not path:
        raise ValueError("SHAP task archive path is empty.")
    out: Dict[str, np.ndarray] = {}
    with np.load(path, allow_pickle=False) as data:
        for key in _REQUIRED_ARCHIVE_KEYS:
            if key not in data.files:
                raise ValueError(f"SHAP task archive missing required key '{key}': {path}")
            out[key] = np.asarray(data[key])
        for key in ("insample_shap_flat", "all_groups_shap_flat"):
            if key in data.files:
                out[key] = np.asarray(data[key])
        for key in ("meta__n_train", "meta__n_all", "meta__include_insample", "meta__include_all_groups"):
            if key in data.files:
                out[key] = np.asarray(data[key])
    return out


def _iter_relevant_model_records(
    performance_records: Iterable[Mapping[str, Any]],
    *,
    include_permuted: bool = False,
    permutation_idx: Optional[int] = None,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for rec in performance_records:
        if not isinstance(rec, Mapping):
            continue
        is_perm = bool(rec.get("is_permuted", False))
        if include_permuted:
            if not is_perm:
                continue
            if permutation_idx is not None:
                try:
                    rec_perm_idx = int(rec.get("permutation_idx"))
                except Exception:
                    rec_perm_idx = None
                if rec_perm_idx != int(permutation_idx):
                    continue
        else:
            if is_perm:
                continue
        out.append(dict(rec))
    return out


def _load_task_records_from_archives(
    archive_paths: Sequence[str],
    *,
    n_train: int,
    n_all: int,
) -> Dict[Tuple[int, int], Dict[str, Any]]:
    task_map: Dict[Tuple[int, int], Dict[str, Any]] = {}
    for path in archive_paths:
        if not path:
            continue
        payload = load_shap_task_archive(path)
        include_insample = bool(np.asarray(payload.get("meta__include_insample", False)).reshape(()))
        include_all_groups = bool(np.asarray(payload.get("meta__include_all_groups", False)).reshape(()))
        if "meta__n_train" in payload and int(np.asarray(payload["meta__n_train"]).reshape(())) != int(n_train):
            raise RuntimeError(f"SHAP task archive {path} has n_train mismatch")
        if include_all_groups and "meta__n_all" in payload and int(np.asarray(payload["meta__n_all"]).reshape(())) != int(n_all):
            raise RuntimeError(f"SHAP task archive {path} has n_all mismatch")
        n_tasks = int(payload["h_hit"].shape[0])
        feat_flat = np.asarray(payload["feature_indices_flat"], dtype=np.int32)
        oof_flat = np.asarray(payload["oof_shap_flat"], dtype=np.float32)
        ins_flat = np.asarray(payload.get("insample_shap_flat", np.zeros((0,), dtype=np.float32)), dtype=np.float32)
        all_flat = np.asarray(payload.get("all_groups_shap_flat", np.zeros((0,), dtype=np.float32)), dtype=np.float32)
        feat_offset = oof_offset = ins_offset = all_offset = 0
        for idx in range(n_tasks):
            key = (int(payload["h_hit"][idx]), int(payload["subset_idx"][idx]))
            if key in task_map:
                raise RuntimeError(f"Duplicate SHAP task entry for subset {key}.")
            feat_count = int(payload["feature_count"][idx])
            feat_idxs = feat_flat[feat_offset: feat_offset + feat_count].astype(np.int32, copy=False)
            feat_offset += feat_count
            oof_count = int(n_train) * feat_count
            oof_vals = oof_flat[oof_offset: oof_offset + oof_count].reshape(int(n_train), feat_count)
            oof_offset += oof_count
            rec: Dict[str, Any] = {
                "h_hit": int(payload["h_hit"][idx]),
                "subset_idx": int(payload["subset_idx"][idx]),
                "feature_indices": feat_idxs.copy(),
                "oof_shap_sum": np.asarray(oof_vals, dtype=np.float32),
            }
            if include_insample:
                ins_count = int(n_train) * feat_count
                ins_vals = ins_flat[ins_offset: ins_offset + ins_count].reshape(int(n_train), feat_count)
                ins_offset += ins_count
                rec["insample_shap_sum"] = np.asarray(ins_vals, dtype=np.float32)
            if include_all_groups:
                all_count = int(n_all) * feat_count
                all_vals = all_flat[all_offset: all_offset + all_count].reshape(int(n_all), feat_count)
                all_offset += all_count
                rec["all_groups_shap_sum"] = np.asarray(all_vals, dtype=np.float32)
            task_map[key] = rec
        if feat_offset != int(feat_flat.shape[0]) or oof_offset != int(oof_flat.shape[0]):
            raise RuntimeError(f"SHAP task archive malformed: {path}")
        if include_insample and ins_offset != int(ins_flat.shape[0]):
            raise RuntimeError(f"SHAP task archive insample payload malformed: {path}")
        if include_all_groups and all_offset != int(all_flat.shape[0]):
            raise RuntimeError(f"SHAP task archive all-groups payload malformed: {path}")
    return task_map








def compute_weighted_shap_from_task_archives(
    *,
    archive_paths: Sequence[str],
    performance_records: Sequence[Mapping[str, Any]],
    subset_weights: Mapping[Tuple[int, int], float],
    all_feature_names: Sequence[str],
    n_train: int,
    n_all: int,
    n_splits: int,
    include_permuted: bool = False,
    permutation_idx: Optional[int] = None,
    require_full: bool = True,
    verbose: bool = True,
) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray], np.ndarray, Dict[str, Any]]:
    if n_splits < 2:
        raise ValueError(f"n_splits must be >= 2, got {n_splits}")
    if not archive_paths:
        raise RuntimeError("No SHAP task archives were provided for aggregation.")
    records = _iter_relevant_model_records(
        performance_records,
        include_permuted=include_permuted,
        permutation_idx=permutation_idx,
    )
    subsets: Dict[Tuple[int, int], Dict[int, Dict[str, Any]]] = {}
    for rec in records:
        try:
            h_hit = int(rec.get("h_hit"))
            subset_idx = int(rec.get("subset_idx"))
            fold_idx = int(rec.get("fold_idx"))
        except Exception:
            continue
        subsets.setdefault((h_hit, subset_idx), {})[fold_idx] = dict(rec)
    task_map = _load_task_records_from_archives(archive_paths, n_train=int(n_train), n_all=int(n_all))
    n_features = int(len(all_feature_names))
    oof_sum = np.zeros((int(n_train), n_features), dtype=np.float32)
    insample_sum = np.zeros((int(n_train), n_features), dtype=np.float32) if require_full else None
    all_sum = np.zeros((int(n_all), n_features), dtype=np.float32) if require_full else None
    feature_weight_sums = np.zeros((n_features,), dtype=np.float32)
    n_subsets_total = int(len(subsets))
    n_subsets_used = 0
    n_subsets_skipped_zero_w = 0
    n_subsets_skipped_incomplete = 0
    n_subsets_skipped_missing_archive = 0
    total_weight_sum = 0.0
    for (h_hit, subset_idx), fold_map in sorted(subsets.items()):
        if len(fold_map) != int(n_splits) or any(i not in fold_map for i in range(int(n_splits))):
            n_subsets_skipped_incomplete += 1
            continue
        w = float(subset_weights.get((int(h_hit), int(subset_idx)), 0.0))
        if not np.isfinite(w) or w <= 0.0:
            n_subsets_skipped_zero_w += 1
            continue
        task = task_map.get((int(h_hit), int(subset_idx)))
        if task is None:
            n_subsets_skipped_missing_archive += 1
            continue
        feat_idxs = np.asarray(task["feature_indices"], dtype=np.int64)
        oof_sum[:, feat_idxs] += np.float32(w) * np.asarray(task["oof_shap_sum"], dtype=np.float32)
        if require_full:
            if "insample_shap_sum" not in task or "all_groups_shap_sum" not in task:
                raise RuntimeError(f"Subset {(h_hit, subset_idx)} missing required full SHAP tensors.")
            insample_sum[:, feat_idxs] += np.float32(w) * np.asarray(task["insample_shap_sum"], dtype=np.float32)
            all_sum[:, feat_idxs] += np.float32(w / float(n_splits)) * np.asarray(task["all_groups_shap_sum"], dtype=np.float32)
        feature_weight_sums[feat_idxs] += np.float32(w)
        total_weight_sum += float(w)
        n_subsets_used += 1
        if verbose and (n_subsets_used % 25 == 0):
            print(f"  [weighted_shap_tasks] processed {n_subsets_used}/{n_subsets_total} weighted subsets")
    if n_subsets_skipped_missing_archive > 0:
        raise RuntimeError(f"Missing SHAP task archives for {int(n_subsets_skipped_missing_archive)} eligible complete subsets.")
    if n_subsets_used == 0:
        only_zero_weight_complete_subsets = (
            n_subsets_total > 0
            and (n_subsets_skipped_zero_w + n_subsets_skipped_incomplete) == n_subsets_total
            and n_subsets_skipped_zero_w > 0
        )
        if not only_zero_weight_complete_subsets:
            raise RuntimeError("No subsets were successfully aggregated from SHAP task archives.")
    denom = feature_weight_sums.astype(np.float32, copy=False)
    denom_safe = np.where(denom > 0, denom, 1.0).astype(np.float32)
    oof_avg = oof_sum / denom_safe[None, :]
    insample_avg = None
    all_avg = None
    if require_full:
        insample_avg = insample_sum / (denom_safe * np.float32(max(int(n_splits) - 1, 1)))[None, :]
        all_avg = all_sum / denom_safe[None, :]
    zero_cols = denom <= 0
    if np.any(zero_cols):
        oof_avg[:, zero_cols] = 0.0
        if insample_avg is not None:
            insample_avg[:, zero_cols] = 0.0
        if all_avg is not None:
            all_avg[:, zero_cols] = 0.0
    diagnostics: Dict[str, Any] = {
        "n_train": int(n_train),
        "n_all": int(n_all),
        "n_features": int(n_features),
        "n_splits": int(n_splits),
        "n_subsets_total": int(n_subsets_total),
        "n_subsets_used": int(n_subsets_used),
        "n_subsets_skipped_zero_weight": int(n_subsets_skipped_zero_w),
        "n_subsets_skipped_incomplete": int(n_subsets_skipped_incomplete),
        "n_subsets_skipped_missing_archive": int(n_subsets_skipped_missing_archive),
        "total_weight_sum": float(total_weight_sum),
        "n_features_with_zero_weight": int(np.sum(feature_weight_sums <= 0)),
        "include_permuted": bool(include_permuted),
        "require_full": bool(require_full),
        "source": "shap_task_archives",
    }
    return oof_avg, insample_avg, all_avg, feature_weight_sums, diagnostics


def reduce_weighted_shap_task_archives_to_partial(
    *,
    archive_paths: Sequence[str],
    performance_records: Sequence[Mapping[str, Any]],
    subset_weights: Mapping[Tuple[int, int], float],
    all_feature_names: Sequence[str],
    n_train: int,
    n_all: int,
    n_splits: int,
    include_permuted: bool = False,
    permutation_idx: Optional[int] = None,
    require_full: bool = True,
    verbose: bool = True,
) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    """Reduce one shard's saved SHAP task archives into weighted additive partials."""

    if n_splits < 2:
        raise ValueError(f"n_splits must be >= 2, got {n_splits}")
    if not archive_paths:
        raise RuntimeError("No SHAP task archives were provided for weighted shard reduction.")

    records = _iter_relevant_model_records(
        performance_records,
        include_permuted=include_permuted,
        permutation_idx=permutation_idx,
    )
    subsets: Dict[Tuple[int, int], Dict[int, Dict[str, Any]]] = {}
    for rec in records:
        try:
            h_hit = int(rec.get("h_hit"))
            subset_idx = int(rec.get("subset_idx"))
            fold_idx = int(rec.get("fold_idx"))
        except Exception:
            continue
        subsets.setdefault((h_hit, subset_idx), {})[fold_idx] = dict(rec)

    task_map = _load_task_records_from_archives(archive_paths, n_train=int(n_train), n_all=int(n_all))
    n_features = int(len(all_feature_names))
    oof_sum = np.zeros((int(n_train), n_features), dtype=np.float32)
    insample_sum = np.zeros((int(n_train), n_features), dtype=np.float32) if require_full else None
    all_sum = np.zeros((int(n_all), n_features), dtype=np.float32) if require_full else None
    feature_weight_sums = np.zeros((n_features,), dtype=np.float32)
    n_subsets_total = int(len(subsets))
    n_subsets_used = 0
    n_subsets_skipped_zero_w = 0
    n_subsets_skipped_incomplete = 0
    n_subsets_skipped_missing_archive = 0
    total_weight_sum = 0.0

    for (h_hit, subset_idx), fold_map in sorted(subsets.items()):
        if len(fold_map) != int(n_splits) or any(i not in fold_map for i in range(int(n_splits))):
            n_subsets_skipped_incomplete += 1
            continue
        w = float(subset_weights.get((int(h_hit), int(subset_idx)), 0.0))
        if not np.isfinite(w) or w <= 0.0:
            n_subsets_skipped_zero_w += 1
            continue
        task = task_map.get((int(h_hit), int(subset_idx)))
        if task is None:
            n_subsets_skipped_missing_archive += 1
            continue

        feat_idxs = np.asarray(task["feature_indices"], dtype=np.int64)
        oof_sum[:, feat_idxs] += np.float32(w) * np.asarray(task["oof_shap_sum"], dtype=np.float32)
        if require_full:
            if "insample_shap_sum" not in task or "all_groups_shap_sum" not in task:
                raise RuntimeError(f"Subset {(h_hit, subset_idx)} missing required full SHAP tensors.")
            insample_sum[:, feat_idxs] += np.float32(w) * np.asarray(task["insample_shap_sum"], dtype=np.float32)
            all_sum[:, feat_idxs] += np.float32(w / float(n_splits)) * np.asarray(task["all_groups_shap_sum"], dtype=np.float32)
        feature_weight_sums[feat_idxs] += np.float32(w)
        total_weight_sum += float(w)
        n_subsets_used += 1
        if verbose and (n_subsets_used % 25 == 0):
            print(f"  [weighted_shap_partial] processed {n_subsets_used}/{n_subsets_total} weighted subsets")

    if n_subsets_skipped_missing_archive > 0:
        raise RuntimeError(f"Missing SHAP task archives for {int(n_subsets_skipped_missing_archive)} eligible complete subsets.")

    partial: Dict[str, np.ndarray] = {
        "oof_shap_num": np.asarray(oof_sum, dtype=np.float32),
        "feature_weight_den": np.asarray(feature_weight_sums, dtype=np.float32),
        "total_weight_sum": np.asarray(float(total_weight_sum), dtype=np.float64),
    }
    if require_full:
        partial["insample_shap_num"] = np.asarray(insample_sum, dtype=np.float32)
        partial["all_shap_num"] = np.asarray(all_sum, dtype=np.float32)

    diagnostics = {
        "n_train": int(n_train),
        "n_all": int(n_all),
        "n_features": int(n_features),
        "n_splits": int(n_splits),
        "n_subsets_total": int(n_subsets_total),
        "n_subsets_used": int(n_subsets_used),
        "n_subsets_skipped_zero_weight": int(n_subsets_skipped_zero_w),
        "n_subsets_skipped_incomplete": int(n_subsets_skipped_incomplete),
        "n_subsets_skipped_missing_archive": int(n_subsets_skipped_missing_archive),
        "total_weight_sum": float(total_weight_sum),
        "include_permuted": bool(include_permuted),
        "require_full": bool(require_full),
        "source": "weighted_shap_shard_reduce",
    }
    return partial, diagnostics


def merge_weighted_shap_partials(
    paths: Sequence[str],
    *,
    require_full: bool,
) -> Dict[str, np.ndarray]:
    if not paths:
        raise ValueError("No weighted SHAP partial paths were provided for merge.")
    required = ["oof_shap_num", "feature_weight_den", "total_weight_sum"]
    if require_full:
        required.extend(["insample_shap_num", "all_shap_num"])
    merged: Optional[Dict[str, np.ndarray]] = None
    for path in paths:
        if not path:
            continue
        with np.load(path, allow_pickle=False) as data:
            payload = {key: np.asarray(data[key]) for key in required}
        if merged is None:
            merged = {k: np.array(v, copy=True) for k, v in payload.items()}
            continue
        for key, value in payload.items():
            if merged[key].shape != value.shape:
                raise ValueError(
                    f"Weighted SHAP partial shape mismatch for key '{key}': expected {merged[key].shape}, got {value.shape}."
                )
            merged[key] = merged[key] + value
    if merged is None:
        raise ValueError("No readable weighted SHAP partial payloads were found.")
    return merged


def finalize_weighted_shap_partial(
    *,
    partial: Mapping[str, Any],
    n_splits: int,
    require_full: bool,
) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray], np.ndarray, Dict[str, Any]]:
    feature_weight_sums = np.asarray(partial["feature_weight_den"], dtype=np.float32)
    denom_safe = np.where(feature_weight_sums > 0, feature_weight_sums, 1.0).astype(np.float32)
    oof_avg = np.asarray(partial["oof_shap_num"], dtype=np.float32) / denom_safe[None, :]
    insample_avg: Optional[np.ndarray] = None
    all_avg: Optional[np.ndarray] = None
    if require_full:
        insample_avg = np.asarray(partial["insample_shap_num"], dtype=np.float32) / (denom_safe * np.float32(max(int(n_splits) - 1, 1)))[None, :]
        all_avg = np.asarray(partial["all_shap_num"], dtype=np.float32) / denom_safe[None, :]
    zero_cols = denom_safe <= 0
    if np.any(zero_cols):
        oof_avg[:, zero_cols] = 0.0
        if insample_avg is not None:
            insample_avg[:, zero_cols] = 0.0
        if all_avg is not None:
            all_avg[:, zero_cols] = 0.0
    diagnostics = {
        "n_features": int(feature_weight_sums.shape[0]),
        "n_features_with_zero_weight": int(np.sum(feature_weight_sums <= 0)),
        "total_weight_sum": float(np.asarray(partial.get("total_weight_sum", 0.0)).reshape(())),
        "source": "weighted_shap_partials",
        "require_full": bool(require_full),
    }
    return oof_avg, insample_avg, all_avg, feature_weight_sums, diagnostics


__all__ = [
    "REAL_SHAP_TASKS_FILENAME",
    "PERM_SHAP_TASKS_FILENAME_TEMPLATE",
    "save_shap_task_archive",
    "load_shap_task_archive",
    "compute_weighted_shap_from_task_archives",
    "WEIGHTED_REAL_SHAP_PARTIAL_FILENAME",
    "WEIGHTED_PERM_SHAP_PARTIAL_FILENAME_TEMPLATE",
    "reduce_weighted_shap_task_archives_to_partial",
    "merge_weighted_shap_partials",
    "finalize_weighted_shap_partial",
]
