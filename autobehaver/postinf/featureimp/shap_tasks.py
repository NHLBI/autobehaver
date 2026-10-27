"""Store and reduce subset-level SHAP sums from FeatureImp shards.

Real runs support out-of-fold, in-sample, and all-groups views; permutation baselines
can use out-of-fold SHAP alone.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from .integrity import strict_bool


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
    """Load and validate required subset SHAP arrays from a task archive."""
    if not path:
        raise ValueError("SHAP task archive path is empty.")
    from .archive_io import load_npz_arrays
    return load_npz_arrays(path, _REQUIRED_ARCHIVE_KEYS,
        ('insample_shap_flat', 'all_groups_shap_flat', 'meta__n_train', 'meta__n_all',
         'meta__include_insample', 'meta__include_all_groups'))


def _iter_relevant_model_records(
    performance_records: Iterable[Mapping[str, Any]],
    *,
    include_permuted: bool = False,
    permutation_idx: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Select real models or the requested permutation models from performance records."""
    out: List[Dict[str, Any]] = []
    for rec in performance_records:
        if not isinstance(rec, Mapping):
            continue
        is_perm = strict_bool(rec.get("is_permuted", False))
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
    """Read unique subset SHAP records and validate their sample dimensions."""
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
    """Aggregate checked subset SHAP arrays with feature-specific weight denominators."""
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
        fold_map = subsets.setdefault((h_hit, subset_idx), {})
        if fold_idx in fold_map:
            raise ValueError("Duplicate task fold identity")
        fold_map[fold_idx] = dict(rec)
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
            raise ValueError("SHAP task archive subset lacks its complete fold identities")
        w = float(subset_weights.get((int(h_hit), int(subset_idx)), 0.0))
        if not np.isfinite(w) or w < 0:
            raise ValueError("Subset weights must be nonnegative finite")
        if w == 0.0:
            n_subsets_skipped_zero_w += 1
            continue
        task = task_map.get((int(h_hit), int(subset_idx)))
        if task is None:
            n_subsets_skipped_missing_archive += 1
            continue
        feat_idxs = np.asarray(task["feature_indices"], dtype=np.int64)
        if feat_idxs.ndim != 1 or len(feat_idxs) == 0 or len(set(feat_idxs.tolist())) != len(feat_idxs) or np.any(feat_idxs < 0) or np.any(feat_idxs >= n_features):
            raise ValueError("SHAP task feature indices are duplicate/outside the feature schema")
        for value_key, rows in (("oof_shap_sum", n_train), ("insample_shap_sum", n_train), ("all_groups_shap_sum", n_all)):
            if value_key in task:
                value_array = np.asarray(task[value_key])
                if value_array.shape != (int(rows), len(feat_idxs)) or not np.isfinite(value_array).all():
                    raise ValueError("SHAP task values are nonfinite or misaligned")

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
        fold_map = subsets.setdefault((h_hit, subset_idx), {})
        if fold_idx in fold_map:
            raise ValueError("Duplicate task fold identity")
        fold_map[fold_idx] = dict(rec)

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
            raise ValueError("SHAP task archive subset lacks its complete fold identities")
        w = float(subset_weights.get((int(h_hit), int(subset_idx)), 0.0))
        if not np.isfinite(w) or w < 0:
            raise ValueError("Subset weights must be nonnegative finite")
        if w == 0.0:
            n_subsets_skipped_zero_w += 1
            continue
        task = task_map.get((int(h_hit), int(subset_idx)))
        if task is None:
            n_subsets_skipped_missing_archive += 1
            continue

        feat_idxs = np.asarray(task["feature_indices"], dtype=np.int64)
        if feat_idxs.ndim != 1 or len(feat_idxs) == 0 or len(set(feat_idxs.tolist())) != len(feat_idxs) or np.any(feat_idxs < 0) or np.any(feat_idxs >= n_features):
            raise ValueError("SHAP task feature indices are duplicate/outside the feature schema")
        for value_key, rows in (("oof_shap_sum", n_train), ("insample_shap_sum", n_train), ("all_groups_shap_sum", n_all)):
            if value_key in task:
                value_array = np.asarray(task[value_key])
                if value_array.shape != (int(rows), len(feat_idxs)) or not np.isfinite(value_array).all():
                    raise ValueError("SHAP task values are nonfinite or misaligned")

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
    """Add compatible weighted-SHAP partials for the requested output views."""
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
    """Normalize merged SHAP numerators by per-feature support.

    Reject unsupported features in a positive-weight ensemble and retain the defined
    all-zero-weight fallback.
    """
    feature_weight_sums = np.asarray(partial["feature_weight_den"], dtype=np.float32)
    if feature_weight_sums.ndim != 1 or not np.isfinite(feature_weight_sums).all() or np.any(feature_weight_sums < 0):
        raise ValueError("SHAP feature support must be a nonnegative finite vector")
    if int(n_splits) < 2:
        raise ValueError("SHAP reduction requires at least two folds")
    unsupported = feature_weight_sums == 0
    required = ['oof_shap_num'] + (['insample_shap_num', 'all_shap_num'] if require_full else [])
    for key in required:
        array = np.asarray(partial[key])
        if array.ndim != 2 or array.shape[1] != len(feature_weight_sums) or not np.isfinite(array).all():
            raise ValueError(f"Invalid SHAP numerator: {key}")
        if np.any(array[:, unsupported] != 0):
            raise ValueError(f"Nonzero SHAP numerator without feature support: {key}")
    denom_safe = np.where(feature_weight_sums > 0, feature_weight_sums, 1.0).astype(np.float32)
    oof_avg = np.asarray(partial["oof_shap_num"], dtype=np.float32) / denom_safe[None, :]
    insample_avg: Optional[np.ndarray] = None
    all_avg: Optional[np.ndarray] = None
    if require_full:
        insample_avg = np.asarray(partial["insample_shap_num"], dtype=np.float32) / (denom_safe * np.float32(max(int(n_splits) - 1, 1)))[None, :]
        all_avg = np.asarray(partial["all_shap_num"], dtype=np.float32) / denom_safe[None, :]
    zero_cols = feature_weight_sums <= 0
    if np.any(zero_cols):
        oof_avg[:, zero_cols] = 0.0
        if insample_avg is not None:
            insample_avg[:, zero_cols] = 0.0
        if all_avg is not None:
            all_avg[:, zero_cols] = 0.0
    diagnostics = {
        "n_features": int(feature_weight_sums.shape[0]),
        "n_features_with_zero_weight": int(np.sum(feature_weight_sums <= 0)),
        "unsupported_feature_indices": np.flatnonzero(unsupported).tolist(),
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
