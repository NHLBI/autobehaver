"""Per-subset raw-margin artifacts for staged FeatureImp predictions.

This module supports a faster staged merge path for prediction aggregation.
Instead of reopening every saved XGBoost model during the final merge just to
recompute raw margins, shard jobs can persist per-subset margin sums/counts while
the model is already in memory. Merge then only needs to apply subset weights
and aggregate those saved subset-local tensors.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .logit_ensemble import LogitEnsemblePredictions, _sigmoid_stable


REAL_MARGIN_TASKS_FILENAME = "REAL_MARGIN_TASKS.npz"
PERM_MARGIN_TASKS_FILENAME_TEMPLATE = "PERM_MARGIN_TASKS_perm{perm_idx:03d}.npz"
WEIGHTED_REAL_MARGIN_PARTIAL_FILENAME = "WEIGHTED_REAL_MARGIN_PARTIAL.npz"


_REQUIRED_ARCHIVE_KEYS = (
    "h_hit",
    "subset_idx",
    "oof_margin_sum",
    "insample_margin_sum",
    "all_groups_margin_sum",
    "test_pred_count",
    "train_pred_count",
    "all_groups_model_count",
)


def _empty_margin_archive(*, n_train: int, n_all: int) -> Dict[str, np.ndarray]:
    return {
        "h_hit": np.zeros((0,), dtype=np.int32),
        "subset_idx": np.zeros((0,), dtype=np.int32),
        "oof_margin_sum": np.zeros((0, n_train), dtype=np.float64),
        "insample_margin_sum": np.zeros((0, n_train), dtype=np.float64),
        "all_groups_margin_sum": np.zeros((0, n_all), dtype=np.float64),
        "test_pred_count": np.zeros((0, n_train), dtype=np.int32),
        "train_pred_count": np.zeros((0, n_train), dtype=np.int32),
        "all_groups_model_count": np.zeros((0,), dtype=np.int32),
    }


def save_margin_task_archive(
    path: str,
    *,
    task_records: Sequence[Mapping[str, Any]],
    n_train: int,
    n_all: int,
    metadata: Optional[Mapping[str, Any]] = None,
) -> None:
    """Write one shard-local stacked archive of per-subset margin outputs."""

    if not task_records:
        arrays = _empty_margin_archive(n_train=int(n_train), n_all=int(n_all))
    else:
        arrays = {
            "h_hit": np.asarray([int(r["h_hit"]) for r in task_records], dtype=np.int32),
            "subset_idx": np.asarray([int(r["subset_idx"]) for r in task_records], dtype=np.int32),
            "oof_margin_sum": np.stack(
                [np.asarray(r["oof_margin_sum"], dtype=np.float64) for r in task_records], axis=0
            ),
            "insample_margin_sum": np.stack(
                [np.asarray(r["insample_margin_sum"], dtype=np.float64) for r in task_records], axis=0
            ),
            "all_groups_margin_sum": np.stack(
                [np.asarray(r["all_groups_margin_sum"], dtype=np.float64) for r in task_records], axis=0
            ),
            "test_pred_count": np.stack(
                [np.asarray(r["test_pred_count"], dtype=np.int32) for r in task_records], axis=0
            ),
            "train_pred_count": np.stack(
                [np.asarray(r["train_pred_count"], dtype=np.int32) for r in task_records], axis=0
            ),
            "all_groups_model_count": np.asarray(
                [int(r.get("all_groups_model_count", 0) or 0) for r in task_records], dtype=np.int32
            ),
        }

    if metadata:
        for key, value in metadata.items():
            arrays[f"meta__{key}"] = np.asarray(value)

    np.savez(path, **arrays)



def load_margin_task_archive(path: str) -> Dict[str, np.ndarray]:
    if not path:
        raise ValueError("Margin archive path is empty.")
    out: Dict[str, np.ndarray] = {}
    with np.load(path, allow_pickle=False) as data:
        for key in _REQUIRED_ARCHIVE_KEYS:
            if key not in data.files:
                raise ValueError(f"Margin archive missing required key '{key}': {path}")
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



def _load_task_records_from_archives(paths: Sequence[str]) -> Dict[Tuple[int, int], Dict[str, Any]]:
    task_map: Dict[Tuple[int, int], Dict[str, Any]] = {}
    for path in paths:
        if not path:
            continue
        payload = load_margin_task_archive(path)
        n_tasks = int(payload["h_hit"].shape[0])
        for idx in range(n_tasks):
            key = (int(payload["h_hit"][idx]), int(payload["subset_idx"][idx]))
            if key in task_map:
                raise RuntimeError(
                    f"Duplicate margin task entry encountered for subset {key} while reading archives."
                )
            task_map[key] = {
                "h_hit": int(payload["h_hit"][idx]),
                "subset_idx": int(payload["subset_idx"][idx]),
                "oof_margin_sum": np.asarray(payload["oof_margin_sum"][idx], dtype=np.float64),
                "insample_margin_sum": np.asarray(payload["insample_margin_sum"][idx], dtype=np.float64),
                "all_groups_margin_sum": np.asarray(payload["all_groups_margin_sum"][idx], dtype=np.float64),
                "test_pred_count": np.asarray(payload["test_pred_count"][idx], dtype=np.int32),
                "train_pred_count": np.asarray(payload["train_pred_count"][idx], dtype=np.int32),
                "all_groups_model_count": int(payload["all_groups_model_count"][idx]),
            }
    return task_map



def compute_logit_ensemble_predictions_from_margin_task_archives(
    *,
    archive_paths: Sequence[str],
    performance_records: List[Dict[str, Any]],
    n_train: int,
    n_all: int,
    num_classes: int,
    n_splits: int,
    subset_weights: Optional[Dict[Tuple[int, int], float]] = None,
    include_permuted: bool = False,
    permutation_idx: Optional[int] = None,
    verbose: bool = True,
) -> LogitEnsemblePredictions:
    """Aggregate saved raw margins from shard-local subset task archives.

    This mirrors the saved-model logit aggregation math, but uses per-subset
    raw-margin sums/counts captured during shard execution so merge does not
    need to reopen every XGBoost model.
    """

    if int(num_classes) != 2:
        raise ValueError(
            f"Margin-task logit aggregation currently supports binary classification only; got num_classes={num_classes}"
        )
    if not archive_paths:
        raise RuntimeError("No margin task archives were provided for aggregation.")

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
        key = (h_hit, subset_idx)
        if key not in subsets:
            subsets[key] = {}
        subsets[key].setdefault(fold_idx, dict(rec))

    task_map = _load_task_records_from_archives(archive_paths)

    oof_margin_sum = np.zeros(int(n_train), dtype=np.float64)
    oof_margin_count = np.zeros(int(n_train), dtype=np.int32)
    insample_margin_sum = np.zeros(int(n_train), dtype=np.float64)
    insample_margin_count = np.zeros(int(n_train), dtype=np.int32)
    all_margin_sum = np.zeros(int(n_all), dtype=np.float64)
    all_model_count = 0

    do_weighted = subset_weights is not None
    oof_margin_sum_w = np.zeros(int(n_train), dtype=np.float64) if do_weighted else None
    oof_weight_sum = np.zeros(int(n_train), dtype=np.float64) if do_weighted else None
    insample_margin_sum_w = np.zeros(int(n_train), dtype=np.float64) if do_weighted else None
    insample_weight_sum = np.zeros(int(n_train), dtype=np.float64) if do_weighted else None
    all_margin_sum_w = np.zeros(int(n_all), dtype=np.float64) if do_weighted else None
    all_weight_sum_scalar = 0.0

    processed_models = 0
    processed_subsets = 0
    skipped_incomplete_subsets = 0
    skipped_missing_margin_subsets = 0

    for (h_hit, subset_idx), fold_map in sorted(subsets.items()):
        if len(fold_map) != int(n_splits) or any(i not in fold_map for i in range(int(n_splits))):
            skipped_incomplete_subsets += 1
            continue

        task = task_map.get((int(h_hit), int(subset_idx)))
        if task is None:
            skipped_missing_margin_subsets += 1
            continue

        test_count = np.asarray(task["test_pred_count"], dtype=np.int32)
        train_count = np.asarray(task["train_pred_count"], dtype=np.int32)
        all_count = int(task["all_groups_model_count"])

        oof_margin_sum += np.asarray(task["oof_margin_sum"], dtype=np.float64)
        oof_margin_count += test_count
        insample_margin_sum += np.asarray(task["insample_margin_sum"], dtype=np.float64)
        insample_margin_count += train_count
        all_margin_sum += np.asarray(task["all_groups_margin_sum"], dtype=np.float64)
        all_model_count += all_count

        if do_weighted:
            w = float(subset_weights.get((int(h_hit), int(subset_idx)), 0.0))
            oof_margin_sum_w += w * np.asarray(task["oof_margin_sum"], dtype=np.float64)
            oof_weight_sum += w * test_count.astype(np.float64)

            w_in = w / float(n_splits - 1) if int(n_splits) > 1 else 0.0
            insample_margin_sum_w += w_in * np.asarray(task["insample_margin_sum"], dtype=np.float64)
            insample_weight_sum += w_in * train_count.astype(np.float64)

            w_all = w / float(n_splits) if int(n_splits) > 0 else 0.0
            all_margin_sum_w += w_all * np.asarray(task["all_groups_margin_sum"], dtype=np.float64)
            all_weight_sum_scalar += w_all * float(all_count)

        processed_models += int(all_count)
        processed_subsets += 1

    if verbose:
        print("\n--- Margin-task logit aggregation summary ---")
        print(f"  Eligible model records: {len(records)}")
        print(f"  Eligible subsets:       {len(subsets)}")
        print(f"  Processed subsets:      {processed_subsets}")
        print(f"  Processed fold-models:  {processed_models}")
        if skipped_incomplete_subsets:
            print(f"  Skipped subsets (incomplete folds): {skipped_incomplete_subsets}")
        if skipped_missing_margin_subsets:
            print(f"  Skipped subsets (missing margin archives): {skipped_missing_margin_subsets}")

    if skipped_missing_margin_subsets > 0:
        raise RuntimeError(
            f"Missing margin task archives for {int(skipped_missing_margin_subsets)} eligible complete subsets."
        )

    if processed_models == 0 or processed_subsets == 0 or all_model_count == 0:
        raise RuntimeError("No subsets were successfully aggregated from margin task archives.")

    if np.any(oof_margin_count == 0) or int(np.min(oof_margin_count)) != int(np.max(oof_margin_count)):
        raise RuntimeError(
            "Margin-task OOF aggregation produced uneven per-sample contribution counts. "
            "This indicates incomplete subset coverage or fold misalignment."
        )
    if np.any(insample_margin_count == 0) or int(np.min(insample_margin_count)) != int(np.max(insample_margin_count)):
        raise RuntimeError(
            "Margin-task in-sample aggregation produced uneven per-sample contribution counts. "
            "This indicates incomplete subset coverage or fold misalignment."
        )

    with np.errstate(divide="ignore", invalid="ignore"):
        oof_margin_mean = np.where(oof_margin_count > 0, oof_margin_sum / oof_margin_count, np.nan)
        insample_margin_mean = np.where(insample_margin_count > 0, insample_margin_sum / insample_margin_count, np.nan)
        all_margin_mean = all_margin_sum / float(all_model_count)

    p1_oof = _sigmoid_stable(oof_margin_mean)
    p1_in = _sigmoid_stable(insample_margin_mean)
    p1_all = _sigmoid_stable(all_margin_mean)

    out = LogitEnsemblePredictions(
        insample_probas=np.stack([1.0 - p1_in, p1_in], axis=1).astype(np.float64),
        oof_probas=np.stack([1.0 - p1_oof, p1_oof], axis=1).astype(np.float64),
        all_groups_probas=np.stack([1.0 - p1_all, p1_all], axis=1).astype(np.float64),
        diagnostics={
            "processed_subsets": int(processed_subsets),
            "processed_models": int(processed_models),
            "eligible_subsets": int(len(subsets)),
            "skipped_incomplete_subsets": int(skipped_incomplete_subsets),
            "skipped_missing_margin_subsets": int(skipped_missing_margin_subsets),
            "all_model_count": int(all_model_count),
            "do_weighted": bool(do_weighted),
            "include_permuted": bool(include_permuted),
            "source": "margin_task_archives",
        },
    )

    if do_weighted:
        if all_weight_sum_scalar <= 0:
            if verbose:
                print(
                    "WARNING: subset_weights provided but sum of weights is 0. "
                    "Weighted predictions will be omitted (None)."
                )
            return out

        with np.errstate(divide="ignore", invalid="ignore"):
            oof_margin_mean_w = np.where(oof_weight_sum > 0, oof_margin_sum_w / oof_weight_sum, np.nan)
            insample_margin_mean_w = np.where(
                insample_weight_sum > 0,
                insample_margin_sum_w / insample_weight_sum,
                np.nan,
            )
            all_margin_mean_w = all_margin_sum_w / float(all_weight_sum_scalar)

        p1_oof_w = _sigmoid_stable(oof_margin_mean_w)
        p1_in_w = _sigmoid_stable(insample_margin_mean_w)
        p1_all_w = _sigmoid_stable(all_margin_mean_w)

        out.oof_probas_weighted = np.stack([1.0 - p1_oof_w, p1_oof_w], axis=1).astype(np.float64)
        out.insample_probas_weighted = np.stack([1.0 - p1_in_w, p1_in_w], axis=1).astype(np.float64)
        out.all_groups_probas_weighted = np.stack([1.0 - p1_all_w, p1_all_w], axis=1).astype(np.float64)
        if out.diagnostics is None:
            out.diagnostics = {}
        out.diagnostics["weight_sum_scalar"] = float(all_weight_sum_scalar)

    return out


def reduce_weighted_margin_task_archives_to_partial(
    *,
    archive_paths: Sequence[str],
    performance_records: Sequence[Mapping[str, Any]],
    subset_weights: Mapping[Tuple[int, int], float],
    n_train: int,
    n_all: int,
    n_splits: int,
    include_permuted: bool = False,
    permutation_idx: Optional[int] = None,
    verbose: bool = True,
) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    """Reduce one shard's saved margin task archives into weighted additive partials."""

    if int(n_splits) < 2:
        raise ValueError(f"n_splits must be >= 2, got {n_splits}")
    if not archive_paths:
        raise RuntimeError("No margin task archives were provided for weighted shard reduction.")

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

    task_map = _load_task_records_from_archives(archive_paths)

    oof_margin_num = np.zeros(int(n_train), dtype=np.float64)
    oof_weight_den = np.zeros(int(n_train), dtype=np.float64)
    insample_margin_num = np.zeros(int(n_train), dtype=np.float64)
    insample_weight_den = np.zeros(int(n_train), dtype=np.float64)
    all_margin_num = np.zeros(int(n_all), dtype=np.float64)
    all_weight_den_scalar = np.asarray(0.0, dtype=np.float64)

    processed_subsets = 0
    skipped_incomplete_subsets = 0
    skipped_missing_margin_subsets = 0
    skipped_zero_weight_subsets = 0
    processed_models = 0

    for (h_hit, subset_idx), fold_map in sorted(subsets.items()):
        if len(fold_map) != int(n_splits) or any(i not in fold_map for i in range(int(n_splits))):
            skipped_incomplete_subsets += 1
            continue
        w = float(subset_weights.get((int(h_hit), int(subset_idx)), 0.0))
        if not np.isfinite(w) or w <= 0.0:
            skipped_zero_weight_subsets += 1
            continue
        task = task_map.get((int(h_hit), int(subset_idx)))
        if task is None:
            skipped_missing_margin_subsets += 1
            continue

        test_count = np.asarray(task["test_pred_count"], dtype=np.int32)
        train_count = np.asarray(task["train_pred_count"], dtype=np.int32)
        all_count = int(task["all_groups_model_count"])

        oof_margin_num += w * np.asarray(task["oof_margin_sum"], dtype=np.float64)
        oof_weight_den += w * test_count.astype(np.float64)

        w_in = w / float(max(int(n_splits) - 1, 1))
        insample_margin_num += w_in * np.asarray(task["insample_margin_sum"], dtype=np.float64)
        insample_weight_den += w_in * train_count.astype(np.float64)

        w_all = w / float(max(int(n_splits), 1))
        all_margin_num += w_all * np.asarray(task["all_groups_margin_sum"], dtype=np.float64)
        all_weight_den_scalar = np.asarray(float(all_weight_den_scalar.reshape(())) + (w_all * float(all_count)), dtype=np.float64)

        processed_subsets += 1
        processed_models += int(all_count)

    if skipped_missing_margin_subsets > 0:
        raise RuntimeError(
            f"Missing margin task archives for {int(skipped_missing_margin_subsets)} eligible complete subsets."
        )

    if verbose:
        print("\n--- Weighted shard margin reduction summary ---")
        print(f"  Eligible model records: {len(records)}")
        print(f"  Eligible subsets:       {len(subsets)}")
        print(f"  Processed subsets:      {processed_subsets}")
        print(f"  Processed fold-models:  {processed_models}")
        if skipped_incomplete_subsets:
            print(f"  Skipped subsets (incomplete folds): {skipped_incomplete_subsets}")
        if skipped_zero_weight_subsets:
            print(f"  Skipped subsets (zero weight):      {skipped_zero_weight_subsets}")

    partial = {
        "oof_margin_num": np.asarray(oof_margin_num, dtype=np.float64),
        "oof_weight_den": np.asarray(oof_weight_den, dtype=np.float64),
        "insample_margin_num": np.asarray(insample_margin_num, dtype=np.float64),
        "insample_weight_den": np.asarray(insample_weight_den, dtype=np.float64),
        "all_margin_num": np.asarray(all_margin_num, dtype=np.float64),
        "all_weight_den_scalar": np.asarray(all_weight_den_scalar, dtype=np.float64),
    }
    diagnostics = {
        "n_train": int(n_train),
        "n_all": int(n_all),
        "n_splits": int(n_splits),
        "n_subsets_total": int(len(subsets)),
        "n_subsets_used": int(processed_subsets),
        "n_subsets_skipped_incomplete": int(skipped_incomplete_subsets),
        "n_subsets_skipped_zero_weight": int(skipped_zero_weight_subsets),
        "n_subsets_skipped_missing_archive": int(skipped_missing_margin_subsets),
        "processed_models": int(processed_models),
        "include_permuted": bool(include_permuted),
        "source": "weighted_margin_shard_reduce",
    }
    return partial, diagnostics


def merge_weighted_margin_partials(paths: Sequence[str]) -> Dict[str, np.ndarray]:
    if not paths:
        raise ValueError("No weighted margin partial paths were provided for merge.")
    merged: Optional[Dict[str, np.ndarray]] = None
    required_keys = (
        "oof_margin_num",
        "oof_weight_den",
        "insample_margin_num",
        "insample_weight_den",
        "all_margin_num",
        "all_weight_den_scalar",
    )
    for path in paths:
        if not path:
            continue
        with np.load(path, allow_pickle=False) as data:
            payload = {key: np.asarray(data[key]) for key in required_keys}
        if merged is None:
            merged = {k: np.array(v, copy=True) for k, v in payload.items()}
            continue
        for key, value in payload.items():
            if merged[key].shape != value.shape:
                raise ValueError(
                    f"Weighted margin partial shape mismatch for key '{key}': expected {merged[key].shape}, got {value.shape}."
                )
            merged[key] = merged[key] + value
    if merged is None:
        raise ValueError("No readable weighted margin partial payloads were found.")
    return merged


def finalize_weighted_margin_partial(
    *,
    partial: Mapping[str, Any],
    num_classes: int,
) -> LogitEnsemblePredictions:
    if int(num_classes) != 2:
        raise ValueError(
            f"Weighted margin partial finalization currently supports binary classification only; got num_classes={num_classes}"
        )

    oof_margin_num = np.asarray(partial["oof_margin_num"], dtype=np.float64)
    oof_weight_den = np.asarray(partial["oof_weight_den"], dtype=np.float64)
    insample_margin_num = np.asarray(partial["insample_margin_num"], dtype=np.float64)
    insample_weight_den = np.asarray(partial["insample_weight_den"], dtype=np.float64)
    all_margin_num = np.asarray(partial["all_margin_num"], dtype=np.float64)
    all_weight_den_scalar = float(np.asarray(partial["all_weight_den_scalar"], dtype=np.float64).reshape(()))

    with np.errstate(divide="ignore", invalid="ignore"):
        oof_margin_mean = np.where(oof_weight_den > 0, oof_margin_num / oof_weight_den, np.nan)
        insample_margin_mean = np.where(
            insample_weight_den > 0,
            insample_margin_num / insample_weight_den,
            np.nan,
        )
        all_margin_mean = all_margin_num / float(all_weight_den_scalar) if all_weight_den_scalar > 0 else np.zeros_like(all_margin_num)

    p1_oof = _sigmoid_stable(oof_margin_mean)
    p1_in = _sigmoid_stable(insample_margin_mean)
    p1_all = _sigmoid_stable(all_margin_mean)

    return LogitEnsemblePredictions(
        insample_probas=np.stack([1.0 - p1_in, p1_in], axis=1).astype(np.float64),
        oof_probas=np.stack([1.0 - p1_oof, p1_oof], axis=1).astype(np.float64),
        all_groups_probas=np.stack([1.0 - p1_all, p1_all], axis=1).astype(np.float64),
        diagnostics={
            "source": "weighted_margin_partials",
            "all_weight_den_scalar": float(all_weight_den_scalar),
        },
    )


__all__ = [
    "REAL_MARGIN_TASKS_FILENAME",
    "PERM_MARGIN_TASKS_FILENAME_TEMPLATE",
    "save_margin_task_archive",
    "load_margin_task_archive",
    "compute_logit_ensemble_predictions_from_margin_task_archives",
    "reduce_weighted_margin_task_archives_to_partial",
    "merge_weighted_margin_partials",
    "finalize_weighted_margin_partial",
]
