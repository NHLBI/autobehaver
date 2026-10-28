"""Aggregate binary specialist predictions by averaging raw margins before applying
sigmoid.

Group A is class 0 and group B is class 1; optional weights apply per complete feature
subset.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import os
from .integrity import strict_bool
import numpy as np
import pandas as pd
import xgboost as xgb

from .hhit import make_hhit_folds
from .permutation import _generate_sex_class_stratified_permutation


def _sigmoid_stable(x: np.ndarray) -> np.ndarray:
    """Apply a numerically stable sigmoid to raw margins."""

    x = np.asarray(x, dtype=np.float64)
    out = np.empty_like(x, dtype=np.float64)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    exp_x = np.exp(x[~pos])
    out[~pos] = exp_x / (1.0 + exp_x)
    return out


def load_classifier_model(path: str):
    """Load persisted classifier metadata across sklearn's estimator-tag API change."""
    model = xgb.XGBClassifier()
    model._estimator_type = 'classifier'
    model.load_model(str(path))
    return model


@dataclass
class LogitEnsemblePredictions:
    """Container for logit-aggregated committee predictions."""

    # Unweighted
    insample_probas: np.ndarray  # (n_train, 2)
    oof_probas: np.ndarray  # (n_train, 2)
    all_groups_probas: np.ndarray  # (n_all, 2)

    # Weighted (optional)
    insample_probas_weighted: Optional[np.ndarray] = None
    oof_probas_weighted: Optional[np.ndarray] = None
    all_groups_probas_weighted: Optional[np.ndarray] = None

    diagnostics: Optional[Dict[str, Any]] = None


def load_subset_weights_mapping(weights_csv_path: str) -> Dict[Tuple[int, int], float]:
    """Load subset weights from SUBSET_WEIGHTS_REAL.csv into a mapping.

    Returns mapping keyed by (h_hit, subset_idx) -> subset_weight.
    """

    df = pd.read_csv(weights_csv_path)
    required = {"h_hit", "subset_idx", "subset_weight"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Weights CSV missing required columns: {sorted(missing)} (path={weights_csv_path})"
        )
    if df.duplicated(subset=['h_hit', 'subset_idx']).any():
        raise ValueError("Duplicate subset identities in weights CSV")
    weights = pd.to_numeric(df['subset_weight'], errors='raise').to_numpy(dtype=float)
    if not np.isfinite(weights).all() or np.any(weights < 0):
        raise ValueError("Subset weights must be nonnegative and finite")

    m: Dict[Tuple[int, int], float] = {}
    for _, row in df.iterrows():
        key = (int(row["h_hit"]), int(row["subset_idx"]))
        w = float(row["subset_weight"])
        m[key] = w
    return m


def _iter_relevant_model_records(
    performance_records: Iterable[Dict[str, Any]],
    *,
    include_permuted: bool = False,
    permutation_idx: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Filter performance records down to the models that drive the committee.

    By default, this yields only *real* models (is_permuted=False).
    When include_permuted=True, it yields only permuted models, and optionally
    filters to a single permutation_idx.
    """

    out: List[Dict[str, Any]] = []
    for rec in performance_records:
        if not isinstance(rec, dict):
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


        out.append(rec)

    return out


def _build_permuted_train_flip_masks(
    *,
    y_original_labels_train: np.ndarray,
    sex_labels_train: np.ndarray,
    folds: List[Tuple[np.ndarray, np.ndarray]],
    h_hit: int,
    permutation_idx: int,
    GLOBAL_RANDOM_SEED: int,
) -> Dict[int, np.ndarray]:
    """Reconstruct training-fold label swaps from saved permutation seeds.

    Negate the corresponding margins before aggregation to express in-sample predictions
    in the original group A/B frame.
    """

    y_original_labels_train = np.asarray(y_original_labels_train)
    sex_labels_train = np.asarray(sex_labels_train)

    if y_original_labels_train.ndim != 1:
        raise ValueError(
            f"y_original_labels_train must be 1D, got shape={y_original_labels_train.shape}"
        )
    if sex_labels_train.ndim != 1:
        raise ValueError(
            f"sex_labels_train must be 1D, got shape={sex_labels_train.shape}"
        )
    if len(y_original_labels_train) != len(sex_labels_train):
        raise ValueError(
            "y_original_labels_train and sex_labels_train must have identical lengths "
            f"(got {len(y_original_labels_train)} and {len(sex_labels_train)})."
        )

    df_with_sex = pd.DataFrame({"sex": sex_labels_train})
    out: Dict[int, np.ndarray] = {}

    for fold_idx, (train_idx, test_idx) in enumerate(folds):
        perm_seed = int(GLOBAL_RANDOM_SEED) + int(permutation_idx) * 100000 + int(h_hit) * 1000 + int(fold_idx)
        y_permuted_full = _generate_sex_class_stratified_permutation(
            y_original=y_original_labels_train,
            df_with_sex=df_with_sex,
            train_indices=train_idx,
            test_indices=test_idx,
            permutation_seed=perm_seed,
        )
        out[int(fold_idx)] = np.asarray(y_original_labels_train != np.asarray(y_permuted_full), dtype=bool)

    return out


def compute_logit_ensemble_predictions_from_saved_models(
    *,
    performance_records: List[Dict[str, Any]],
    X_train_full: pd.DataFrame,
    X_all_groups: pd.DataFrame,
    y_encoded_train: np.ndarray,
    sex_group_encoded: np.ndarray,
    num_classes: int,
    n_splits: int,
    GLOBAL_RANDOM_SEED: Optional[int],
    subset_weights: Optional[Dict[Tuple[int, int], float]] = None,
    include_permuted: bool = False,
    permutation_idx: Optional[int] = None,
    y_original_labels_train: Optional[np.ndarray] = None,
    sex_labels_train: Optional[np.ndarray] = None,
    verbose: bool = True,
) -> LogitEnsemblePredictions:
    """Re-score saved specialist models and aggregate predictions in logit space.

    This does **not** retrain models; it loads each model from `model_path` in
    the performance records.

    Weighting is optional. When provided, weighting is applied at the subset
    (h_hit, subset_idx) level.

    For permuted committees (`include_permuted=True`), callers must also pass
    `y_original_labels_train` and `sex_labels_train` so the exact TRAIN-fold
    label swaps can be reconstructed. Without those, in-sample permuted
    predictions cannot be re-expressed in the original class frame safely.
    """

    if int(num_classes) != 2:
        raise ValueError(
            f"Logit ensemble aggregation currently supports binary classification only; got num_classes={num_classes}" 
        )

    y_encoded_train = np.asarray(y_encoded_train)
    sex_group_encoded = np.asarray(sex_group_encoded)

    n_train = int(X_train_full.shape[0])
    n_all = int(X_all_groups.shape[0])

    if include_permuted and permutation_idx is None:
        raise ValueError(
            "permutation_idx must be provided when include_permuted=True "
            "(cannot safely mix permutations with different fold seeds)."
        )
    if include_permuted and GLOBAL_RANDOM_SEED is None:
        raise ValueError(
            "Cannot reconstruct permuted in-sample predictions from saved models when "
            "GLOBAL_RANDOM_SEED is None. Use a deterministic seed, or fall back to the "
            "original permuted export files."
        )
    if include_permuted and y_original_labels_train is None:
        raise ValueError(
            "include_permuted=True requires y_original_labels_train so TRAIN-fold label "
            "swaps can be reconstructed during re-scoring."
        )
    if include_permuted and sex_labels_train is None:
        raise ValueError(
            "include_permuted=True requires sex_labels_train so TRAIN-fold label swaps "
            "can be reconstructed during re-scoring."
        )

    if y_original_labels_train is not None:
        y_original_labels_train = np.asarray(y_original_labels_train)
        if y_original_labels_train.ndim != 1:
            raise ValueError(
                f"y_original_labels_train must be 1D, got shape={y_original_labels_train.shape}"
            )
        if len(y_original_labels_train) != n_train:
            raise ValueError(
                "y_original_labels_train must align to X_train_full "
                f"(got len={len(y_original_labels_train)} vs n_train={n_train})."
            )
    if sex_labels_train is not None:
        sex_labels_train = np.asarray(sex_labels_train)
        if sex_labels_train.ndim != 1:
            raise ValueError(
                f"sex_labels_train must be 1D, got shape={sex_labels_train.shape}"
            )
        if len(sex_labels_train) != n_train:
            raise ValueError(
                "sex_labels_train must align to X_train_full "
                f"(got len={len(sex_labels_train)} vs n_train={n_train})."
            )

    records = _iter_relevant_model_records(
        performance_records,
        include_permuted=include_permuted,
        permutation_idx=permutation_idx,
    )
    if len(records) == 0:
        raise ValueError("No eligible performance records found for logit aggregation.")

    # Group records by (h_hit, subset_idx) and enforce **subset completeness**.
    #
    # A "subset" is only usable if we have all folds (0..n_splits-1) and all
    # corresponding model files exist. This prevents silent drift where the
    # ensemble aggregates partial subsets (e.g., when some workers failed).
    subsets: Dict[Tuple[int, int], Dict[int, Dict[str, Any]]] = {}
    for rec in records:
        try:
            h_hit = int(rec.get("h_hit"))
            subset_idx = int(rec.get("subset_idx"))
            fold_idx = int(rec.get("fold_idx"))
        except Exception:
            continue
        if fold_idx < 0:
            continue
        key = (h_hit, subset_idx)
        if key not in subsets:
            subsets[key] = {}
        if fold_idx in subsets[key]:
            raise ValueError(f"Duplicate saved-model fold identity: {key}, fold {fold_idx}")
        subsets[key][fold_idx] = rec

    # Fold indices are deterministic per h_hit seed.
    folds_by_hhit: Dict[int, List[Tuple[np.ndarray, np.ndarray]]] = {}
    permuted_flip_masks_by_hhit: Dict[int, Dict[int, np.ndarray]] = {}

    # Unweighted accumulators
    oof_margin_sum = np.zeros(n_train, dtype=np.float64)
    oof_margin_count = np.zeros(n_train, dtype=np.int32)

    insample_margin_sum = np.zeros(n_train, dtype=np.float64)
    insample_margin_count = np.zeros(n_train, dtype=np.int32)

    all_margin_sum = np.zeros(n_all, dtype=np.float64)
    all_model_count = 0

    # Weighted accumulators (optional)
    do_weighted = subset_weights is not None

    oof_margin_sum_w = np.zeros(n_train, dtype=np.float64) if do_weighted else None
    oof_weight_sum = np.zeros(n_train, dtype=np.float64) if do_weighted else None

    # For weighted in-sample, weight per fold model is w/(n_splits-1) so each subset
    # contributes total weight w.
    insample_margin_sum_w = np.zeros(n_train, dtype=np.float64) if do_weighted else None
    insample_weight_sum = np.zeros(n_train, dtype=np.float64) if do_weighted else None

    all_margin_sum_w = np.zeros(n_all, dtype=np.float64) if do_weighted else None
    all_weight_sum_scalar = 0.0

    skipped_incomplete_subsets = 0
    skipped_missing_model_subsets = 0
    skipped_bad_feature_subsets = 0

    processed_models = 0
    processed_subsets = 0

    for (h_hit, subset_idx), fold_map in sorted(subsets.items()):
        # Require folds 0..n_splits-1 for subset completeness.
        if len(fold_map) != int(n_splits) or any(i not in fold_map for i in range(int(n_splits))):
            raise ValueError(f"Saved-model subset {(h_hit, subset_idx)} lacks its complete folds")

        # Resolve + validate feature list (must be consistent across folds).
        features = None
        for i in range(int(n_splits)):
            rec = fold_map[i]
            feats = rec.get("specialist_features")
            if feats is None:
                feats = rec.get("features")
            if isinstance(feats, (pd.Index, np.ndarray, tuple)):
                feats = list(feats)
            if not isinstance(feats, list) or len(feats) == 0:
                features = None
                break
            if features is None:
                features = list(feats)
            else:
                if list(feats) != list(features):
                    # Feature order matters for XGBoost; inconsistent records imply corruption.
                    features = None
                    break
        if features is None:
            raise ValueError(f"Saved-model subset {(h_hit, subset_idx)} has inconsistent feature identities/order")
        if len(set(features)) != len(features):
            raise ValueError("Saved-model subset has duplicate features")

        # Ensure all model paths exist.
        model_paths: List[str] = []
        missing = False
        for i in range(int(n_splits)):
            mp = fold_map[i].get("model_path")
            if not mp or not isinstance(mp, str) or not os.path.exists(mp):
                missing = True
                break
            model_paths.append(str(mp))
        if missing:
            raise ValueError(f"Saved-model subset {(h_hit, subset_idx)} lacks a required model file")

        # Folds for this h_hit (deterministic per seed).
        if h_hit not in folds_by_hhit:
            # Fold indices must match the seed used during training.
            # - Real committees:  GLOBAL_RANDOM_SEED + h_hit*1000
            # - Perm committees:  GLOBAL_RANDOM_SEED + perm_idx*100000 + h_hit*1000
            if GLOBAL_RANDOM_SEED is None:
                split_seed = None
            else:
                split_seed = GLOBAL_RANDOM_SEED + h_hit * 1000
                if include_permuted and permutation_idx is not None:
                    split_seed = GLOBAL_RANDOM_SEED + int(permutation_idx) * 100000 + h_hit * 1000
            folds_by_hhit[h_hit] = make_hhit_folds(
                X_train_full,
                sex_group_encoded,
                n_splits=n_splits,
                random_state=split_seed,
            )
            if include_permuted:
                permuted_flip_masks_by_hhit[h_hit] = _build_permuted_train_flip_masks(
                    y_original_labels_train=np.asarray(y_original_labels_train),
                    sex_labels_train=np.asarray(sex_labels_train),
                    folds=folds_by_hhit[h_hit],
                    h_hit=h_hit,
                    permutation_idx=int(permutation_idx),
                    GLOBAL_RANDOM_SEED=int(GLOBAL_RANDOM_SEED),
                )

        # Slice once per subset.
        X_train_sub = X_train_full[features]
        X_all_sub = X_all_groups[features]

        # Pre-fetch subset weight (applied per fold below).
        w_subset = float(subset_weights.get((h_hit, subset_idx), 0.0)) if do_weighted else 0.0

        for fold_idx in range(int(n_splits)):
            try:
                train_idx, test_idx = folds_by_hhit[h_hit][fold_idx]
            except Exception as exc:
                raise ValueError("Saved-model subset lacks its required fold indices") from exc

            model = load_classifier_model(model_paths[fold_idx])

            margin_train_full = model.predict(X_train_sub, output_margin=True)
            margin_all_full = model.predict(X_all_sub, output_margin=True)

            margin_train_full = np.asarray(margin_train_full, dtype=np.float64)
            margin_all_full = np.asarray(margin_all_full, dtype=np.float64)
            if margin_train_full.shape != (n_train,) or margin_all_full.shape != (n_all,) or not np.isfinite(margin_train_full).all() or not np.isfinite(margin_all_full).all():
                raise ValueError("Required saved-model margins have invalid shape/nonfinite values")

            # For permuted committees, TRAIN-fold margins must be negated for the
            # samples whose labels were swapped during training so the aggregated
            # in-sample probabilities remain in the original A/B label frame.
            margin_train_for_insample = margin_train_full[train_idx]
            if include_permuted:
                flip_mask_full = permuted_flip_masks_by_hhit[h_hit][fold_idx]
                flip_mask_train = np.asarray(flip_mask_full[train_idx], dtype=bool)
                if np.any(flip_mask_train):
                    margin_train_for_insample = np.asarray(margin_train_for_insample, dtype=np.float64).copy()
                    margin_train_for_insample[flip_mask_train] *= -1.0

            # Unweighted aggregation
            oof_margin_sum[test_idx] += margin_train_full[test_idx]
            oof_margin_count[test_idx] += 1

            insample_margin_sum[train_idx] += margin_train_for_insample
            insample_margin_count[train_idx] += 1

            all_margin_sum += margin_all_full
            all_model_count += 1

            # Weighted aggregation
            if do_weighted:
                w = w_subset

                # OOF: each subset contributes exactly one fold prediction per sample.
                oof_margin_sum_w[test_idx] += w * margin_train_full[test_idx]
                oof_weight_sum[test_idx] += w

                # In-sample: each subset contributes (n_splits-1) fold predictions per sample.
                # Normalize so each subset has total weight w.
                if n_splits > 1:
                    w_in = w / float(n_splits - 1)
                else:
                    w_in = 0.0
                insample_margin_sum_w[train_idx] += w_in * margin_train_for_insample
                insample_weight_sum[train_idx] += w_in

                # All-groups: average across folds within subset, then apply w.
                w_all = w / float(n_splits) if n_splits > 0 else 0.0
                all_margin_sum_w += w_all * margin_all_full
                all_weight_sum_scalar += w_all

            processed_models += 1

        if missing:
            skipped_incomplete_subsets += 1
            continue

        processed_subsets += 1

    if verbose:
        print("\n--- Logit ensemble re-score summary ---")
        print(f"  Eligible model records: {len(records)}")
        print(f"  Eligible subsets:       {len(subsets)}")
        print(f"  Processed subsets:      {processed_subsets}")
        print(f"  Processed fold-models:  {processed_models}")
        if skipped_incomplete_subsets:
            print(f"  Skipped subsets (incomplete folds): {skipped_incomplete_subsets}")
        if skipped_missing_model_subsets:
            print(f"  Skipped subsets (missing model files): {skipped_missing_model_subsets}")
        if skipped_bad_feature_subsets:
            print(f"  Skipped subsets (bad/inconsistent feature lists): {skipped_bad_feature_subsets}")

    if processed_models == 0 or processed_subsets == 0 or all_model_count == 0:
        raise RuntimeError("No models were successfully re-scored for logit aggregation.")

    # With subset completeness enforced, every training sample should have the
    # same number of OOF contributions (one per subset). If not, something is
    # misaligned and we should not silently proceed.
    if np.any(oof_margin_count == 0) or int(np.min(oof_margin_count)) != int(np.max(oof_margin_count)):
        raise RuntimeError(
            "Logit OOF aggregation produced uneven per-sample contribution counts. "
            "This indicates incomplete subset coverage or fold misalignment."
        )
    if np.any(insample_margin_count == 0) or int(np.min(insample_margin_count)) != int(np.max(insample_margin_count)):
        raise RuntimeError(
            "Logit in-sample aggregation produced uneven per-sample contribution counts. "
            "This indicates incomplete subset coverage or fold misalignment."
        )

    # Finalize means
    with np.errstate(divide="ignore", invalid="ignore"):
        oof_margin_mean = np.where(oof_margin_count > 0, oof_margin_sum / oof_margin_count, np.nan)
        insample_margin_mean = np.where(
            insample_margin_count > 0, insample_margin_sum / insample_margin_count, np.nan
        )
        all_margin_mean = all_margin_sum / float(all_model_count)

    p1_oof = _sigmoid_stable(oof_margin_mean)
    p1_in = _sigmoid_stable(insample_margin_mean)
    p1_all = _sigmoid_stable(all_margin_mean)

    # Shape (n,2) with class0 then class1
    oof_probas = np.stack([1.0 - p1_oof, p1_oof], axis=1).astype(np.float64)
    insample_probas = np.stack([1.0 - p1_in, p1_in], axis=1).astype(np.float64)
    all_groups_probas = np.stack([1.0 - p1_all, p1_all], axis=1).astype(np.float64)

    out = LogitEnsemblePredictions(
        insample_probas=insample_probas,
        oof_probas=oof_probas,
        all_groups_probas=all_groups_probas,
        diagnostics={
            "processed_subsets": int(processed_subsets),
            "processed_models": int(processed_models),
            "eligible_subsets": int(len(subsets)),
            "skipped_incomplete_subsets": int(skipped_incomplete_subsets),
            "skipped_missing_model_subsets": int(skipped_missing_model_subsets),
            "skipped_bad_feature_subsets": int(skipped_bad_feature_subsets),
            "all_model_count": int(all_model_count),
            "do_weighted": bool(do_weighted),
            "include_permuted": bool(include_permuted),
            "permuted_insample_flip_mode": (
                "reconstructed_from_labels" if include_permuted else "not_applicable"
            ),
        },
    )

    if do_weighted:
        # Handle the case where all weights are zero (or missing)
        if all_weight_sum_scalar <= 0:
            if verbose:
                print(
                    "WARNING: subset_weights provided but sum of weights is 0. "
                    "Weighted predictions will be omitted (None)."
                )
            return out

        with np.errstate(divide="ignore", invalid="ignore"):
            oof_margin_mean_w = np.where(
                oof_weight_sum > 0,
                oof_margin_sum_w / oof_weight_sum,
                np.nan,
            )
            insample_margin_mean_w = np.where(
                insample_weight_sum > 0,
                insample_margin_sum_w / insample_weight_sum,
                np.nan,
            )
            all_margin_mean_w = all_margin_sum_w / float(all_weight_sum_scalar)

        p1_oof_w = _sigmoid_stable(oof_margin_mean_w)
        p1_in_w = _sigmoid_stable(insample_margin_mean_w)
        p1_all_w = _sigmoid_stable(all_margin_mean_w)

        out.oof_probas_weighted = np.stack([1.0 - p1_oof_w, p1_oof_w], axis=1).astype(
            np.float64
        )
        out.insample_probas_weighted = np.stack([1.0 - p1_in_w, p1_in_w], axis=1).astype(
            np.float64
        )
        out.all_groups_probas_weighted = np.stack([1.0 - p1_all_w, p1_all_w], axis=1).astype(
            np.float64
        )

        if out.diagnostics is None:
            out.diagnostics = {}
        out.diagnostics.update(
            {
                "weight_sum_scalar": float(all_weight_sum_scalar),
            }
        )

    return out
