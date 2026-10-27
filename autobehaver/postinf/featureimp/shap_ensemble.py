"""Aggregate positive-class SHAP from complete saved specialist subsets using subset
weights.

Normalize each feature by its own contributing weights; incomplete subsets contribute
nothing.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Mapping, Sequence, Tuple

import numpy as np
from .integrity import strict_bool
import pandas as pd
import xgboost as xgb

from .shap_compat import make_tree_explainer
from .logit_ensemble import load_classifier_model


def _coerce_shap_array(shap_values: Any, positive_class_index: int = 1) -> np.ndarray:
    """Normalize SHAP return types.

    For binary classification, SHAP may return either:
      * ndarray of shape (n_samples, n_features)
      * list([neg_class, pos_class])

    We always return an ndarray of shape (n_samples, n_features).
    """

    if isinstance(shap_values, list):
        if len(shap_values) <= positive_class_index:
            raise ValueError(
                f"Unexpected SHAP list length={len(shap_values)}; cannot take index={positive_class_index}"
            )
        shap_values = shap_values[positive_class_index]
    return np.asarray(shap_values)



def _record_features(record):
    features = record.get("specialist_features")
    if features is None:
        features = record.get("features")
    if features is None:
        return []
    if isinstance(features, str):
        return [features]
    return list(features)

def compute_weighted_shap_from_saved_models(
    *,
    performance_records: Sequence[Mapping[str, Any]],
    subset_weights: Mapping[Tuple[int, int], float],
    fold_indices_by_h_hit: Mapping[int, Sequence[Tuple[np.ndarray, np.ndarray]]],
    X_all_groups: pd.DataFrame,
    all_feature_names: Sequence[str],
    n_train: int,
    n_splits: int,
    skip_permuted: bool = False,
    positive_class_index: int = 1,
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """Aggregate weighted SHAP matrices from saved models.

    Parameters
    ----------
    performance_records:
        Fold-level performance dicts returned by the committee runs.
        Must include: h_hit, subset_idx, fold_idx, model_path, and either
        `specialist_features` (preferred) or `features`.
    subset_weights:
        Mapping from (h_hit, subset_idx) -> weight.
    fold_indices_by_h_hit:
        Mapping from h_hit -> list of (train_idx, test_idx) folds.
    X_all_groups:
        DataFrame of *all* individuals across all groups (training first).
    all_feature_names:
        Global ordered feature list (columns of the master matrix).
    n_train:
        Number of training rows (the first ``n_train`` rows of X_all_groups).
    n_splits:
        Number of CV folds.

    Returns
    -------
    oof_shap_weighted:
        (n_train, n_features) weighted out-of-fold SHAP.
    insample_shap_weighted:
        (n_train, n_features) weighted in-sample SHAP (averaged across train-folds).
    all_groups_shap_weighted:
        (n_all, n_features) weighted SHAP for all groups (averaged across folds per subset).
    feature_weight_sums:
        (n_features,) sum of subset weights contributing to each feature.
    diagnostics:
        Small dict with counts and basic summaries.
    """

    if n_splits < 2:
        raise ValueError(f"n_splits must be >= 2, got {n_splits}")

    n_all = int(X_all_groups.shape[0])
    n_features = int(len(all_feature_names))
    if n_features != int(X_all_groups.shape[1]):
        raise ValueError(
            f"X_all_groups has {X_all_groups.shape[1]} columns but all_feature_names has {n_features}"
        )
    if n_train > n_all:
        raise ValueError(f"n_train={n_train} cannot exceed n_all={n_all}")

    feat_to_idx = {str(f): i for i, f in enumerate(all_feature_names)}

    # Group fold records by subset.
    subset_to_entry: Dict[Tuple[int, int], Dict[str, Any]] = {}
    for rec in performance_records:
        if not isinstance(rec, Mapping):
            continue
        try:
            if skip_permuted and strict_bool(rec.get("is_permuted", False)):
                continue

            h_hit = int(rec.get("h_hit"))
            subset_idx = int(rec.get("subset_idx"))
            fold_idx = int(rec.get("fold_idx"))
        except Exception as exc:
            raise ValueError("Invalid saved-model SHAP task identity") from exc

        key = (h_hit, subset_idx)
        entry = subset_to_entry.get(key)
        if entry is None:
            feats = _record_features(rec)
            entry = {"features": list(feats), "model_paths": {}}
            subset_to_entry[key] = entry

        if fold_idx in entry["model_paths"]:
            raise ValueError("Duplicate saved-model SHAP fold identity")
        current_features = _record_features(rec)
        if current_features != entry["features"]:
            raise ValueError("Saved-model SHAP feature order differs across folds")
        entry["model_paths"][fold_idx] = rec.get("model_path")

    # Allocate accumulators.
    oof_sum = np.zeros((n_train, n_features), dtype=np.float32)
    insample_sum = np.zeros((n_train, n_features), dtype=np.float32)
    all_sum = np.zeros((n_all, n_features), dtype=np.float32)
    feature_weight_sums = np.zeros((n_features,), dtype=np.float32)

    # Diagnostics
    n_subsets_total = len(subset_to_entry)
    n_subsets_used = 0
    n_subsets_skipped_zero_w = 0
    n_subsets_skipped_incomplete = 0
    n_models_loaded = 0
    total_weight_sum = 0.0

    # Local import to keep module import light outside FeatureImp.
    import shap  # type: ignore

    for (h_hit, subset_idx), entry in subset_to_entry.items():
        w = float(subset_weights.get((h_hit, subset_idx), 0.0))
        if not np.isfinite(w) or w < 0:
            raise ValueError("Subset weights must be nonnegative finite")
        if w == 0.0:
            n_subsets_skipped_zero_w += 1
            continue

        features = list(entry.get("features") or [])
        if not features:
            raise ValueError(f"Positive-weight subset {(h_hit, subset_idx)} has no features")

        # Validate folds
        folds = fold_indices_by_h_hit.get(h_hit)
        if folds is None or len(folds) != n_splits:
            raise ValueError(f"Positive-weight subset {(h_hit, subset_idx)} lacks its complete fold plan")

        # Validate model paths for ALL folds upfront.
        model_paths = dict(entry.get("model_paths") or {})
        missing_model = False
        for fold_idx in range(n_splits):
            mp = model_paths.get(fold_idx)
            if (not mp) or (not isinstance(mp, str)) or (not os.path.exists(mp)):
                missing_model = True
                break
        if missing_model:
            raise ValueError(f"Positive-weight subset {(h_hit, subset_idx)} lacks a required saved model")

        # Map subset features to global indices.
        try:
            feat_idxs = np.fromiter((feat_to_idx[f] for f in features), dtype=np.int64)
        except Exception as exc:
            raise ValueError(f"Positive-weight subset {(h_hit, subset_idx)} references an absent feature") from exc
        if len(set(feat_idxs.tolist())) != len(feat_idxs):
            raise ValueError("Subset contains duplicate feature identities")

        # Extract subset feature matrix once.
        X_sub = X_all_groups[features]

        # Local (per-subset) accumulators so incomplete subsets contribute nothing.
        local_oof = np.zeros((n_train, len(feat_idxs)), dtype=np.float32)
        local_ins = np.zeros((n_train, len(feat_idxs)), dtype=np.float32)
        local_all = np.zeros((n_all, len(feat_idxs)), dtype=np.float32)

        per_fold_weight = np.float32(w / float(n_splits))

        subset_ok = True
        try:
            for fold_idx in range(n_splits):
                model_path = model_paths.get(fold_idx)
                # model_path existence validated above

                model = load_classifier_model(str(model_path))
                n_models_loaded += 1

                explainer = make_tree_explainer(model)
                shap_vals = _coerce_shap_array(
                    explainer.shap_values(X_sub),
                    positive_class_index=positive_class_index,
                ).astype(np.float32, copy=False)

                # SHAP must align with X_sub
                if shap_vals.shape[0] != n_all or shap_vals.shape[1] != len(feat_idxs):
                    raise ValueError(
                        f"Unexpected SHAP shape {shap_vals.shape}; expected ({n_all}, {len(feat_idxs)})"
                    )
                if not np.isfinite(shap_vals).all():
                    raise ValueError("Required saved-model SHAP values are nonfinite")

                train_idx, test_idx = folds[fold_idx]

                # Weighted all-groups SHAP (average over folds per subset).
                local_all += per_fold_weight * shap_vals

                # Weighted OOF/insample SHAP for training rows.
                shap_train = shap_vals[:n_train, :]
                local_oof[test_idx, :] += np.float32(w) * shap_train[test_idx, :]
                local_ins[train_idx, :] += np.float32(w) * shap_train[train_idx, :]
        except Exception as exc:
            raise RuntimeError(f'Failed to read/evaluate required positive-weight subset {(h_hit, subset_idx)}') from exc

        if not subset_ok:
            n_subsets_skipped_incomplete += 1
            continue

        # Commit this subset's contribution.
        oof_sum[:, feat_idxs] += local_oof
        insample_sum[:, feat_idxs] += local_ins
        all_sum[:, feat_idxs] += local_all

        feature_weight_sums[feat_idxs] += np.float32(w)
        total_weight_sum += float(w)
        n_subsets_used += 1

        if verbose and (n_subsets_used % 25 == 0):
            print(
                f"  [weighted_shap] processed {n_subsets_used}/{n_subsets_total} weighted subsets "
                f"(skipped_zero_w={n_subsets_skipped_zero_w}, skipped_incomplete={n_subsets_skipped_incomplete})"
            )

    # Normalize.
    if not np.isfinite(total_weight_sum) or total_weight_sum <= 0:
        raise ValueError("No positive-weight complete SHAP committee is available")
    denom = feature_weight_sums.astype(np.float32, copy=False)
    denom_safe = np.where(denom > 0, denom, 1.0).astype(np.float32)

    oof_avg = oof_sum / denom_safe[None, :]
    insample_avg = insample_sum / (denom_safe * np.float32(n_splits - 1))[None, :]
    all_avg = all_sum / denom_safe[None, :]

    # Zero out columns with denom=0.
    zero_cols = denom <= 0
    if np.any(zero_cols):
        oof_avg[:, zero_cols] = 0.0
        insample_avg[:, zero_cols] = 0.0
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
        "n_models_loaded": int(n_models_loaded),
        "total_weight_sum": float(total_weight_sum),
        "n_features_with_zero_weight": int(np.sum(feature_weight_sums <= 0)),
    }

    return oof_avg, insample_avg, all_avg, feature_weight_sums, diagnostics
