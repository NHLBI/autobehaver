"""autobehaver.postinf.featureimp.hhit

Utilities for the *h_hit-first* specialist-committee training design.

In the FeatureImp pipeline, each "h_hit" round:
  1) creates a fresh StratifiedKFold split, and
  2) generates feature subsets such that every feature is included at least once
     across the subsets for that round.

This module centralizes the shared h-hit logic used by both the real-model and
permutation specialist committees.

Important: this module is intended to be *pure* (no global settings) and
should not import heavy ML deps (xgboost/shap).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.model_selection import StratifiedKFold


@dataclass(frozen=True)
class HHITSubsetPlan:
    """Result of generating feature subsets for a single h_hit round."""

    feature_subsets: List[List[str]]
    feature_usage_count: Dict[str, int]
    reuse_count: int


def summarize_strata_counts(stratify_labels: np.ndarray) -> Dict[str, int]:
    """Return per-stratum counts preserving first-seen order.

    Parameters
    ----------
    stratify_labels:
        One-dimensional array-like of stratification labels.

    Returns
    -------
    dict
        Mapping of ``str(label) -> count``.
    """

    arr = np.asarray(stratify_labels)
    if arr.ndim != 1:
        raise ValueError(f"stratify_labels must be 1-D, got shape {arr.shape!r}")

    counts: Dict[str, int] = {}
    for label in arr.tolist():
        key = str(label)
        counts[key] = int(counts.get(key, 0) + 1)
    return counts


def validate_stratified_cv_labels(
    stratify_labels: np.ndarray,
    *,
    n_splits: int,
    context: str = "StratifiedKFold",
) -> Dict[str, int]:
    """Validate that every stratum can support the requested fold count.

    FeatureImp uses sex+group composite strata for committee fold generation.
    A too-small stratum does not just make the split fragile; it makes the
    requested ``n_splits`` invalid.  We fail early with a clear message rather
    than letting ``StratifiedKFold`` surface a lower-level exception later.
    """

    if not isinstance(n_splits, int) or n_splits < 2:
        raise ValueError(f"n_splits must be an int >= 2, got {n_splits!r}")

    counts = summarize_strata_counts(stratify_labels)
    if not counts:
        raise ValueError(f"{context}: received no stratification labels.")

    too_small = {label: count for label, count in counts.items() if int(count) < int(n_splits)}
    if too_small:
        counts_preview = ", ".join(f"{label}={count}" for label, count in counts.items())
        too_small_preview = ", ".join(f"{label}={count}" for label, count in too_small.items())
        raise ValueError(
            f"{context}: n_splits={n_splits} requires at least {n_splits} samples in every stratification stratum, "
            f"but the following strata are too small: {too_small_preview}. "
            f"Observed stratum counts: {counts_preview}. "
            "Reduce n_splits or use a dataset/sex filter with enough samples in each sex+group combination."
        )

    return counts


def make_hhit_folds(
    X_train_full,
    stratify_labels: np.ndarray,
    *,
    n_splits: int,
    random_state: Optional[int],
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Create fold indices for a single h_hit round."""

    if not isinstance(n_splits, int) or n_splits < 2:
        raise ValueError(f"n_splits must be an int >= 2, got {n_splits!r}")

    validate_stratified_cv_labels(
        stratify_labels,
        n_splits=n_splits,
        context="FeatureImp h-hit fold generation",
    )

    skf_round = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    out: List[Tuple[np.ndarray, np.ndarray]] = []
    for _, (train_idx, test_idx) in enumerate(skf_round.split(X_train_full, stratify_labels)):
        out.append((train_idx, test_idx))
    return out


def _select_padding_features_min_extra_rep(
    candidates: Sequence[str],
    n_to_select: int,
    *,
    extra_rep_count: Dict[str, int],
) -> List[str]:
    """Select padding features with a *min-count* policy.

    We prefer candidates that have received fewer "extra representations" across
    previous h_hit rounds (tracked in `extra_rep_count`).

    Selection rules:
      - choose from the lowest extra_rep_count first
      - break ties randomly (via numpy RNG already seeded by the caller)
      - sample without replacement

    Notes:
      - `candidates` must not contain duplicates.
      - This function assumes numpy's RNG has already been seeded for
        deterministic behavior.
    """

    if n_to_select <= 0:
        return []

    # Group candidates by their extra-representation count.
    by_count: Dict[int, List[str]] = {}
    for f in candidates:
        c = int(extra_rep_count.get(f, 0))
        by_count.setdefault(c, []).append(f)

    selected: List[str] = []
    remaining = int(n_to_select)

    for c in sorted(by_count.keys()):
        pool = by_count[c]
        if remaining <= 0:
            break

        if len(pool) <= remaining:
            # Take all from this tier, but shuffle order for determinism.
            if len(pool) <= 1:
                chosen = list(pool)
            else:
                chosen = list(np.random.choice(pool, len(pool), replace=False))
            selected.extend(chosen)
            remaining -= len(pool)
        else:
            chosen = list(np.random.choice(pool, remaining, replace=False))
            selected.extend(chosen)
            remaining = 0

    return selected


def generate_feature_subsets_for_hhit(
    all_feature_names: Sequence[str],
    *,
    n_features_per_model: int,
    seed: Optional[int],
    log_every: int = 10,
    total_features: Optional[int] = None,
    context_label: str = "",
    extra_rep_count: Optional[Dict[str, int]] = None,
) -> HHITSubsetPlan:
    """Generate feature subsets for one h_hit round.

    This matches the historical FeatureImp subset-construction behavior:
      - repeatedly sample subsets until every feature is used at least once
      - if the last subset cannot be filled with unused features, pad with already-used
        features (no duplicates within a subset)

    New behavior (optional):
      - if `extra_rep_count` is provided, padding features are chosen using a
        *min-count* policy so that features that have already received extra
        representation across previous h_hit rounds are less likely to be
        chosen again.

    IMPORTANT:
      - This only affects *padding* in the final subset of a round.
      - Callers must update `extra_rep_count` across h_hit rounds to get the
        intended variance reduction.
    """

    feats = list(all_feature_names)
    if total_features is None:
        total_features = len(feats)

    if n_features_per_model <= 0:
        raise ValueError(f"n_features_per_model must be > 0, got {n_features_per_model!r}")

    if seed is not None:
        np.random.seed(seed)

    feature_subsets: List[List[str]] = []
    feature_usage_count: Dict[str, int] = {feat: 0 for feat in feats}

    subset_idx = 0

    while any(count == 0 for count in feature_usage_count.values()):
        unused_features = [f for f, count in feature_usage_count.items() if count == 0]

        if len(unused_features) >= n_features_per_model:
            selected_features = list(
                np.random.choice(unused_features, n_features_per_model, replace=False)
            )
        else:
            selected_features = list(unused_features)
            remaining = n_features_per_model - len(selected_features)

            used_features = [
                f
                for f in feats
                if feature_usage_count[f] > 0 and f not in selected_features
            ]

            if len(used_features) >= remaining:
                if extra_rep_count is None:
                    additional = list(np.random.choice(used_features, remaining, replace=False))
                else:
                    additional = _select_padding_features_min_extra_rep(
                        used_features,
                        remaining,
                        extra_rep_count=extra_rep_count,
                    )
                selected_features.extend(additional)
            else:
                selected_features.extend(used_features)

        # Ensure uniqueness and cap length.
        selected_features = list(dict.fromkeys(selected_features))
        selected_features = selected_features[:n_features_per_model]

        if len(selected_features) != len(set(selected_features)):
            selected_features = list(dict.fromkeys(selected_features))

        for feat in set(selected_features):
            feature_usage_count[feat] += 1

        feature_subsets.append(selected_features)
        subset_idx += 1

        if log_every and subset_idx % log_every == 0:
            used_count = sum(1 for c in feature_usage_count.values() if c > 0)
            suffix = f" {context_label}" if context_label else ""
            print(f"  Subset {subset_idx}: {used_count}/{total_features} features used{suffix}")

    if not all(count >= 1 for count in feature_usage_count.values()):
        missing = [f for f, c in feature_usage_count.items() if c < 1]
        raise RuntimeError(
            f"Feature subset generation failure: {len(missing)} feature(s) never used. "
            f"Example missing (up to 20): {missing[:20]}"
        )

    reuse_count = sum(1 for count in feature_usage_count.values() if count > 1)

    return HHITSubsetPlan(
        feature_subsets=feature_subsets,
        feature_usage_count=feature_usage_count,
        reuse_count=reuse_count,
    )
