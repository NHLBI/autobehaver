"""Shuffle training-fold labels within sex strata and restore predictions to the
original binary label frame.

Test-fold labels remain unchanged.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd


def _generate_sex_class_stratified_permutation(
    y_original: Sequence[str] | np.ndarray,
    df_with_sex: pd.DataFrame,
    train_indices: Sequence[int] | np.ndarray,
    test_indices: Sequence[int] | np.ndarray,
    permutation_seed: int | None = None,
) -> np.ndarray:
    """Generate a sex-stratified permutation of labels for permutation testing.

    The permutation is intentionally limited to train-fold labels:

    - **Only TRAIN fold labels are permuted** (within sex strata).
    - TEST fold labels are left unchanged.

    Downstream, the pipeline flips predicted probabilities for permuted training
    samples so that probabilities remain interpretable relative to the *original*
    A/B label coding.

    Args:
        y_original:
            Full label vector (length N), typically strings like group names.
        df_with_sex:
            DataFrame aligned to y_original with a required 'sex' column.
        train_indices:
            Indices into y_original that form the training fold.
        test_indices:
            Indices into y_original that form the test fold. (Kept for API
            clarity; test labels are not modified.)
        permutation_seed:
            Optional seed for deterministic shuffling.

    Returns:
        y_permuted_full:
            A full-length numpy array where only training-fold labels have been
            shuffled within sex strata.
    """

    y_original = np.asarray(y_original)
    train_indices = np.asarray(train_indices)
    _ = np.asarray(test_indices)  # documented but unused

    if 'sex' not in df_with_sex.columns:
        raise KeyError("df_with_sex must contain a 'sex' column.")

    rng = np.random.default_rng(permutation_seed)

    # Start from true labels for everyone; then overwrite train fold labels.
    y_permuted_full = y_original.copy()

    if len(train_indices) <= 1:
        return y_permuted_full

    train_sex = df_with_sex['sex'].iloc[train_indices].to_numpy()
    train_labels = y_original[train_indices].copy()

    # Shuffle labels within each sex stratum.
    for sex in np.unique(train_sex):
        idx = np.where(train_sex == sex)[0]
        if len(idx) <= 1:
            continue
        train_labels[idx] = rng.permutation(train_labels[idx])

    y_permuted_full[train_indices] = train_labels
    return y_permuted_full

def _flip_probabilities_for_permuted_samples(
    probas: np.ndarray,
    y_original_full: Sequence[str] | np.ndarray,
    y_permuted_full: Sequence[str] | np.ndarray,
    train_indices: Sequence[int] | np.ndarray,
) -> np.ndarray:
    """Flip probabilities for permuted-label TRAIN samples.

    In this pipeline, permutation testing permutes labels **only within the
    training fold** (test-fold labels remain true). When we aggregate
    probabilities across models, we keep probabilities interpretable relative to
    the *original* A/B label coding by flipping probabilities for the training
    samples whose labels were permuted.

    For binary problems, flipping the positive-class probability is:
        p -> 1 - p

    Args:
        probas:
            Predicted probabilities for the TRAIN fold only.
            - If 1D: shape (n_train,), interpreted as P(positive_class).
            - If 2D: shape (n_train, 2), interpreted as [P(class0), P(class1)].
              In this case we swap the two columns.
        y_original_full:
            Full label vector (length N) before permutation.
        y_permuted_full:
            Full label vector (length N) after permutation.
        train_indices:
            Indices (into the full label vectors) for the training fold.

    Returns:
        adjusted_probas: Same shape as `probas`.
    """

    probas = np.asarray(probas)
    train_indices = np.asarray(train_indices)
    y_original_full = np.asarray(y_original_full)
    y_permuted_full = np.asarray(y_permuted_full)

    if probas.shape[0] != len(train_indices):
        raise ValueError(
            "Expected probas to be aligned to train_indices (len mismatch: "
            f"probas={probas.shape[0]} vs train_indices={len(train_indices)})."
        )

    # Which TRAIN samples had permuted labels?
    mask_train = y_original_full[train_indices] != y_permuted_full[train_indices]

    adjusted = probas.copy()
    if adjusted.ndim == 1:
        adjusted = adjusted.astype(float, copy=False)
        adjusted[mask_train] = 1.0 - adjusted[mask_train]
    elif adjusted.ndim == 2 and adjusted.shape[1] == 2:
        adjusted[mask_train] = adjusted[mask_train][:, ::-1]
    else:
        raise ValueError(
            "Expected probas to be either 1D (n_train,) or 2D (n_train, 2); "
            f"got shape={adjusted.shape}."
        )

    return adjusted
