import numpy as np
import pandas as pd

from autobehaver.postinf.featureimp.hhit import make_hhit_folds


def _fold_signature(folds):
    """Hashable signature of folds to compare determinism."""
    return tuple(
        (
            tuple(map(int, train_idx.tolist())),
            tuple(map(int, test_idx.tolist())),
        )
        for train_idx, test_idx in folds
    )


def test_make_hhit_folds_is_deterministic_for_fixed_seed():
    rng = np.random.RandomState(0)
    X = pd.DataFrame(rng.randn(50, 3), columns=["a", "b", "c"])

    # Two strata (balanced)
    strat = np.array([0, 1] * 25, dtype=int)

    folds1 = make_hhit_folds(X, strat, n_splits=5, random_state=123)
    folds2 = make_hhit_folds(X, strat, n_splits=5, random_state=123)

    assert _fold_signature(folds1) == _fold_signature(folds2)


def test_make_hhit_folds_changes_with_seed_most_of_the_time():
    rng = np.random.RandomState(1)
    X = pd.DataFrame(rng.randn(60, 2), columns=["x", "y"])
    strat = np.array([0, 1] * 30, dtype=int)

    folds_a = make_hhit_folds(X, strat, n_splits=6, random_state=111)
    folds_b = make_hhit_folds(X, strat, n_splits=6, random_state=222)

    # It is theoretically possible for two seeds to produce the same split,
    # but with n=60 and shuffle=True this should be extremely unlikely.
    assert _fold_signature(folds_a) != _fold_signature(folds_b)
