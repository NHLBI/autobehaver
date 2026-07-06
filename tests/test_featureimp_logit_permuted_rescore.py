from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from autobehaver.postinf.featureimp.hhit import make_hhit_folds
from autobehaver.postinf.featureimp.permutation import (
    _generate_sex_class_stratified_permutation,
)


def _sigmoid(x):
    x = np.asarray(x, dtype=np.float64)
    return 1.0 / (1.0 + np.exp(-x))


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("dummy", encoding="utf-8")


class DummyFoldMarginModel:
    margins = {0: 1.25, 1: -0.75, 2: 0.5}

    def __init__(self):
        self._margin = 0.0

    def load_model(self, path):
        base = Path(str(path)).stem
        m = re.search(r"_f(\d+)$", base)
        if m is None:
            raise ValueError(f"Could not parse fold index from model path: {path}")
        fold_idx = int(m.group(1))
        self._margin = float(self.margins[fold_idx])

    def predict(self, X, output_margin=False):
        arr = np.full((len(X),), self._margin, dtype=np.float64)
        if output_margin:
            return arr
        return (arr >= 0).astype(int)


def test_permuted_logit_rescore_reconstructs_insample_train_flips(tmp_path: Path, monkeypatch) -> None:
    from autobehaver.postinf.featureimp import logit_ensemble as le

    GLOBAL_RANDOM_SEED = 17
    permutation_idx = 0
    h_hit = 1
    n_splits = 3

    y_groups = np.array(
        ["A", "B", "A", "B", "A", "B", "A", "B", "A", "B", "A", "B"],
        dtype=object,
    )
    sex = np.array(
        ["M", "M", "F", "F", "M", "M", "F", "F", "M", "M", "F", "F"],
        dtype=object,
    )
    y_encoded = np.array([0 if g == "A" else 1 for g in y_groups], dtype=int)
    sex_group = np.array([f"{s}_{g}" for s, g in zip(sex, y_groups)], dtype=object)
    sex_group_encoded, _ = pd.factorize(sex_group)

    X_train = pd.DataFrame(
        {
            "f0": np.linspace(-1.0, 1.0, len(y_groups)),
            "f1": np.linspace(0.5, -0.5, len(y_groups)),
        }
    )
    X_all = pd.concat(
        [
            X_train,
            pd.DataFrame(
                {
                    "f0": [0.1, -0.2, 0.3],
                    "f1": [-0.1, 0.2, -0.3],
                }
            ),
        ],
        ignore_index=True,
    )

    model_dir = tmp_path / "MODELS"
    records = []
    for fold_idx in range(n_splits):
        model_path = model_dir / f"h{h_hit}_s0_f{fold_idx}.json"
        _touch(model_path)
        records.append(
            {
                "h_hit": h_hit,
                "subset_idx": 0,
                "fold_idx": fold_idx,
                "model_path": str(model_path),
                "specialist_features": ["f0", "f1"],
                "is_permuted": True,
                "permutation_idx": permutation_idx,
            }
        )

    monkeypatch.setattr(le.xgb, "XGBClassifier", DummyFoldMarginModel)

    out = le.compute_logit_ensemble_predictions_from_saved_models(
        performance_records=records,
        X_train_full=X_train,
        X_all_groups=X_all,
        y_encoded_train=y_encoded,
        sex_group_encoded=sex_group_encoded,
        num_classes=2,
        n_splits=n_splits,
        GLOBAL_RANDOM_SEED=GLOBAL_RANDOM_SEED,
        include_permuted=True,
        permutation_idx=permutation_idx,
        y_original_labels_train=y_groups,
        sex_labels_train=sex,
        verbose=False,
    )

    split_seed = GLOBAL_RANDOM_SEED + permutation_idx * 100000 + h_hit * 1000
    folds = make_hhit_folds(
        X_train,
        sex_group_encoded,
        n_splits=n_splits,
        random_state=split_seed,
    )

    insample_margin_sum = np.zeros(len(X_train), dtype=np.float64)
    insample_margin_count = np.zeros(len(X_train), dtype=np.int32)
    unadjusted_margin_sum = np.zeros(len(X_train), dtype=np.float64)
    oof_margin = np.zeros(len(X_train), dtype=np.float64)
    saw_any_flip = False

    for fold_idx, (train_idx, test_idx) in enumerate(folds):
        margin = float(DummyFoldMarginModel.margins[fold_idx])
        oof_margin[test_idx] = margin
        unadjusted_margin_sum[train_idx] += margin
        insample_margin_count[train_idx] += 1

        perm_seed = GLOBAL_RANDOM_SEED + permutation_idx * 100000 + h_hit * 1000 + fold_idx
        y_perm_full = _generate_sex_class_stratified_permutation(
            y_original=y_groups,
            df_with_sex=pd.DataFrame({"sex": sex}),
            train_indices=train_idx,
            test_indices=test_idx,
            permutation_seed=perm_seed,
        )
        flip_mask = np.asarray(y_groups != y_perm_full, dtype=bool)
        saw_any_flip = saw_any_flip or bool(np.any(flip_mask[train_idx]))
        insample_margin_sum[train_idx] += np.where(flip_mask[train_idx], -margin, margin)

    assert saw_any_flip, "Fixture should contain at least one permuted training-sample label swap."

    insample_margin_mean = insample_margin_sum / insample_margin_count
    unadjusted_margin_mean = unadjusted_margin_sum / insample_margin_count
    expected_oof = _sigmoid(oof_margin)
    expected_insample = _sigmoid(insample_margin_mean)
    expected_insample_unadjusted = _sigmoid(unadjusted_margin_mean)
    expected_all_groups = _sigmoid(np.mean(list(DummyFoldMarginModel.margins.values())))

    assert np.allclose(out.oof_probas[:, 1], expected_oof)
    assert np.allclose(out.insample_probas[:, 1], expected_insample)
    assert np.allclose(out.all_groups_probas[:, 1], expected_all_groups)
    assert not np.allclose(out.insample_probas[:, 1], expected_insample_unadjusted)
    assert out.diagnostics is not None
    assert out.diagnostics.get("permuted_insample_flip_mode") == "reconstructed_from_labels"
