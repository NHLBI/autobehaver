from __future__ import annotations

import numpy as np
import pandas as pd

from autobehaver.postinf.featureimp import committee, committee_fast
from autobehaver.postinf.featureimp.hhit import make_hhit_folds
from autobehaver.postinf.featureimp.permutation import (
    _flip_probabilities_for_permuted_samples,
    _generate_sex_class_stratified_permutation,
)


def _make_fixture_data():
    rng = np.random.RandomState(0)
    n_train = 12
    y_groups = np.array(["A", "B"] * (n_train // 2), dtype=object)
    y_encoded = np.array([0, 1] * (n_train // 2), dtype=int)
    sex = np.array(["M", "F"] * (n_train // 2), dtype=object)

    X_train = pd.DataFrame(
        {
            "f0": rng.normal(loc=y_encoded * 0.6, scale=0.4, size=n_train),
            "f1": rng.normal(loc=(1 - y_encoded) * 0.5, scale=0.5, size=n_train),
            "f2": rng.normal(loc=np.arange(n_train) / 10.0, scale=0.2, size=n_train),
            "f3": rng.normal(loc=(np.arange(n_train) % 3) / 5.0, scale=0.3, size=n_train),
        }
    )

    X_other = pd.DataFrame(
        {
            "f0": rng.normal(loc=0.2, scale=0.4, size=3),
            "f1": rng.normal(loc=0.1, scale=0.5, size=3),
            "f2": rng.normal(loc=0.0, scale=0.2, size=3),
            "f3": rng.normal(loc=0.3, scale=0.3, size=3),
        }
    )
    X_all = pd.concat([X_train, X_other], ignore_index=True)

    le = committee.FixedBinaryLabelEncoder("A", "B")
    strat = y_encoded.copy()
    folds = make_hhit_folds(X_train, strat, n_splits=3, random_state=123)

    df_with_sex = pd.DataFrame({"group": y_groups, "sex": sex})
    return X_train, X_all, y_groups, y_encoded, le, folds, df_with_sex



def _reference_normal_aggregate(X_train, X_all, y_encoded, le, folds, selected_features):
    committee._init_pool_worker(X_train, X_all, y_encoded, le)

    n_train = len(y_encoded)
    n_all = len(X_all)
    n_feats = len(selected_features)
    out = {
        "oof_probas_sum": np.zeros((n_train, 2), dtype=np.float32),
        "oof_shap_sum": np.zeros((n_train, n_feats), dtype=np.float32),
        "insample_probas_sum": np.zeros((n_train, 2), dtype=np.float32),
        "insample_shap_sum": np.zeros((n_train, n_feats), dtype=np.float32),
        "train_pred_count": np.zeros(n_train, dtype=np.int32),
        "test_pred_count": np.zeros(n_train, dtype=np.int32),
        "all_groups_probas_sum": np.zeros((n_all, 2), dtype=np.float32),
        "all_groups_shap_sum": np.zeros((n_all, n_feats), dtype=np.float32),
        "all_groups_model_count": 0,
        "fold_performance": [],
        "fold_metrics": [],
    }

    from sklearn.metrics import log_loss, roc_auc_score

    for fold_idx, (train_idx, test_idx) in enumerate(folds):
        args = (
            1,
            0,
            fold_idx,
            selected_features,
            train_idx,
            test_idx,
            None,
            None,
            None,
            None,
            None,
            None,
            2,
            None,
            1.0,
            5,
            0.1,
            3,
            {},
            False,
            None,
            None,
            None,
        )
        result = committee._train_subset_fold_worker(args)
        assert not result.get("__worker_error__")

        out["oof_probas_sum"][test_idx] += result["probas_test"]
        out["oof_shap_sum"][test_idx] += result["shap_test"]
        out["test_pred_count"][test_idx] += 1
        out["insample_probas_sum"][train_idx] += result["probas_train"]
        out["insample_shap_sum"][train_idx] += result["shap_train"]
        out["train_pred_count"][train_idx] += 1
        out["all_groups_probas_sum"] += result["probas_all_groups"]
        out["all_groups_shap_sum"] += result["shap_all_groups"]
        out["all_groups_model_count"] += 1

        p_test = np.asarray(result["probas_test"], dtype=np.float64)[:, 1]
        y_test = y_encoded[test_idx]
        if len(np.unique(y_test)) == 2:
            auc_weighted_sum = float(roc_auc_score(y_test, p_test)) * len(test_idx)
            auc_weighted_n = int(len(test_idx))
        else:
            auc_weighted_sum = 0.0
            auc_weighted_n = 0
        out["fold_metrics"].append(
            {
                "fold_idx": fold_idx,
                "oof_logloss_sum": float(log_loss(y_test, p_test, labels=[0, 1], normalize=False)),
                "n_test_seen_total": int(len(test_idx)),
                "auc_weighted_sum": auc_weighted_sum,
                "auc_weighted_n": auc_weighted_n,
            }
        )
        out["fold_performance"].append(
            {
                "h_hit": 1,
                "subset_idx": 0,
                "fold_idx": fold_idx,
                "accuracy_test": result["accuracy_test"],
                "accuracy_train": result["accuracy_train"],
                "model_path": result.get("model_path"),
            }
        )

    return out



def _reference_permuted_aggregate(X_train, X_all, y_groups, y_encoded, le, folds, df_with_sex, selected_features):
    committee._init_pool_worker(X_train, X_all, y_encoded, le)

    n_train = len(y_encoded)
    n_all = len(X_all)
    n_feats = len(selected_features)
    out = {
        "oof_probas_sum": np.zeros((n_train, 2), dtype=np.float32),
        "oof_shap_sum": np.zeros((n_train, n_feats), dtype=np.float32),
        "insample_probas_sum": np.zeros((n_train, 2), dtype=np.float32),
        "insample_shap_sum": np.zeros((n_train, n_feats), dtype=np.float32),
        "train_pred_count": np.zeros(n_train, dtype=np.int32),
        "test_pred_count": np.zeros(n_train, dtype=np.int32),
        "all_groups_probas_sum": np.zeros((n_all, 2), dtype=np.float32),
        "all_groups_shap_sum": np.zeros((n_all, n_feats), dtype=np.float32),
        "all_groups_model_count": 0,
        "fold_performance": [],
        "fold_metrics": [],
        "y_permuted_by_fold": [],
    }

    from sklearn.metrics import log_loss, roc_auc_score

    for fold_idx, (train_idx, test_idx) in enumerate(folds):
        y_perm_full = _generate_sex_class_stratified_permutation(
            y_original=y_groups,
            df_with_sex=df_with_sex,
            train_indices=train_idx,
            test_indices=test_idx,
            permutation_seed=555 + fold_idx,
        )
        out["y_permuted_by_fold"].append(y_perm_full)

        args = (
            1,
            0,
            fold_idx,
            selected_features,
            train_idx,
            test_idx,
            None,
            None,
            None,
            None,
            None,
            None,
            2,
            None,
            1.0,
            5,
            0.1,
            3,
            {},
            True,
            0,
            y_perm_full,
            None,
        )
        result = committee._train_subset_fold_worker(args)
        assert not result.get("__worker_error__")

        probas_train_flipped = _flip_probabilities_for_permuted_samples(
            probas=result["probas_train"],
            y_original_full=y_groups,
            y_permuted_full=y_perm_full,
            train_indices=train_idx,
        ).astype(np.float32)

        out["oof_probas_sum"][test_idx] += result["probas_test"]
        out["oof_shap_sum"][test_idx] += result["shap_test"]
        out["test_pred_count"][test_idx] += 1
        out["insample_probas_sum"][train_idx] += probas_train_flipped
        out["insample_shap_sum"][train_idx] += result["shap_train"]
        out["train_pred_count"][train_idx] += 1
        out["all_groups_probas_sum"] += result["probas_all_groups"]
        out["all_groups_shap_sum"] += result["shap_all_groups"]
        out["all_groups_model_count"] += 1

        p_test = np.asarray(result["probas_test"], dtype=np.float64)[:, 1]
        y_test = y_encoded[test_idx]
        if len(np.unique(y_test)) == 2:
            auc_weighted_sum = float(roc_auc_score(y_test, p_test)) * len(test_idx)
            auc_weighted_n = int(len(test_idx))
        else:
            auc_weighted_sum = 0.0
            auc_weighted_n = 0
        out["fold_metrics"].append(
            {
                "fold_idx": fold_idx,
                "oof_logloss_sum": float(log_loss(y_test, p_test, labels=[0, 1], normalize=False)),
                "n_test_seen_total": int(len(test_idx)),
                "auc_weighted_sum": auc_weighted_sum,
                "auc_weighted_n": auc_weighted_n,
            }
        )
        out["fold_performance"].append(
            {
                "h_hit": 1,
                "subset_idx": 0,
                "fold_idx": fold_idx,
                "accuracy_test": result["accuracy_test"],
                "accuracy_train": result["accuracy_train"],
                "model_path": result.get("model_path"),
            }
        )

    out["y_permuted_by_fold"] = tuple(out["y_permuted_by_fold"])
    return out



def _assert_results_match(reference, optimized):
    for key in (
        "oof_probas_sum",
        "oof_shap_sum",
        "insample_probas_sum",
        "insample_shap_sum",
        "train_pred_count",
        "test_pred_count",
        "all_groups_probas_sum",
        "all_groups_shap_sum",
    ):
        assert np.allclose(reference[key], optimized[key], rtol=1e-5, atol=1e-6), key

    assert int(reference["all_groups_model_count"]) == int(optimized["all_groups_model_count"])

    reference_perf = sorted(reference["fold_performance"], key=lambda x: x["fold_idx"])
    opt_perf = sorted(optimized["fold_performance"], key=lambda x: x["fold_idx"])
    assert [x["fold_idx"] for x in reference_perf] == [x["fold_idx"] for x in opt_perf]
    assert np.allclose([x["accuracy_test"] for x in reference_perf], [x["accuracy_test"] for x in opt_perf], rtol=1e-6, atol=1e-7)
    assert np.allclose([x["accuracy_train"] for x in reference_perf], [x["accuracy_train"] for x in opt_perf], rtol=1e-6, atol=1e-7)

    reference_metrics = sorted(reference["fold_metrics"], key=lambda x: x["fold_idx"])
    opt_metrics = sorted(optimized["fold_metrics"], key=lambda x: x["fold_idx"])
    assert [x["fold_idx"] for x in reference_metrics] == [x["fold_idx"] for x in opt_metrics]
    for lk, ok in zip(reference_metrics, opt_metrics):
        assert np.isclose(lk["oof_logloss_sum"], ok["oof_logloss_sum"], rtol=1e-6, atol=1e-7)
        assert int(lk["n_test_seen_total"]) == int(ok["n_test_seen_total"])
        assert np.isclose(lk["auc_weighted_sum"], ok["auc_weighted_sum"], rtol=1e-6, atol=1e-7)
        assert int(lk["auc_weighted_n"]) == int(ok["auc_weighted_n"])



def test_subset_worker_matches_reference_non_permuted():
    X_train, X_all, _y_groups, y_encoded, le, folds, _df_with_sex = _make_fixture_data()
    selected_features = ["f0", "f2"]
    reference = _reference_normal_aggregate(X_train, X_all, y_encoded, le, folds, selected_features)

    committee_fast._init_fast_pool_worker(X_train, X_all, y_encoded, le)
    task = {
        "h_hit": 1,
        "subset_idx": 0,
        "selected_features": tuple(selected_features),
        "fold_indices_round": tuple(folds),
        "num_classes": 2,
        "model_subsample": 1.0,
        "n_estimators": 5,
        "learning_rate": 0.1,
        "max_depth": 3,
        "global_random_seed": 7,
        "paths": {},
        "is_permuted": False,
        "permutation_idx": None,
    }
    optimized = committee_fast._train_subset_worker(task)
    assert not optimized.get("__worker_error__")
    assert optimized.get("failed_folds") == []
    _assert_results_match(reference, optimized)



def test_subset_worker_matches_reference_permuted():
    X_train, X_all, y_groups, y_encoded, le, folds, df_with_sex = _make_fixture_data()
    selected_features = ["f1", "f3"]
    reference = _reference_permuted_aggregate(X_train, X_all, y_groups, y_encoded, le, folds, df_with_sex, selected_features)

    committee_fast._init_fast_pool_worker(X_train, X_all, y_encoded, le)
    task = {
        "h_hit": 1,
        "subset_idx": 0,
        "selected_features": tuple(selected_features),
        "fold_indices_round": tuple(folds),
        "num_classes": 2,
        "model_subsample": 1.0,
        "n_estimators": 5,
        "learning_rate": 0.1,
        "max_depth": 3,
        "global_random_seed": 7,
        "paths": {},
        "is_permuted": True,
        "permutation_idx": 0,
        "y_permuted_by_fold": reference["y_permuted_by_fold"],
        "y_original_full": y_groups,
    }
    optimized = committee_fast._train_subset_worker(task)
    assert not optimized.get("__worker_error__")
    assert optimized.get("failed_folds") == []
    _assert_results_match(reference, optimized)
