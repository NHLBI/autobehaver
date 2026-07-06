"""Optimized subset-level workers for FeatureImp committee training.

This module preserves the committee-design analysis while reducing hot-path
redundancy for the common binary case.

Key differences versus the per-fold worker path:
- one worker task trains all folds for a given feature subset;
- each fold computes predictions/SHAP on the all-groups matrix once, then slices
  train/test rows from that result;
- worker payloads return per-subset aggregated sums/counts instead of separate
  large arrays for every fold; and
- DataFrame cleaning / feature-name lookup is done once per worker process.

The per-fold worker remains in :mod:`committee` for low-level regression tests.
"""

from __future__ import annotations

import gc
import os
import traceback
from typing import Any, Mapping

import numpy as np
import pandas as pd
import shap
import xgboost as xgb

from .shap_compat import make_tree_explainer
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score

_FAST_WORKER_Y_ENCODED_TRAIN = None
_FAST_WORKER_LE = None
_FAST_WORKER_X_TRAIN_NP = None
_FAST_WORKER_X_SHAP_NP = None
_FAST_WORKER_FEATURE_INDEX_TRAIN = None
_FAST_WORKER_FEATURE_INDEX_SHAP = None


def _prepare_feature_matrix_and_index(df_or_array):
    """Return a clean ndarray plus feature->column mapping."""
    if df_or_array is None:
        return None, None
    if isinstance(df_or_array, pd.DataFrame):
        clean = df_or_array
        if clean.columns.duplicated().any():
            clean = clean.loc[:, ~clean.columns.duplicated()]
        if isinstance(clean.columns, pd.MultiIndex):
            clean = clean.copy()
            clean.columns = clean.columns.get_level_values(-1)
        columns = list(clean.columns)
        arr = np.ascontiguousarray(clean.to_numpy(copy=False))
        return arr, {feat: idx for idx, feat in enumerate(columns)}
    arr = np.ascontiguousarray(np.asarray(df_or_array))
    return arr, None


def _init_fast_pool_worker(X_train_full, X_shap_full, y_encoded_train, le):
    """Initializer for optimized subset-level workers."""
    global _FAST_WORKER_Y_ENCODED_TRAIN, _FAST_WORKER_LE
    global _FAST_WORKER_X_TRAIN_NP, _FAST_WORKER_X_SHAP_NP
    global _FAST_WORKER_FEATURE_INDEX_TRAIN, _FAST_WORKER_FEATURE_INDEX_SHAP

    _FAST_WORKER_Y_ENCODED_TRAIN = y_encoded_train
    _FAST_WORKER_LE = le
    _FAST_WORKER_X_TRAIN_NP, _FAST_WORKER_FEATURE_INDEX_TRAIN = _prepare_feature_matrix_and_index(
        X_train_full
    )
    _FAST_WORKER_X_SHAP_NP, _FAST_WORKER_FEATURE_INDEX_SHAP = _prepare_feature_matrix_and_index(
        X_shap_full
    )
    if _FAST_WORKER_X_SHAP_NP is None:
        _FAST_WORKER_X_SHAP_NP = _FAST_WORKER_X_TRAIN_NP
        _FAST_WORKER_FEATURE_INDEX_SHAP = _FAST_WORKER_FEATURE_INDEX_TRAIN


def _build_worker_error(
    *,
    h_hit: int,
    subset_idx: int,
    fold_idx: int | None,
    is_permuted: bool,
    permutation_idx: int | None,
    exc: BaseException,
) -> dict[str, Any]:
    return {
        '__worker_error__': True,
        'h_hit': h_hit,
        'subset_idx': subset_idx,
        'fold_idx': fold_idx,
        'is_permuted': bool(is_permuted),
        'permutation_idx': permutation_idx,
        'error_type': type(exc).__name__,
        'error_message': str(exc),
        'traceback': traceback.format_exc(),
    }


def _save_model_path(*, model, paths, h_hit, subset_idx, fold_idx, is_permuted, permutation_idx):
    if not paths or 'models_folder' not in paths:
        return None
    try:
        base_models_folder = paths['models_folder']
        os.makedirs(base_models_folder, exist_ok=True)
        if is_permuted:
            perm_tag = f"perm{permutation_idx}" if permutation_idx is not None else "perm"
            model_folder = os.path.join(base_models_folder, perm_tag)
            os.makedirs(model_folder, exist_ok=True)
        else:
            model_folder = base_models_folder
        model_path = os.path.join(model_folder, f'h{h_hit}_s{subset_idx}_f{fold_idx}.json')
        model.save_model(model_path)
        return model_path
    except Exception:
        return None


def _positive_class_shap(explainer: shap.TreeExplainer, X: np.ndarray) -> np.ndarray:
    shap_values = explainer.shap_values(X)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    return np.asarray(shap_values, dtype=np.float32)


def _train_subset_worker(task: Mapping[str, Any]) -> dict[str, Any]:
    """Train all folds for one feature subset and return aggregated outputs."""
    global _FAST_WORKER_X_TRAIN_NP, _FAST_WORKER_X_SHAP_NP
    global _FAST_WORKER_Y_ENCODED_TRAIN, _FAST_WORKER_LE
    global _FAST_WORKER_FEATURE_INDEX_TRAIN, _FAST_WORKER_FEATURE_INDEX_SHAP

    h_hit = int(task['h_hit'])
    subset_idx = int(task['subset_idx'])
    selected_features = list(task['selected_features'])
    fold_indices_round = task['fold_indices_round']
    num_classes = int(task['num_classes'])
    model_subsample = float(task['model_subsample'])
    n_estimators = int(task['n_estimators'])
    learning_rate = float(task['learning_rate'])
    max_depth = int(task['max_depth'])
    global_random_seed = task.get('global_random_seed', None)
    paths = task.get('paths', None)
    is_permuted = bool(task.get('is_permuted', False))
    permutation_idx = task.get('permutation_idx', None)
    y_permuted_by_fold = task.get('y_permuted_by_fold', None)
    y_original_full = task.get('y_original_full', None)

    if _FAST_WORKER_X_TRAIN_NP is None or _FAST_WORKER_Y_ENCODED_TRAIN is None:
        raise RuntimeError(
            "Optimized worker missing shared matrices/labels. Did you forget initializer=_init_fast_pool_worker?"
        )

    try:
        selected_idx_train = np.asarray(
            [_FAST_WORKER_FEATURE_INDEX_TRAIN[f] for f in selected_features], dtype=np.int64
        )
        selected_idx_shap = np.asarray(
            [_FAST_WORKER_FEATURE_INDEX_SHAP[f] for f in selected_features], dtype=np.int64
        )
        X_train_selected = np.take(_FAST_WORKER_X_TRAIN_NP, selected_idx_train, axis=1)
        X_all_selected = np.take(_FAST_WORKER_X_SHAP_NP, selected_idx_shap, axis=1)

        n_train = int(len(_FAST_WORKER_Y_ENCODED_TRAIN))
        n_all = int(X_all_selected.shape[0])
        n_subset_features = int(len(selected_features))

        subset_oof_probas_sum = np.zeros((n_train, num_classes), dtype=np.float32)
        subset_insample_probas_sum = np.zeros((n_train, num_classes), dtype=np.float32)
        subset_oof_shap_sum = np.zeros((n_train, n_subset_features), dtype=np.float32)
        subset_insample_shap_sum = np.zeros((n_train, n_subset_features), dtype=np.float32)
        subset_oof_margin_sum = np.zeros(n_train, dtype=np.float64)
        subset_insample_margin_sum = np.zeros(n_train, dtype=np.float64)
        subset_train_pred_count = np.zeros(n_train, dtype=np.int32)
        subset_test_pred_count = np.zeros(n_train, dtype=np.int32)
        all_groups_probas_sum = np.zeros((n_all, num_classes), dtype=np.float32)
        all_groups_shap_sum = np.zeros((n_all, n_subset_features), dtype=np.float32)
        all_groups_margin_sum = np.zeros(n_all, dtype=np.float64)
        all_groups_model_count = 0

        fold_performance: list[dict[str, Any]] = []
        fold_metrics: list[dict[str, Any]] = []
        failed_folds: list[dict[str, Any]] = []

        flip_probabilities = None
        if is_permuted:
            if y_permuted_by_fold is None or y_original_full is None:
                raise ValueError(
                    "Permuted subset worker requires y_permuted_by_fold and y_original_full."
                )
            from .permutation import _flip_probabilities_for_permuted_samples
            flip_probabilities = _flip_probabilities_for_permuted_samples

        for fold_idx, (train_indices, test_indices) in enumerate(fold_indices_round):
            train_indices = np.asarray(train_indices, dtype=np.int64)
            test_indices = np.asarray(test_indices, dtype=np.int64)
            try:
                X_train_fold = X_train_selected[train_indices]
                y_test_fold = _FAST_WORKER_Y_ENCODED_TRAIN[test_indices]
                if is_permuted:
                    y_permuted_full = np.asarray(y_permuted_by_fold[fold_idx])
                    y_train_fold = _FAST_WORKER_LE.transform(y_permuted_full[train_indices])
                else:
                    y_permuted_full = None
                    y_train_fold = _FAST_WORKER_Y_ENCODED_TRAIN[train_indices]

                model = xgb.XGBClassifier(
                    n_estimators=n_estimators,
                    learning_rate=learning_rate,
                    max_depth=max_depth,
                    subsample=model_subsample,
                    random_state=(
                        int(global_random_seed) + h_hit * 10000 + subset_idx * 100 + fold_idx
                        if global_random_seed is not None
                        else None
                    ),
                    objective='binary:logistic',
                    eval_metric='logloss',
                    n_jobs=1,
                )
                model.fit(X_train_fold, y_train_fold)
                explainer = make_tree_explainer(model)
                probas_all = np.asarray(model.predict_proba(X_all_selected), dtype=np.float32)
                shap_all = _positive_class_shap(explainer, X_all_selected)
                margins_all = np.asarray(model.predict(X_all_selected, output_margin=True), dtype=np.float64)

                probas_train = probas_all[train_indices]
                probas_test = probas_all[test_indices]
                shap_train = shap_all[train_indices]
                shap_test = shap_all[test_indices]
                margin_train = margins_all[train_indices]
                margin_test = margins_all[test_indices]
                if flip_probabilities is not None:
                    probas_train_for_agg = flip_probabilities(
                        probas=probas_train,
                        y_original_full=y_original_full,
                        y_permuted_full=y_permuted_full,
                        train_indices=train_indices,
                    ).astype(np.float32)
                    flip_mask_train = np.asarray(
                        np.asarray(y_original_full)[train_indices] != np.asarray(y_permuted_full)[train_indices],
                        dtype=bool,
                    )
                    if np.any(flip_mask_train):
                        margin_train_for_agg = np.asarray(margin_train, dtype=np.float64).copy()
                        margin_train_for_agg[flip_mask_train] *= -1.0
                    else:
                        margin_train_for_agg = margin_train
                else:
                    probas_train_for_agg = probas_train
                    margin_train_for_agg = margin_train

                subset_oof_shap_sum[test_indices, :] += shap_test
                subset_oof_probas_sum[test_indices, :] += probas_test
                subset_oof_margin_sum[test_indices] += margin_test
                subset_test_pred_count[test_indices] += 1
                subset_insample_shap_sum[train_indices, :] += shap_train
                subset_insample_probas_sum[train_indices, :] += probas_train_for_agg
                subset_insample_margin_sum[train_indices] += margin_train_for_agg
                subset_train_pred_count[train_indices] += 1
                all_groups_shap_sum += shap_all
                all_groups_probas_sum += probas_all
                all_groups_margin_sum += margins_all
                all_groups_model_count += 1

                y_pred_train = (probas_train[:, 1] >= 0.5).astype(int)
                y_pred_test = (probas_test[:, 1] >= 0.5).astype(int)
                accuracy_train = float(accuracy_score(y_train_fold, y_pred_train))
                accuracy_test = float(accuracy_score(y_test_fold, y_pred_test))
                p_test = np.asarray(probas_test[:, 1], dtype=np.float64)
                fold_logloss_sum = float(log_loss(y_test_fold, p_test, labels=list(range(num_classes)), normalize=False))
                if len(np.unique(y_test_fold)) == 2:
                    fold_auc = float(roc_auc_score(y_test_fold, p_test))
                    auc_weighted_sum = fold_auc * float(len(test_indices))
                    auc_weighted_n = int(len(test_indices))
                else:
                    auc_weighted_sum = 0.0
                    auc_weighted_n = 0

                model_path = _save_model_path(
                    model=model,
                    paths=paths,
                    h_hit=h_hit,
                    subset_idx=subset_idx,
                    fold_idx=fold_idx,
                    is_permuted=is_permuted,
                    permutation_idx=permutation_idx,
                )
                fold_performance.append(
                    {
                        'h_hit': h_hit,
                        'subset_idx': subset_idx,
                        'fold_idx': fold_idx,
                        'accuracy_test': accuracy_test,
                        'accuracy_train': accuracy_train,
                        'model_path': model_path,
                    }
                )
                fold_metrics.append(
                    {
                        'fold_idx': fold_idx,
                        'oof_logloss_sum': fold_logloss_sum,
                        'n_test_seen_total': int(len(test_indices)),
                        'auc_weighted_sum': auc_weighted_sum,
                        'auc_weighted_n': auc_weighted_n,
                    }
                )
                del model, explainer, probas_all, shap_all, margins_all, probas_train, probas_test, shap_train, shap_test, margin_train, margin_test
            except Exception as exc:
                failed_folds.append(
                    _build_worker_error(
                        h_hit=h_hit,
                        subset_idx=subset_idx,
                        fold_idx=fold_idx,
                        is_permuted=is_permuted,
                        permutation_idx=permutation_idx,
                        exc=exc,
                    )
                )
                continue

        gc.collect()
        return {
            'h_hit': h_hit,
            'subset_idx': subset_idx,
            'features': selected_features,
            'n_features': n_subset_features,
            'oof_probas_sum': subset_oof_probas_sum,
            'oof_shap_sum': subset_oof_shap_sum,
            'oof_margin_sum': subset_oof_margin_sum,
            'insample_probas_sum': subset_insample_probas_sum,
            'insample_shap_sum': subset_insample_shap_sum,
            'insample_margin_sum': subset_insample_margin_sum,
            'train_pred_count': subset_train_pred_count,
            'test_pred_count': subset_test_pred_count,
            'all_groups_probas_sum': all_groups_probas_sum,
            'all_groups_shap_sum': all_groups_shap_sum,
            'all_groups_margin_sum': all_groups_margin_sum,
            'all_groups_model_count': int(all_groups_model_count),
            'fold_performance': fold_performance,
            'fold_metrics': fold_metrics,
            'failed_folds': failed_folds,
            'is_permuted': is_permuted,
            'permutation_idx': permutation_idx,
        }
    except Exception as exc:
        return _build_worker_error(
            h_hit=h_hit,
            subset_idx=subset_idx,
            fold_idx=None,
            is_permuted=is_permuted,
            permutation_idx=permutation_idx,
            exc=exc,
        )
