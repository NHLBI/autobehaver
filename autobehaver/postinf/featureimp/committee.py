"""Train individual specialist folds and extract predictions and positive-class SHAP
values for FeatureImp.
"""

from __future__ import annotations

import gc
import multiprocessing
import os
import traceback
from typing import Any

import numpy as np
import pandas as pd
import shap
import xgboost as xgb

from .shap_compat import make_tree_explainer
from sklearn.metrics import accuracy_score

from .config import DEFAULT_SETTINGS
from .prep import filter_data_by_sex


# Binary label encoder + fixed index constants


class FixedBinaryLabelEncoder:
    """Encode group A as class 0 and group B as class 1.

    This fixed order defines probability columns and the positive-class SHAP direction.
    """

    def __init__(self, class0, class1):
        self.classes_ = np.array([class0, class1], dtype=object)
        self._class_to_int = {class0: 0, class1: 1}

    def transform(self, y):
        """Encode labels into {0,1} with a fixed mapping."""
        y_arr = np.asarray(y, dtype=object)
        try:
            return np.vectorize(self._class_to_int.__getitem__)(y_arr).astype(int)
        except KeyError as e:
            raise ValueError(
                f"Unknown class label {e}. Expected only: {list(self._class_to_int.keys())}"
            )

    def inverse_transform(self, y_int):
        """Decode {0,1} back into string labels."""
        y_int = np.asarray(y_int, dtype=int)
        if np.any((y_int < 0) | (y_int >= len(self.classes_))):
            bad = np.unique(y_int[(y_int < 0) | (y_int >= len(self.classes_))])
            raise ValueError(f"Encoded labels out of range: {bad}")
        return self.classes_[y_int]


# By design in this script:
#   class 0 -> training_group_a
#   class 1 -> training_group_b
NEGATIVE_CLASS_INDEX = 0
POSITIVE_CLASS_INDEX = 1


# XGBoost committee training
# Multiprocessing shared state
# These are populated once per worker process via the Pool initializer
_WORKER_X_TRAIN_FULL = None
_WORKER_X_SHAP_FULL = None
_WORKER_Y_ENCODED_TRAIN = None
_WORKER_LE = None

def _init_pool_worker(X_train_full, X_shap_full, y_encoded_train, le):
    """Cache read-only matrices and label metadata once per worker process."""
    global _WORKER_X_TRAIN_FULL, _WORKER_X_SHAP_FULL, _WORKER_Y_ENCODED_TRAIN, _WORKER_LE
    _WORKER_X_TRAIN_FULL = X_train_full
    _WORKER_X_SHAP_FULL = X_shap_full
    _WORKER_Y_ENCODED_TRAIN = y_encoded_train
    _WORKER_LE = le


def _train_subset_fold_worker(args):
    """Train one specialist fold and return its predictions, SHAP arrays, and task
    metadata.

    The task tuple identifies the coverage round, subset, fold, feature order, and
    optional permutation. Failures return a structured worker-error dictionary.
    """
    if len(args) not in {23, 24}:
        raise ValueError(
            f"Unexpected number of arguments: {len(args)}. Expected 23 or 24."
        )

    if len(args) == 23:
        (h_hit, subset_idx, fold_idx, selected_features,
         train_indices, test_indices,
         X_train_full, X_shap_full, y_encoded_train,
         df_train_fold, df_test_fold, df_for_shap_and_agg,
         num_classes, le, model_subsample, n_estimators,
         learning_rate, max_depth, paths,
         is_permuted, permutation_idx, y_permuted_train, df_for_training) = args
        global_random_seed = DEFAULT_SETTINGS.get("GLOBAL_RANDOM_SEED", 7)
    else:
        (h_hit, subset_idx, fold_idx, selected_features,
         train_indices, test_indices,
         X_train_full, X_shap_full, y_encoded_train,
         df_train_fold, df_test_fold, df_for_shap_and_agg,
         num_classes, le, model_subsample, n_estimators,
         learning_rate, max_depth, paths,
         is_permuted, permutation_idx, y_permuted_train, df_for_training,
         global_random_seed) = args

    # Reuse the read-only matrices initialized in each worker.
    # Allow these to be None and pull from Pool initializer globals instead.
    global _WORKER_X_TRAIN_FULL, _WORKER_X_SHAP_FULL, _WORKER_Y_ENCODED_TRAIN, _WORKER_LE

    if X_train_full is None:
        X_train_full = _WORKER_X_TRAIN_FULL
    if X_shap_full is None:
        X_shap_full = _WORKER_X_SHAP_FULL
    if y_encoded_train is None:
        y_encoded_train = _WORKER_Y_ENCODED_TRAIN
    if le is None:
        le = _WORKER_LE

    # Require initialized matrices before accessing fold indices.
    if X_train_full is None:
        raise RuntimeError(
            "Worker missing X_train_full. Did you forget to create the Pool with "
            "initializer=_init_pool_worker and initargs=(X_train_full, X_shap_full, y_encoded_train, le)?"
        )
    if y_encoded_train is None:
        raise RuntimeError(
            "Worker missing y_encoded_train. Did you forget to create the Pool with "
            "initializer=_init_pool_worker and initargs=(X_train_full, X_shap_full, y_encoded_train, le)?"
        )
    try:
        # Train the selected real or permuted specialist fold.
        X_train_subset = X_train_full.iloc[train_indices][selected_features]
        X_test_subset = X_train_full.iloc[test_indices][selected_features]
        
        if is_permuted and y_permuted_train is not None:
            # Use permuted labels for training (encode them)
            y_train_fold_raw = y_permuted_train[train_indices]  
            y_train_fold = le.transform(y_train_fold_raw)  
            # Test set: always use true labels (not permuted)
            y_test_fold = y_encoded_train[test_indices]
        else:
            # Use true labels (normal case)
            y_train_fold = y_encoded_train[train_indices]
            y_test_fold = y_encoded_train[test_indices]
    
        if isinstance(X_train_subset, pd.DataFrame):
            if X_train_subset.columns.duplicated().any():
                X_train_subset = X_train_subset.loc[:, ~X_train_subset.columns.duplicated()]
            if isinstance(X_train_subset.columns, pd.MultiIndex):
                X_train_subset.columns = X_train_subset.columns.get_level_values(-1)
            X_train_subset = X_train_subset.values
        
        if isinstance(X_test_subset, pd.DataFrame):
            if X_test_subset.columns.duplicated().any():
                X_test_subset = X_test_subset.loc[:, ~X_test_subset.columns.duplicated()]
            if isinstance(X_test_subset.columns, pd.MultiIndex):
                X_test_subset.columns = X_test_subset.columns.get_level_values(-1)
            X_test_subset = X_test_subset.values
        
        model = xgb.XGBClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            subsample=model_subsample,
            random_state=(int(global_random_seed) + h_hit * 10000 + subset_idx * 100 + fold_idx) if global_random_seed is not None else None,
            objective='binary:logistic',
            eval_metric='logloss',
            n_jobs=1
        )
        model.fit(X_train_subset, y_train_fold)
        
        # Calculate SHAP values (separate for train and test)
        explainer = make_tree_explainer(model)
        
        # In-sample SHAP
        shap_train = explainer.shap_values(X_train_subset)
        if isinstance(shap_train, list):
            shap_train = shap_train[1]  # Binary classification
        
        # Out-of-sample SHAP
        shap_test = explainer.shap_values(X_test_subset)
        if isinstance(shap_test, list):
            shap_test = shap_test[1]
        
        probas_train = model.predict_proba(X_train_subset).astype(np.float32)
        probas_test = model.predict_proba(X_test_subset).astype(np.float32)
        
        # Track BOTH training and test accuracy.
        # - accuracy_test: generalization estimate (fold test set)
        # - accuracy_train: in-sample fit (helps diagnose overfitting)
        y_pred_train = model.predict(X_train_subset)
        accuracy_train = accuracy_score(y_train_fold, y_pred_train)
        y_pred_test = model.predict(X_test_subset)
        accuracy_test = accuracy_score(y_test_fold, y_pred_test)

        
        # Calculate SHAP for all groups (if needed)
        shap_all_groups = None
        probas_all_groups = None
        if X_shap_full is not None:
            X_shap_subset = X_shap_full[selected_features]
            if isinstance(X_shap_subset, pd.DataFrame):
                if X_shap_subset.columns.duplicated().any():
                    X_shap_subset = X_shap_subset.loc[:, ~X_shap_subset.columns.duplicated()]
                if isinstance(X_shap_subset.columns, pd.MultiIndex):
                    X_shap_subset.columns = X_shap_subset.columns.get_level_values(-1)
                X_shap_subset = X_shap_subset.values
            
            shap_all_groups = explainer.shap_values(X_shap_subset)
            if isinstance(shap_all_groups, list):
                shap_all_groups = shap_all_groups[1]
            probas_all_groups = model.predict_proba(X_shap_subset).astype(np.float32)
        
        # Save model
        # - Real models are reloaded later for logit aggregation and (optionally) weighted SHAP.
        # - Permuted models must never overwrite real models; save them into a perm-specific folder.
        model_path = None
        if paths and 'models_folder' in paths:
            try:
                base_models_folder = paths['models_folder']
                os.makedirs(base_models_folder, exist_ok=True)

                if is_permuted:
                    perm_tag = f"perm{permutation_idx}" if permutation_idx is not None else "perm"
                    perm_folder = os.path.join(base_models_folder, perm_tag)
                    os.makedirs(perm_folder, exist_ok=True)
                    model_path = os.path.join(
                        perm_folder,
                        f'h{h_hit}_s{subset_idx}_f{fold_idx}.json'
                    )
                else:
                    model_path = os.path.join(
                        base_models_folder,
                        f'h{h_hit}_s{subset_idx}_f{fold_idx}.json'
                    )

                model.save_model(model_path)
            except Exception:
                model_path = None
        
        del model, explainer
        gc.collect()
        
        return {
            'h_hit': h_hit,
            'subset_idx': subset_idx,
            'fold_idx': fold_idx,
            'features': selected_features,
                'accuracy_test': accuracy_test,
                'accuracy_train': accuracy_train,
                'n_features': len(selected_features),
            'shap_train': shap_train.astype(np.float32),
            'shap_test': shap_test.astype(np.float32),
            'shap_all_groups': shap_all_groups.astype(np.float32) if shap_all_groups is not None else None,
            'probas_train': probas_train,
            'probas_test': probas_test,
            'probas_all_groups': probas_all_groups if probas_all_groups is not None else None,
            'train_indices': train_indices,
            'test_indices': test_indices,
                'model_path': model_path,
                'is_permuted': is_permuted,
                'permutation_idx': permutation_idx
        }
        
    except Exception as e:
        return {
            '__worker_error__': True,
            'h_hit': h_hit,
            'subset_idx': subset_idx,
            'fold_idx': fold_idx,
            'is_permuted': bool(is_permuted),
            'permutation_idx': permutation_idx,
            'error_type': type(e).__name__,
            'error_message': str(e),
            'traceback': traceback.format_exc(),
        }

