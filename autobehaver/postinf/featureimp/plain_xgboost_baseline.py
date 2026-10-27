"""Train full-feature XGBoost comparison models using FeatureImp fold plans.

Aggregate predictions in logit space and report margin-scale SHAP, permutation
baselines, repeat stability, and optional FSEA.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import shutil
import time
import traceback
from datetime import datetime
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from .integrity import strict_bool
import numpy as np
import pandas as pd

from .compat import (
    _generate_kfold_shap_summaries,
    _generate_male_female_comparison,
)
from .config import DEFAULT_SETTINGS
from .baseline_common import (
    _build_predictions_dataframe,
    _build_shap_summary_frames,
    _build_training_to_all_positions,
    _compute_binary_metrics,
    _make_group_labels_binary,
    _probabilities_from_logits,
    _resolve_feature_matrix,
    _write_shap_outputs,
)
from .finalize import _aggregate_and_summarize_permuted_results
from .io_utils import save_to_excel_sheets
from .permutation import _generate_sex_class_stratified_permutation
from .reporting import create_formatted_predictions_for_graphpad, save_predictions_with_formatting
from .runtime_settings import FSEARuntimeSettings

OUTPUT_DIRNAME = "PLAIN_XGBOOST_BASELINE"
FOLD_PERFORMANCE_FILENAME = "PLAIN_XGBOOST_FOLD_PERFORMANCE.csv"
REPEAT_PERFORMANCE_FILENAME = "PLAIN_XGBOOST_REPEAT_PERFORMANCE.csv"
POOLED_PERFORMANCE_FILENAME = "PLAIN_XGBOOST_POOLED_PERFORMANCE.csv"
CONFIG_FILENAME = "PLAIN_XGBOOST_CONFIG.json"
PREDICTIONS_FULL_FILENAME = "AVERAGED_PREDICTIONS_FULL.xlsx"
PREDICTIONS_GRAPHPAD_FILENAME = "AVERAGED_PREDICTIONS_FORMATTED_FOR_GRAPHPAD.xlsx"
SHAP_SUMMARY_FILENAME = "SHAP_CONSOLIDATED_SUMMARY.xlsx"
SHAP_COMBINED_FILENAME = "SHAP_CONSOLIDATED_SUMMARY_INSAMPLE_VS_OUTOFSAMPLE.xlsx"
SHAP_STABILITY_FILENAME = "SHAP_FEATURE_STABILITY_ACROSS_REPEATS.csv"
SHAP_PER_INDIVIDUAL_FILENAME = "SHAP_PER_INDIVIDUAL_SUMMARY.xlsx"
SHAP_MALE_FEMALE_FILENAME = "SHAP_CONSOLIDATED_SUMMARY_MALE_VS_FEMALE.xlsx"
GLOBAL_IMPORTANCE_RANKING_FILENAME = "GLOBAL_IMPORTANCE_RANKING.csv"
MANIFEST_FILENAME = "PLAIN_XGBOOST_SHAP_MANIFEST.json"


def _baseline_log(started_at: float, message: str) -> None:
    elapsed = max(0.0, float(time.time() - float(started_at)))
    print(f"[featureimp plain_xgb +{elapsed:8.1f}s] {message}", flush=True)


def _build_real_fold_indices_local(fold_plan_record: Mapping[str, Any]) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
    folds = []
    for rec in list(fold_plan_record.get("folds") or []):
        train_idx = np.asarray(rec.get("train_indices") or [], dtype=np.int64)
        test_idx = np.asarray(rec.get("test_indices") or [], dtype=np.int64)
        folds.append((train_idx, test_idx))
    return tuple(folds)


def _positive_class_shap(explainer: Any, X: np.ndarray) -> np.ndarray:
    shap_values = explainer.shap_values(X)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    return np.asarray(shap_values, dtype=np.float32)


def _fit_plain_xgboost_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    n_estimators: int,
    learning_rate: float,
    max_depth: int,
    subsample: float,
    random_state: int,
    n_jobs: int,
) -> Any:
    """Fit one full-feature binary XGBoost classifier with the supplied model settings."""
    import xgboost as xgb

    model = xgb.XGBClassifier(
        n_estimators=int(n_estimators),
        learning_rate=float(learning_rate),
        max_depth=int(max_depth),
        subsample=float(subsample),
        colsample_bytree=1.0,
        random_state=int(random_state),
        objective="binary:logistic",
        eval_metric="logloss",
        n_jobs=int(max(1, n_jobs)),
    )
    model.fit(np.asarray(X_train, dtype=np.float32), np.asarray(y_train, dtype=np.int32))
    return model


def _prefix_perm_prediction_columns(df: pd.DataFrame, perm_idx: int) -> pd.DataFrame:
    out = df.copy()
    rename: Dict[str, str] = {}
    for col in out.columns:
        if col in {"individual", "group"}:
            continue
        rename[col] = f"Perm{int(perm_idx)}_{col}"
    return out.rename(columns=rename)


def _merge_prediction_views(*, base_df: pd.DataFrame, permuted_dfs: Sequence[Tuple[int, pd.DataFrame]]) -> pd.DataFrame:
    merged = base_df.copy()
    key_cols = ["individual", "group"]
    if not permuted_dfs:
        return merged
    merged["___SEPARATOR___"] = np.nan
    perm_prefixed = [_prefix_perm_prediction_columns(perm_df, int(perm_idx)) for perm_idx, perm_df in permuted_dfs]
    combined_permuted = perm_prefixed[0].copy()
    for perm_df in perm_prefixed[1:]:
        perm_df_no_base = perm_df.drop(columns=key_cols, errors="ignore")
        combined_permuted = pd.concat([combined_permuted, perm_df_no_base], axis=1)
    merged = pd.merge(merged, combined_permuted, on=key_cols, how="left", suffixes=("", "_perm"))
    return merged.drop(columns=["individual_perm", "group_perm"], errors="ignore")


def _finalize_metric_matrix(matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    arr = np.asarray(matrix, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D matrix for stability metrics, got shape {arr.shape!r}")
    mean_vals = np.nanmean(arr, axis=0)
    if arr.shape[0] < 2:
        std_vals = np.full_like(mean_vals, np.nan, dtype=np.float64)
        stability = np.full_like(mean_vals, np.nan, dtype=np.float64)
        return mean_vals.astype(float), std_vals.astype(float), stability.astype(float)
    std_vals = np.nanstd(arr, axis=0, ddof=0)
    stability = np.full_like(mean_vals, np.nan, dtype=np.float64)
    valid = np.isfinite(std_vals) & (std_vals > 0)
    stability[valid] = mean_vals[valid] / std_vals[valid]
    return mean_vals.astype(float), std_vals.astype(float), stability.astype(float)


def _compute_across_repeat_shap_stability(
    *,
    feature_names: Sequence[str],
    repeat_global_abs: Sequence[np.ndarray],
    repeat_group_signed: Mapping[str, Sequence[np.ndarray]],
    repeat_group_abs: Mapping[str, Sequence[np.ndarray]],
) -> pd.DataFrame:
    """Summarize mean, deviation, and stability of global and group SHAP across repeats."""
    rows: Dict[str, Any] = {"feature_name": list(feature_names)}
    if repeat_global_abs:
        global_matrix = np.vstack([np.asarray(v, dtype=np.float64).reshape(1, -1) for v in repeat_global_abs])
        mean_abs, std_abs, stab_abs = _finalize_metric_matrix(global_matrix)
    else:
        n = len(feature_names)
        mean_abs = np.full((n,), np.nan, dtype=np.float64)
        std_abs = np.full((n,), np.nan, dtype=np.float64)
        stab_abs = np.full((n,), np.nan, dtype=np.float64)
    rows["AcrossRepeats_NRepeats"] = np.full((len(feature_names),), int(len(repeat_global_abs)), dtype=np.int32)
    rows["AcrossRepeats_GlobalMeanAbsSHAP"] = mean_abs
    rows["AcrossRepeats_GlobalStdAbsSHAP"] = std_abs
    rows["AcrossRepeats_GlobalStabilityAbsSHAP"] = stab_abs

    ordered_groups = list(repeat_group_signed.keys())
    for group_name in ordered_groups:
        signed_list = list(repeat_group_signed.get(group_name) or [])
        abs_list = list(repeat_group_abs.get(group_name) or [])
        if signed_list:
            signed_matrix = np.vstack([np.asarray(v, dtype=np.float64).reshape(1, -1) for v in signed_list])
            mean_signed, std_signed, stab_signed = _finalize_metric_matrix(signed_matrix)
        else:
            n = len(feature_names)
            mean_signed = np.full((n,), np.nan, dtype=np.float64)
            std_signed = np.full((n,), np.nan, dtype=np.float64)
            stab_signed = np.full((n,), np.nan, dtype=np.float64)
        if abs_list:
            abs_matrix = np.vstack([np.asarray(v, dtype=np.float64).reshape(1, -1) for v in abs_list])
            mean_abs_g, std_abs_g, stab_abs_g = _finalize_metric_matrix(abs_matrix)
        else:
            n = len(feature_names)
            mean_abs_g = np.full((n,), np.nan, dtype=np.float64)
            std_abs_g = np.full((n,), np.nan, dtype=np.float64)
            stab_abs_g = np.full((n,), np.nan, dtype=np.float64)
        rows[f"AcrossRepeats_MeanSHAP_{group_name}"] = mean_signed
        rows[f"AcrossRepeats_StdSHAP_{group_name}"] = std_signed
        rows[f"AcrossRepeats_StabilitySHAP_{group_name}"] = stab_signed
        rows[f"AcrossRepeats_MeanAbsSHAP_{group_name}"] = mean_abs_g
        rows[f"AcrossRepeats_StdAbsSHAP_{group_name}"] = std_abs_g
        rows[f"AcrossRepeats_StabilityAbsSHAP_{group_name}"] = stab_abs_g
    return pd.DataFrame(rows)


def _cleanup_stale_permutation_outputs(out_dir: str) -> None:
    patterns = [
        os.path.join(out_dir, "PERMUTED_*_RESULTS"),
    ]
    for pattern in patterns:
        for path_str in glob.glob(pattern):
            try:
                if os.path.isdir(path_str):
                    shutil.rmtree(path_str)
                elif os.path.exists(path_str):
                    os.remove(path_str)
            except FileNotFoundError:
                continue


def _finalize_global_importance_columns(summary_df: pd.DataFrame) -> pd.DataFrame:
    """Normalize and cumulatively sum global importance after ordering features."""
    if summary_df is None or summary_df.empty or "Global_Importance" not in summary_df.columns:
        return summary_df
    out = summary_df.copy()
    out = out.sort_values(by=["Global_Importance", "feature_name"], ascending=[False, True], kind="mergesort").reset_index(drop=True)
    total_gi = float(pd.to_numeric(out["Global_Importance"], errors="coerce").fillna(0.0).sum())
    if total_gi > 1e-12:
        out["Global_Importance_Normalized"] = pd.to_numeric(out["Global_Importance"], errors="coerce").astype(float) / total_gi
        out["Global_Importance_Cumulative"] = pd.to_numeric(out["Global_Importance_Normalized"], errors="coerce").astype(float).cumsum()
        if len(out) > 0:
            out.iloc[-1, out.columns.get_loc("Global_Importance_Cumulative")] = 1.0
    else:
        out["Global_Importance_Normalized"] = np.nan
        out["Global_Importance_Cumulative"] = np.nan
    return out


def _write_global_importance_ranking(*, out_dir: str, summary_df: pd.DataFrame) -> str:
    """Export the ordered global-importance ranking and retained diagnostic columns."""
    if summary_df is None or summary_df.empty:
        ranking_df = pd.DataFrame(columns=["rank", "feature_name", "Global_Importance"])
    else:
        ranking_df = _finalize_global_importance_columns(summary_df)
        keep_cols = [
            c
            for c in [
                "feature_name",
                "Global_Importance",
                "Global_Importance_Raw",
                "Global_Importance_PermBaseline",
                "Global_Importance_Normalized",
                "Global_Importance_Cumulative",
            ]
            if c in ranking_df.columns
        ]
        ranking_df = ranking_df[keep_cols].copy()
        ranking_df.insert(0, "rank", np.arange(1, len(ranking_df) + 1, dtype=int))
    ranking_path = os.path.join(out_dir, GLOBAL_IMPORTANCE_RANKING_FILENAME)
    ranking_df.to_csv(ranking_path, index=False)
    return ranking_path


def _run_optional_plain_xgboost_fsea(
    *,
    out_dir: str,
    settings_snapshot: Optional[Mapping[str, Any]],
    oof_shap_values: np.ndarray,
    all_groups_avg_shap_values: np.ndarray,
    df_for_training: pd.DataFrame,
    df_for_shap_and_agg: pd.DataFrame,
    feature_names: Sequence[str],
    training_group_a: str,
    training_group_b: str,
    le: Any,
    num_classes: int,
) -> None:
    """Run baseline enrichment using its resolved settings and SHAP arrays."""
    from .fsea_runner import _perform_fsea_analysis

    settings = FSEARuntimeSettings.from_mapping(settings_snapshot or {})
    status = _perform_fsea_analysis(
        oof_shap_values=np.asarray(oof_shap_values, dtype=np.float64),
        df_for_training=df_for_training,
        all_feature_names=list(feature_names),
        training_group_a=str(training_group_a),
        training_group_b=str(training_group_b),
        df_for_shap_and_agg=df_for_shap_and_agg,
        paths={"analysis_folder": str(out_dir)},
        le=le,
        num_classes=int(num_classes),
        all_groups_avg_shap_values=np.asarray(all_groups_avg_shap_values, dtype=np.float64),
        settings=settings,
    )
    if status.get("status") in {"partial", "failed"}:
        raise RuntimeError("Required baseline FSEA incomplete: " + "; ".join(status.get("visualization_failures", [])))


def run_plain_xgboost_baseline(
    *,
    X_train_full: pd.DataFrame,
    df_for_training: pd.DataFrame,
    training_group_a: str,
    training_group_b: str,
    analysis_folder_path: str,
    n_splits: int,
    random_state: int,
    df_for_shap_and_agg: Optional[pd.DataFrame] = None,
    X_shap_full: Optional[pd.DataFrame] = None,
    le: Any = None,
    num_classes: Optional[int] = None,
    settings_snapshot: Optional[Mapping[str, Any]] = None,
    fold_plan_records: Optional[Mapping[int, Mapping[str, Any]]] = None,
    run_fsea_analysis: Optional[bool] = None,
) -> str:
    """Train full-feature XGBoost models for the frozen repeated folds and write
    comparison outputs.

    Use logit-space predictions and positive-class margin SHAP; optional permutation
    runs shuffle only training-fold labels within sex strata.
    """
    from .committee import FixedBinaryLabelEncoder
    from .shap_compat import make_tree_explainer

    started_at = time.time()

    if len(X_train_full) != len(df_for_training):
        raise ValueError("X_train_full and df_for_training must have the same number of rows.")
    if int(n_splits) < 2:
        raise ValueError(f"Plain XGBoost baseline requires n_splits >= 2, got {n_splits!r}.")
    if not fold_plan_records:
        raise ValueError(
            "Plain XGBoost baseline requires fold_plan_records from the staged FeatureImp plan so it can reuse the exact h-hit folds."
        )

    settings_map = dict(settings_snapshot or {})
    resolved_run_fsea = strict_bool(
        run_fsea_analysis if run_fsea_analysis is not None else settings_map.get("run_fsea_analysis", False)
    )
    run_permutation_test = strict_bool(settings_map.get("run_permutation_test", False))
    n_permutations = int(settings_map.get("n_permutations", 0) or 0) if run_permutation_test else 0
    num_workers = max(1, int(settings_map.get("num_workers", 1) or 1))

    feature_names = [str(c) for c in list(X_train_full.columns)]
    X_np = _resolve_feature_matrix(X_train_full, feature_names)
    df_train = df_for_training.copy()
    y = _make_group_labels_binary(df_train, training_group_a, training_group_b)
    y_original_labels = df_train["group"].astype(str).to_numpy(copy=False)

    if df_for_shap_and_agg is None:
        df_all = df_train.copy()
    else:
        df_all = df_for_shap_and_agg.copy()
    if X_shap_full is None:
        X_all_np = X_np.copy()
    else:
        X_all_np = _resolve_feature_matrix(X_shap_full, feature_names)
    if len(df_all) != int(X_all_np.shape[0]):
        raise ValueError(
            "df_for_shap_and_agg and X_shap_full must have the same number of rows. "
            f"Observed {len(df_all)} vs {X_all_np.shape[0]}."
        )

    if le is None:
        le = FixedBinaryLabelEncoder(class0=str(training_group_a), class1=str(training_group_b))
    resolved_num_classes = int(num_classes) if num_classes is not None else 2
    if int(resolved_num_classes) != 2:
        raise ValueError(
            f"Plain XGBoost baseline expects binary classification, got num_classes={resolved_num_classes!r}."
        )

    train_to_all_positions = _build_training_to_all_positions(df_train, df_all)
    n_train = int(len(df_train))
    n_all = int(len(df_all))
    n_feats = int(len(feature_names))

    out_dir = os.path.join(str(analysis_folder_path), OUTPUT_DIRNAME)
    os.makedirs(out_dir, exist_ok=True)
    _cleanup_stale_permutation_outputs(out_dir)

    X_np32 = np.ascontiguousarray(np.asarray(X_np, dtype=np.float32))
    X_all_np32 = np.ascontiguousarray(np.asarray(X_all_np, dtype=np.float32))

    n_estimators = int(settings_map.get("n_estimators", DEFAULT_SETTINGS.get("n_estimators", 1000)))
    learning_rate = float(settings_map.get("learning_rate", DEFAULT_SETTINGS.get("learning_rate", 0.01)))
    max_depth = int(settings_map.get("max_depth", DEFAULT_SETTINGS.get("max_depth", 6)))
    model_subsample = float(settings_map.get("model_subsample", DEFAULT_SETTINGS.get("model_subsample", 0.8)))

    _baseline_log(
        started_at,
        f"starting plain-XGBoost baseline: n_train={n_train}, n_all={n_all}, n_features={n_feats}, n_estimators={n_estimators}, max_depth={max_depth}, num_workers={num_workers}, permutations={n_permutations}",
    )

    oof_margin_sum = np.zeros(n_train, dtype=np.float64)
    oof_count = np.zeros(n_train, dtype=np.int32)
    insample_margin_sum = np.zeros(n_train, dtype=np.float64)
    insample_count = np.zeros(n_train, dtype=np.int32)
    all_margin_sum = np.zeros(n_all, dtype=np.float64)
    all_model_count = 0

    oof_shap_sum = np.zeros((n_train, n_feats), dtype=np.float32)
    insample_shap_sum = np.zeros((n_train, n_feats), dtype=np.float32)
    all_shap_sum = np.zeros((n_all, n_feats), dtype=np.float32)

    fold_rows: List[Dict[str, Any]] = []
    repeat_rows: List[Dict[str, Any]] = []
    repeat_global_abs: List[np.ndarray] = []
    all_groups = [str(g) for g in pd.unique(df_all["group"].astype(str))]
    training_groups = {str(training_group_a), str(training_group_b)}
    ordered_stability_groups = [str(training_group_a), str(training_group_b)] + [g for g in all_groups if g not in training_groups]
    repeat_group_signed: Dict[str, List[np.ndarray]] = {g: [] for g in ordered_stability_groups}
    repeat_group_abs: Dict[str, List[np.ndarray]] = {g: [] for g in ordered_stability_groups}

    sorted_h_hits = sorted(int(h) for h in list(fold_plan_records.keys()))
    if not sorted_h_hits:
        raise ValueError("Plain XGBoost baseline received an empty fold plan.")
    planned_repeats = int(len(sorted_h_hits))

    actual_outer_splits: Optional[int] = None
    for repeat_zero_idx, h_hit in enumerate(sorted_h_hits):
        repeat_idx = int(repeat_zero_idx + 1)
        _baseline_log(started_at, f"starting h_hit={h_hit} ({repeat_idx}/{planned_repeats})")
        fold_indices_round = _build_real_fold_indices_local(fold_plan_records[int(h_hit)])
        if actual_outer_splits is None:
            actual_outer_splits = int(len(fold_indices_round))
        elif int(actual_outer_splits) != int(len(fold_indices_round)):
            raise ValueError(
                "Plain XGBoost baseline encountered inconsistent outer fold counts across h_hit rounds. "
                f"Expected {actual_outer_splits}, observed {len(fold_indices_round)} in h_hit={h_hit}."
            )
        if int(len(fold_indices_round)) != int(n_splits):
            raise ValueError(
                "Plain XGBoost baseline fold-plan mismatch. "
                f"Requested n_splits={n_splits}, plan contains {len(fold_indices_round)} folds for h_hit={h_hit}."
            )

        repeat_oof_margin = np.full(n_train, np.nan, dtype=np.float64)
        repeat_oof_count = np.zeros(n_train, dtype=np.int32)
        repeat_oof_shap_sum = np.zeros((n_train, n_feats), dtype=np.float32)
        repeat_oof_shap_count = np.zeros(n_train, dtype=np.int32)
        repeat_all_shap_sum = np.zeros((n_all, n_feats), dtype=np.float32)
        repeat_all_model_count = 0

        for fold_zero_idx, (train_idx, test_idx) in enumerate(fold_indices_round):
            fold_idx = int(fold_zero_idx + 1)
            train_idx = np.asarray(train_idx, dtype=np.int64)
            test_idx = np.asarray(test_idx, dtype=np.int64)
            train_all_pos = train_to_all_positions[train_idx]
            test_all_pos = train_to_all_positions[test_idx]

            model = _fit_plain_xgboost_model(
                X_np32[train_idx],
                y[train_idx],
                n_estimators=int(n_estimators),
                learning_rate=float(learning_rate),
                max_depth=int(max_depth),
                subsample=float(model_subsample),
                random_state=int(random_state + h_hit * 10000 + fold_idx),
                n_jobs=int(num_workers),
            )
            explainer = make_tree_explainer(model)
            margins_all = np.asarray(
                model.predict(X_all_np32, output_margin=True),
                dtype=np.float64,
            )
            shap_all = _positive_class_shap(explainer, X_all_np32)
            if margins_all.shape[0] != n_all:
                raise ValueError(
                    "Plain XGBoost baseline margin length mismatch. "
                    f"Expected {n_all}, observed {margins_all.shape[0]}."
                )
            if shap_all.shape != (n_all, n_feats):
                raise ValueError(
                    "Plain XGBoost baseline SHAP shape mismatch. "
                    f"Expected {(n_all, n_feats)}, observed {shap_all.shape}."
                )

            oof_margin = margins_all[test_all_pos]
            insample_margin = margins_all[train_all_pos]
            oof_margin_sum[test_idx] += oof_margin
            oof_count[test_idx] += 1
            insample_margin_sum[train_idx] += insample_margin
            insample_count[train_idx] += 1
            all_margin_sum += margins_all
            all_model_count += 1

            oof_shap_sum[test_idx, :] += shap_all[test_all_pos, :]
            insample_shap_sum[train_idx, :] += shap_all[train_all_pos, :]
            all_shap_sum += shap_all

            repeat_oof_margin[test_idx] = oof_margin
            repeat_oof_count[test_idx] += 1
            repeat_oof_shap_sum[test_idx, :] += shap_all[test_all_pos, :]
            repeat_oof_shap_count[test_idx] += 1
            repeat_all_shap_sum += shap_all
            repeat_all_model_count += 1

            test_metrics = _compute_binary_metrics(y[test_idx], _probabilities_from_logits(oof_margin))
            fold_rows.append({
                "repeat_idx": int(repeat_idx),
                "h_hit": int(h_hit),
                "fold_idx": int(fold_idx),
                "n_train": int(len(train_idx)),
                "n_test": int(len(test_idx)),
                "n_estimators": int(n_estimators),
                "learning_rate": float(learning_rate),
                "max_depth": int(max_depth),
                "subsample": float(model_subsample),
                "accuracy_test": float(test_metrics.get("accuracy", np.nan)),
                "accuracy_train": np.nan,
                **test_metrics,
            })
            del explainer
            del model

        if np.any(repeat_oof_count != 1):
            raise ValueError(
                "Plain XGBoost baseline repeat OOF bookkeeping failed. "
                f"Expected exactly one OOF prediction per training row in h_hit={h_hit}."
            )
        if np.any(repeat_oof_shap_count != 1):
            raise ValueError(
                "Plain XGBoost baseline repeat OOF SHAP bookkeeping failed. "
                f"Expected exactly one OOF SHAP contribution per training row in h_hit={h_hit}."
            )
        if int(repeat_all_model_count) != int(len(fold_indices_round)):
            raise ValueError(
                "Plain XGBoost baseline repeat all-groups SHAP bookkeeping failed. "
                f"Expected {len(fold_indices_round)} models, observed {repeat_all_model_count} in h_hit={h_hit}."
            )

        repeat_oof_probs = _probabilities_from_logits(repeat_oof_margin)
        repeat_metrics = _compute_binary_metrics(y, repeat_oof_probs)
        repeat_rows.append({
            "repeat_idx": int(repeat_idx),
            "h_hit": int(h_hit),
            "n_outer_folds": int(len(fold_indices_round)),
            **{f"oof_{k}": v for k, v in repeat_metrics.items()},
        })

        repeat_oof_shap = (repeat_oof_shap_sum / np.maximum(repeat_oof_shap_count[:, None], 1)).astype(np.float32)
        repeat_all_shap = (repeat_all_shap_sum / float(max(repeat_all_model_count, 1))).astype(np.float32)
        repeat_global_abs.append(np.mean(np.abs(repeat_oof_shap), axis=0).astype(np.float32))
        for group_name in ordered_stability_groups:
            if group_name in training_groups:
                mask = df_train["group"].astype(str).to_numpy(copy=False) == str(group_name)
                group_vals = repeat_oof_shap[mask, :]
            else:
                mask = df_all["group"].astype(str).to_numpy(copy=False) == str(group_name)
                group_vals = repeat_all_shap[mask, :]
            if group_vals.shape[0] == 0:
                continue
            repeat_group_signed[group_name].append(np.mean(group_vals, axis=0).astype(np.float32))
            repeat_group_abs[group_name].append(np.mean(np.abs(group_vals), axis=0).astype(np.float32))

    if actual_outer_splits is None:
        raise ValueError("Plain XGBoost baseline did not produce any outer folds.")

    n_repeats = int(len(sorted_h_hits))
    if np.any(oof_count != int(n_repeats)):
        raise ValueError(
            "Plain XGBoost baseline OOF bookkeeping failed. "
            f"Expected every training row to receive {n_repeats} OOF predictions."
        )
    expected_insample_count = int(n_repeats) * max(int(actual_outer_splits) - 1, 0)
    if np.any(insample_count != expected_insample_count):
        raise ValueError(
            "Plain XGBoost baseline in-sample bookkeeping failed. "
            f"Expected every training row to receive {expected_insample_count} in-sample predictions."
        )
    if int(all_model_count) != int(n_repeats) * int(actual_outer_splits):
        raise ValueError(
            "Plain XGBoost baseline all-model accumulator count mismatch. "
            f"Expected {int(n_repeats) * int(actual_outer_splits)}, observed {all_model_count}."
        )

    oof_margin = oof_margin_sum / np.maximum(oof_count, 1)
    insample_margin = insample_margin_sum / np.maximum(insample_count, 1)
    all_margin = all_margin_sum / float(max(all_model_count, 1))
    oof_prob_group_b = _probabilities_from_logits(oof_margin)
    insample_prob_group_b = _probabilities_from_logits(insample_margin)
    all_prob_group_b = _probabilities_from_logits(all_margin)
    oof_shap_values = (oof_shap_sum / np.maximum(oof_count[:, None], 1)).astype(np.float64)
    insample_shap_values = (insample_shap_sum / np.maximum(insample_count[:, None], 1)).astype(np.float64)
    all_groups_avg_shap_values = (all_shap_sum / float(max(all_model_count, 1))).astype(np.float64)

    # Optional permutation baseline and reports.
    permuted_global_importances: List[np.ndarray] = []
    permuted_predictions_views: List[Tuple[int, pd.DataFrame]] = []
    if int(n_permutations) > 0:
        for perm_idx in range(int(n_permutations)):
            _baseline_log(started_at, f"starting permutation {int(perm_idx) + 1}/{int(n_permutations)}")
            perm_oof_margin_sum = np.zeros(n_train, dtype=np.float64)
            perm_oof_count = np.zeros(n_train, dtype=np.int32)
            perm_insample_margin_sum = np.zeros(n_train, dtype=np.float64)
            perm_insample_count = np.zeros(n_train, dtype=np.int32)
            perm_all_margin_sum = np.zeros(n_all, dtype=np.float64)
            perm_all_model_count = 0
            perm_oof_shap_sum = np.zeros((n_train, n_feats), dtype=np.float32)
            perm_insample_shap_sum = np.zeros((n_train, n_feats), dtype=np.float32)
            perm_all_shap_sum = np.zeros((n_all, n_feats), dtype=np.float32)
            perm_fold_rows: List[Dict[str, Any]] = []

            for repeat_zero_idx, h_hit in enumerate(sorted_h_hits):
                repeat_idx = int(repeat_zero_idx + 1)
                fold_indices_round = _build_real_fold_indices_local(fold_plan_records[int(h_hit)])
                for fold_zero_idx, (train_idx, test_idx) in enumerate(fold_indices_round):
                    fold_idx = int(fold_zero_idx + 1)
                    train_idx = np.asarray(train_idx, dtype=np.int64)
                    test_idx = np.asarray(test_idx, dtype=np.int64)
                    train_all_pos = train_to_all_positions[train_idx]
                    test_all_pos = train_to_all_positions[test_idx]
                    y_permuted_full = _generate_sex_class_stratified_permutation(
                        y_original=y_original_labels,
                        df_with_sex=df_train,
                        train_indices=train_idx,
                        test_indices=test_idx,
                        permutation_seed=int(random_state + (perm_idx + 1) * 1_000_000 + h_hit * 10_000 + fold_idx),
                    )
                    y_train_permuted = np.asarray(np.asarray(y_permuted_full)[train_idx] == str(training_group_b), dtype=np.int32)
                    model = _fit_plain_xgboost_model(
                        X_np32[train_idx],
                        y_train_permuted,
                        n_estimators=int(n_estimators),
                        learning_rate=float(learning_rate),
                        max_depth=int(max_depth),
                        subsample=float(model_subsample),
                        random_state=int(random_state + (perm_idx + 1) * 1_000_000 + h_hit * 10_000 + fold_idx),
                        n_jobs=int(num_workers),
                    )
                    explainer = make_tree_explainer(model)
                    margins_all = np.asarray(model.predict(X_all_np32, output_margin=True), dtype=np.float64)
                    shap_all = _positive_class_shap(explainer, X_all_np32)
                    margins_all_for_agg = np.asarray(margins_all, dtype=np.float64).copy()
                    flip_mask_train = np.asarray(np.asarray(y_original_labels)[train_idx] != np.asarray(y_permuted_full)[train_idx], dtype=bool)
                    if np.any(flip_mask_train):
                        margins_all_for_agg[train_all_pos[flip_mask_train]] *= -1.0

                    perm_oof_margin = margins_all_for_agg[test_all_pos]
                    perm_insample_margin = margins_all_for_agg[train_all_pos]
                    perm_oof_margin_sum[test_idx] += perm_oof_margin
                    perm_oof_count[test_idx] += 1
                    perm_insample_margin_sum[train_idx] += perm_insample_margin
                    perm_insample_count[train_idx] += 1
                    perm_all_margin_sum += margins_all_for_agg
                    perm_all_model_count += 1
                    perm_oof_shap_sum[test_idx, :] += shap_all[test_all_pos, :]
                    perm_insample_shap_sum[train_idx, :] += shap_all[train_all_pos, :]
                    perm_all_shap_sum += shap_all

                    fold_metrics = _compute_binary_metrics(y[test_idx], _probabilities_from_logits(perm_oof_margin))
                    perm_fold_rows.append({
                        "permutation_idx": int(perm_idx),
                        "repeat_idx": int(repeat_idx),
                        "h_hit": int(h_hit),
                        "fold_idx": int(fold_idx),
                        "n_train": int(len(train_idx)),
                        "n_test": int(len(test_idx)),
                        "accuracy_test": float(fold_metrics.get("accuracy", np.nan)),
                        "accuracy_train": np.nan,
                        **fold_metrics,
                    })
                    del explainer
                    del model

            if np.any(perm_oof_count != int(n_repeats)):
                raise ValueError(
                    "Plain XGBoost permuted baseline OOF bookkeeping failed. "
                    f"Expected every training row to receive {n_repeats} OOF predictions for permutation {perm_idx}."
                )
            if np.any(perm_insample_count != expected_insample_count):
                raise ValueError(
                    "Plain XGBoost permuted baseline in-sample bookkeeping failed. "
                    f"Expected every training row to receive {expected_insample_count} in-sample predictions for permutation {perm_idx}."
                )
            if int(perm_all_model_count) != int(n_repeats) * int(actual_outer_splits):
                raise ValueError(
                    "Plain XGBoost permuted baseline all-model count mismatch. "
                    f"Expected {int(n_repeats) * int(actual_outer_splits)}, observed {perm_all_model_count} for permutation {perm_idx}."
                )

            perm_oof_margin = perm_oof_margin_sum / np.maximum(perm_oof_count, 1)
            perm_insample_margin = perm_insample_margin_sum / np.maximum(perm_insample_count, 1)
            perm_all_margin = perm_all_margin_sum / float(max(perm_all_model_count, 1))
            perm_oof_prob_group_b = _probabilities_from_logits(perm_oof_margin)
            perm_insample_prob_group_b = _probabilities_from_logits(perm_insample_margin)
            perm_all_prob_group_b = _probabilities_from_logits(perm_all_margin)
            perm_oof_shap = (perm_oof_shap_sum / np.maximum(perm_oof_count[:, None], 1)).astype(np.float64)
            perm_insample_shap = (perm_insample_shap_sum / np.maximum(perm_insample_count[:, None], 1)).astype(np.float64)
            perm_all_shap = (perm_all_shap_sum / float(max(perm_all_model_count, 1))).astype(np.float64)
            permuted_global_importances.append(np.mean(np.abs(perm_oof_shap), axis=0).astype(np.float64))
            perm_predictions_df = _build_predictions_dataframe(
                df_for_training=df_train,
                df_for_shap_and_agg=df_all,
                train_to_all_positions=train_to_all_positions,
                training_group_a=str(training_group_a),
                training_group_b=str(training_group_b),
                oof_prob_group_b=perm_oof_prob_group_b,
                insample_prob_group_b=perm_insample_prob_group_b,
                all_prob_group_b=perm_all_prob_group_b,
            )
            permuted_predictions_views.append((int(perm_idx), perm_predictions_df))
            _aggregate_and_summarize_permuted_results(
                df=df_all,
                selected_features=feature_names,
                all_permuted_performance=perm_fold_rows,
                all_permuted_shap_values=None,
                le=le,
                paths={"analysis_folder": out_dir, "file_suffix": ""},
                permutation_idx=int(perm_idx),
                permuted_insample_shap=perm_insample_shap,
                permuted_oof_shap=perm_oof_shap,
                df_for_training=df_train,
                train_groups_list=[str(training_group_a), str(training_group_b)],
                n_splits=int(n_splits),
                permuted_all_groups_avg_probas=np.column_stack([1.0 - perm_all_prob_group_b, perm_all_prob_group_b]).astype(np.float32),
                permuted_all_groups_avg_shap_values=perm_all_shap,
                df_for_shap_and_agg=df_all,
                permuted_insample_probas=np.column_stack([1.0 - perm_insample_prob_group_b, perm_insample_prob_group_b]).astype(np.float32),
                permuted_oof_probas=np.column_stack([1.0 - perm_oof_prob_group_b, perm_oof_prob_group_b]).astype(np.float32),
                all_fold_indices=None,
            )
    perm_baseline = (
        np.mean(np.stack(permuted_global_importances, axis=0), axis=0).astype(np.float64)
        if permuted_global_importances
        else np.zeros((n_feats,), dtype=np.float64)
    )

    _baseline_log(started_at, "writing plain-XGBoost baseline outputs")

    fold_perf_df = pd.DataFrame(fold_rows)
    repeat_perf_df = pd.DataFrame(repeat_rows)
    pooled_perf_df = pd.DataFrame([
        {
            "n_repeats": int(n_repeats),
            "outer_n_splits_requested": int(n_splits),
            "outer_n_splits_actual": int(actual_outer_splits),
            "n_outer_models": int(all_model_count),
            "oof_prediction_count_per_training_row": int(np.min(oof_count)) if len(oof_count) > 0 else 0,
            "insample_prediction_count_per_training_row": int(np.min(insample_count)) if len(insample_count) > 0 else 0,
            **{f"oof_{k}": v for k, v in _compute_binary_metrics(y, oof_prob_group_b).items()},
            **{f"insample_{k}": v for k, v in _compute_binary_metrics(y, insample_prob_group_b).items()},
        }
    ])

    base_predictions_df = _build_predictions_dataframe(
        df_for_training=df_train,
        df_for_shap_and_agg=df_all,
        train_to_all_positions=train_to_all_positions,
        training_group_a=str(training_group_a),
        training_group_b=str(training_group_b),
        oof_prob_group_b=oof_prob_group_b,
        insample_prob_group_b=insample_prob_group_b,
        all_prob_group_b=all_prob_group_b,
    )
    all_predictions_df = _merge_prediction_views(base_df=base_predictions_df, permuted_dfs=permuted_predictions_views)
    save_predictions_with_formatting(all_predictions_df, os.path.join(out_dir, PREDICTIONS_FULL_FILENAME))
    create_formatted_predictions_for_graphpad(
        base_predictions_df,
        training_group_a=str(training_group_a),
        training_group_b=str(training_group_b),
        analysis_folder=out_dir,
        file_suffix="",
        perm_idx=0,
    )

    summary_df, combined_df = _build_shap_summary_frames(
        feature_names=feature_names,
        df_for_training=df_train,
        df_for_shap_and_agg=df_all,
        training_group_a=str(training_group_a),
        training_group_b=str(training_group_b),
        oof_shap_values=oof_shap_values,
        insample_shap_values=insample_shap_values,
        all_groups_avg_shap_values=all_groups_avg_shap_values,
    )
    raw_map = summary_df.set_index("feature_name")["Global_Importance"].astype(float) if "Global_Importance" in summary_df.columns else pd.Series(dtype=float)
    perm_map = {str(name): float(val) for name, val in zip(feature_names, perm_baseline.tolist())}
    if "Global_Importance" in summary_df.columns:
        summary_df["Global_Importance_Raw"] = summary_df["Global_Importance"].astype(float)
        summary_df["Global_Importance_PermBaseline"] = summary_df["feature_name"].map(perm_map).astype(float)
        summary_df["Global_Importance"] = np.clip(
            summary_df["Global_Importance_Raw"].astype(float) - summary_df["Global_Importance_PermBaseline"].astype(float),
            a_min=0.0,
            a_max=None,
        )
        summary_df["Global_Importance_Normalized"] = np.nan
        summary_df["Global_Importance_Cumulative"] = np.nan
    if "Global_Importance_OUTOFSAMPLE" in combined_df.columns:
        combined_df["Global_Importance_OUTOFSAMPLE_Raw"] = combined_df["Global_Importance_OUTOFSAMPLE"].astype(float)
        combined_df["Global_Importance_OUTOFSAMPLE_PermBaseline"] = combined_df["feature_name"].map(perm_map).astype(float)
        combined_df["Global_Importance_OUTOFSAMPLE"] = np.clip(
            combined_df["Global_Importance_OUTOFSAMPLE_Raw"].astype(float) - combined_df["Global_Importance_OUTOFSAMPLE_PermBaseline"].astype(float),
            a_min=0.0,
            a_max=None,
        )

    stability_df = _compute_across_repeat_shap_stability(
        feature_names=feature_names,
        repeat_global_abs=repeat_global_abs,
        repeat_group_signed=repeat_group_signed,
        repeat_group_abs=repeat_group_abs,
    )
    stability_path = os.path.join(out_dir, SHAP_STABILITY_FILENAME)
    stability_df.to_csv(stability_path, index=False)
    summary_df = pd.merge(summary_df, stability_df, on="feature_name", how="left")
    summary_df = _finalize_global_importance_columns(summary_df)
    if len(combined_df) > 0 and "Global_Importance_OUTOFSAMPLE" in combined_df.columns:
        combined_df = combined_df.sort_values(by=["Global_Importance_OUTOFSAMPLE", "feature_name"], ascending=[False, True], kind="mergesort").reset_index(drop=True)

    summary_path, combined_path = _write_shap_outputs(out_dir=out_dir, summary_df=summary_df, combined_df=combined_df)
    ranking_path = _write_global_importance_ranking(out_dir=out_dir, summary_df=summary_df)

    # Per-individual and male/female comparison workbooks for parity with committee outputs.
    out_paths = {"analysis_folder": out_dir, "file_suffix": ""}
    outofsample_result = _generate_kfold_shap_summaries(
        shap_values=oof_shap_values,
        df=df_train,
        selected_features=feature_names,
        le=le,
        paths=out_paths,
        suffix="outofsample",
        train_groups=[str(training_group_a), str(training_group_b)],
    )
    insample_result = _generate_kfold_shap_summaries(
        shap_values=insample_shap_values,
        df=df_train,
        selected_features=feature_names,
        le=le,
        paths=out_paths,
        suffix="insample",
        train_groups=[str(training_group_a), str(training_group_b)],
    )
    if outofsample_result is not None and insample_result is not None:
        out_sheets, _ = outofsample_result
        in_sheets, _ = insample_result
        if out_sheets is not None and in_sheets is not None:
            combined_individual_sheets: Dict[str, pd.DataFrame] = {}
            for sheet_name, sheet_df in out_sheets.items():
                combined_individual_sheets[f"OutOfSample_{sheet_name}"[:31]] = sheet_df
            for sheet_name, sheet_df in in_sheets.items():
                combined_individual_sheets[f"InSample_{sheet_name}"[:31]] = sheet_df
            save_to_excel_sheets(combined_individual_sheets, os.path.join(out_dir, SHAP_PER_INDIVIDUAL_FILENAME))

    hybrid_shap = np.asarray(all_groups_avg_shap_values, dtype=np.float32).copy()
    hybrid_shap[np.asarray(train_to_all_positions, dtype=np.int64), :] = np.asarray(oof_shap_values, dtype=np.float32)
    _generate_male_female_comparison(
        df=df_all,
        selected_features=feature_names,
        all_shap_file_paths=[],
        le=le,
        paths=out_paths,
        avg_shap_values_override=hybrid_shap,
    )

    fold_perf_df.to_csv(os.path.join(out_dir, FOLD_PERFORMANCE_FILENAME), index=False)
    repeat_perf_df.to_csv(os.path.join(out_dir, REPEAT_PERFORMANCE_FILENAME), index=False)
    pooled_perf_df.to_csv(os.path.join(out_dir, POOLED_PERFORMANCE_FILENAME), index=False)

    config_payload = {
        "training_group_a": str(training_group_a),
        "training_group_b": str(training_group_b),
        "n_training_rows": int(len(df_train)),
        "n_all_group_rows": int(len(df_all)),
        "n_features": int(len(feature_names)),
        "outer_n_splits_requested": int(n_splits),
        "outer_n_splits_actual": int(actual_outer_splits),
        "n_repeats": int(n_repeats),
        "n_outer_models": int(all_model_count),
        "reuse_frozen_hhit_fold_plan": True,
        "n_estimators": int(n_estimators),
        "learning_rate": float(learning_rate),
        "max_depth": int(max_depth),
        "subsample": float(model_subsample),
        "colsample_bytree": 1.0,
        "random_state": int(random_state),
        "num_workers": int(num_workers),
        "run_permutation_test": bool(run_permutation_test),
        "n_permutations": int(n_permutations),
        "outputs": {
            "fold_performance": FOLD_PERFORMANCE_FILENAME,
            "repeat_performance": REPEAT_PERFORMANCE_FILENAME,
            "pooled_performance": POOLED_PERFORMANCE_FILENAME,
            "predictions_full": PREDICTIONS_FULL_FILENAME,
            "predictions_graphpad": PREDICTIONS_GRAPHPAD_FILENAME,
            "shap_summary": SHAP_SUMMARY_FILENAME,
            "shap_insample_vs_outofsample": SHAP_COMBINED_FILENAME,
            "shap_per_individual": SHAP_PER_INDIVIDUAL_FILENAME,
            "shap_male_vs_female": SHAP_MALE_FEMALE_FILENAME,
            "shap_stability": SHAP_STABILITY_FILENAME,
            "global_importance_ranking": GLOBAL_IMPORTANCE_RANKING_FILENAME,
        },
    }
    with open(os.path.join(out_dir, CONFIG_FILENAME), "w", encoding="utf-8") as f:
        json.dump(config_payload, f, indent=2)

    with open(os.path.join(out_dir, MANIFEST_FILENAME), "w", encoding="utf-8") as f:
        json.dump(
            {
                "n_repeats": int(n_repeats),
                "n_features": int(len(feature_names)),
                "global_importance_formula": "max(0, Global_Importance_Raw - Global_Importance_PermBaseline)",
                "permutation_baseline_used": bool(int(n_permutations) > 0),
                "stability_source": "across_repeats",
                "training_groups": [str(training_group_a), str(training_group_b)],
                "nontraining_groups": [g for g in all_groups if g not in training_groups],
            },
            f,
            indent=2,
        )

    if resolved_run_fsea:
        _baseline_log(started_at, "starting optional FSEA")
        _run_optional_plain_xgboost_fsea(
            out_dir=out_dir,
            settings_snapshot=settings_map,
            oof_shap_values=oof_shap_values,
            all_groups_avg_shap_values=all_groups_avg_shap_values,
            df_for_training=df_train,
            df_for_shap_and_agg=df_all,
            feature_names=feature_names,
            training_group_a=str(training_group_a),
            training_group_b=str(training_group_b),
            le=le,
            num_classes=resolved_num_classes,
        )
        _baseline_log(started_at, "optional FSEA completed")
    _baseline_log(started_at, "plain-XGBoost baseline complete")
    return out_dir


def rerun_plain_xgboost_baseline_from_analysis_folder(
    *,
    analysis_folder_path: str,
    settings_override: Optional[Mapping[str, Any]] = None,
    run_fsea_analysis: Optional[bool] = None,
) -> str:
    """Restore a staged FeatureImp run and execute its full-feature XGBoost comparison."""
    from .baseline_outputs import (
        PLAIN_XGBOOST_DIRNAME,
        PLAIN_XGBOOST_STAGE_NAME,
        collect_plain_xgboost_artifact_paths,
        load_featureimp_baseline_context,
        upsert_postanalysis_stage_record,
        write_baseline_outputs_summary,
    )
    from .postanalysis_logging import append_postanalysis_error_log

    loaded = load_featureimp_baseline_context(str(analysis_folder_path))
    context = dict(loaded["context"])
    settings_snapshot = loaded.get("settings_snapshot")
    effective_settings = dict(settings_snapshot or {})
    effective_settings.update(dict(settings_override or {}))
    resolved_run_fsea = strict_bool(
        run_fsea_analysis
        if run_fsea_analysis is not None
        else effective_settings.get("run_fsea_analysis", True)
    )
    training_group_a = str(loaded["training_group_a"])
    training_group_b = str(loaded["training_group_b"])
    n_splits = int(loaded["n_splits"])
    random_state = int(effective_settings.get("GLOBAL_RANDOM_SEED", loaded["global_random_seed"]))
    started_at = datetime.now().isoformat(timespec="seconds")

    try:
        out_dir = run_plain_xgboost_baseline(
            X_train_full=context["X_train_full"],
            df_for_training=context["df_for_training"],
            training_group_a=training_group_a,
            training_group_b=training_group_b,
            analysis_folder_path=str(analysis_folder_path),
            n_splits=n_splits,
            random_state=random_state,
            df_for_shap_and_agg=context.get("df_for_shap_and_agg"),
            X_shap_full=context.get("X_shap_full"),
            le=context.get("le"),
            num_classes=context.get("num_classes"),
            settings_snapshot=effective_settings,
            fold_plan_records=context.get("fold_plan_records"),
            run_fsea_analysis=resolved_run_fsea,
        )
        artifact_paths = collect_plain_xgboost_artifact_paths(str(analysis_folder_path))
        stage_record = {
            "stage": PLAIN_XGBOOST_STAGE_NAME,
            "status": "completed",
            "started_at": started_at,
            "completed_at": datetime.now().isoformat(timespec="seconds"),
            "output_dir": os.path.relpath(out_dir, str(analysis_folder_path)),
            "result_path": os.path.relpath(out_dir, str(analysis_folder_path)),
            "artifact_paths": [os.path.relpath(path, str(analysis_folder_path)) for path in artifact_paths],
            "context": {
                "invoked_by": "standalone_rerun",
                "training_group_a": training_group_a,
                "training_group_b": training_group_b,
                "feature_count": int(len(context["selected_features"])),
                "n_splits": int(n_splits),
                "n_repeats": int(len(context.get("fold_plan_records") or {})),
            },
            "summary": "Plain XGBoost comparison baseline outputs updated via standalone rerun.",
        }
        upsert_postanalysis_stage_record(str(analysis_folder_path), stage_record)
        write_baseline_outputs_summary(str(analysis_folder_path), settings_snapshot=effective_settings)
        return out_dir
    except Exception as e:
        tb = traceback.format_exc()
        log_path = append_postanalysis_error_log(
            analysis_folder=str(analysis_folder_path),
            stage_name=PLAIN_XGBOOST_STAGE_NAME,
            error=e,
            traceback_str=tb,
            context={
                "invoked_by": "standalone_rerun",
                "output_dir": os.path.join(str(analysis_folder_path), PLAIN_XGBOOST_DIRNAME),
            },
        )
        stage_record = {
            "stage": PLAIN_XGBOOST_STAGE_NAME,
            "status": "failed",
            "started_at": started_at,
            "failed_at": datetime.now().isoformat(timespec="seconds"),
            "output_dir": PLAIN_XGBOOST_DIRNAME,
            "error_type": type(e).__name__,
            "error": str(e),
            "error_log": os.path.relpath(log_path, str(analysis_folder_path)),
            "context": {"invoked_by": "standalone_rerun"},
        }
        upsert_postanalysis_stage_record(str(analysis_folder_path), stage_record)
        write_baseline_outputs_summary(str(analysis_folder_path), settings_snapshot=effective_settings)
        raise


def _build_arg_parser() -> argparse.ArgumentParser:
    """Build the standalone comparison-baseline parser."""
    parser = argparse.ArgumentParser(
        description="Rerun the FeatureImp plain-XGBoost baseline from an existing staged analysis folder.",
    )
    parser.add_argument("--analysis_folder_path", required=True, help="Path to an existing staged FeatureImp analysis folder.")
    parser.add_argument(
        "--run_fsea_analysis",
        dest="run_fsea_analysis",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Whether to run FSEA after the plain-XGBoost baseline. Defaults to the saved analysis setting, or enabled if that setting is absent.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run a standalone full-feature XGBoost comparison from the command line."""
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    out_dir = rerun_plain_xgboost_baseline_from_analysis_folder(
        analysis_folder_path=str(args.analysis_folder_path),
        run_fsea_analysis=args.run_fsea_analysis,
    )
    print(out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
