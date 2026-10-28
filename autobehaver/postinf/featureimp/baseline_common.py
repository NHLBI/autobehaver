"""Shared label, probability, prediction, and SHAP-summary helpers for FeatureImp
comparison baselines.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

from .reporting import save_consolidated_summary_with_formatting

SHAP_SUMMARY_FILENAME = "SHAP_CONSOLIDATED_SUMMARY.xlsx"
SHAP_COMBINED_FILENAME = "SHAP_CONSOLIDATED_SUMMARY_INSAMPLE_VS_OUTOFSAMPLE.xlsx"
_LOG_LOSS_EPS = 1e-15


def _make_group_labels_binary(df_for_training: pd.DataFrame, group_a: str, group_b: str) -> np.ndarray:
    """Encode the declared training groups as class 0 and class 1, rejecting other
    labels.
    """
    groups = df_for_training["group"].astype(str).to_numpy(copy=False)
    out = np.empty(len(groups), dtype=np.int32)
    for idx, g in enumerate(groups.tolist()):
        if g == str(group_a):
            out[idx] = 0
        elif g == str(group_b):
            out[idx] = 1
        else:
            raise ValueError(
                "FeatureImp baseline received df_for_training rows outside the two training groups. "
                f"Unexpected group value: {g!r}."
            )
    return out

def _sigmoid(logits: np.ndarray) -> np.ndarray:
    """Convert margins to probabilities with a numerically stable sigmoid."""
    arr = np.asarray(logits, dtype=np.float64)
    out = np.empty_like(arr, dtype=np.float64)
    pos_mask = arr >= 0
    neg_mask = ~pos_mask
    out[pos_mask] = 1.0 / (1.0 + np.exp(-arr[pos_mask]))
    exp_vals = np.exp(arr[neg_mask])
    out[neg_mask] = exp_vals / (1.0 + exp_vals)
    return out

def _probabilities_from_logits(logits: np.ndarray) -> np.ndarray:
    """Convert margins to bounded class-1 probabilities."""
    probs = _sigmoid(np.asarray(logits, dtype=np.float64))
    return np.clip(probs, _LOG_LOSS_EPS, 1.0 - _LOG_LOSS_EPS)

def _compute_binary_metrics(y_true: np.ndarray, prob_class1: np.ndarray) -> Dict[str, float]:
    """Calculate binary metrics from finite class-1 probabilities and aligned labels."""
    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        brier_score_loss,
        f1_score,
        log_loss,
        roc_auc_score,
    )

    y = np.asarray(y_true, dtype=np.int32)
    prob1 = np.asarray(prob_class1, dtype=np.float64)
    mask = np.isfinite(prob1)
    y = y[mask]
    prob1 = prob1[mask]
    if y.size == 0:
        return {
            "accuracy": float("nan"),
            "balanced_accuracy": float("nan"),
            "f1": float("nan"),
            "auc": float("nan"),
            "log_loss": float("nan"),
            "brier": float("nan"),
            "mean_predicted_probability_group_b": float("nan"),
        }

    prob1 = np.clip(prob1, _LOG_LOSS_EPS, 1.0 - _LOG_LOSS_EPS)
    preds = (prob1 >= 0.5).astype(np.int32)
    two_col_probas = np.column_stack([1.0 - prob1, prob1])
    auc_val = float("nan")
    try:
        if len(np.unique(y)) == 2:
            auc_val = float(roc_auc_score(y, prob1))
    except Exception:
        auc_val = float("nan")

    return {
        "accuracy": float(accuracy_score(y, preds)),
        "balanced_accuracy": float(balanced_accuracy_score(y, preds)),
        "f1": float(f1_score(y, preds, zero_division=0)),
        "auc": auc_val,
        "log_loss": float(log_loss(y, two_col_probas, labels=[0, 1])),
        "brier": float(brier_score_loss(y, prob1)),
        "mean_predicted_probability_group_b": float(np.mean(prob1)),
    }

def _resolve_feature_matrix(df_or_array: pd.DataFrame | np.ndarray, feature_names: Sequence[str]) -> np.ndarray:
    """Select ordered feature columns or validate the width of an array input."""
    if isinstance(df_or_array, pd.DataFrame):
        return np.asarray(df_or_array.loc[:, list(feature_names)], dtype=np.float64)
    arr = np.asarray(df_or_array, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D matrix, got shape={arr.shape!r}")
    if arr.shape[1] != int(len(feature_names)):
        raise ValueError(
            "FeatureImp baseline feature-matrix width mismatch. "
            f"Expected {len(feature_names)} columns, observed {arr.shape[1]}."
        )
    return arr

def _build_training_to_all_positions(df_for_training: pd.DataFrame, df_for_shap_and_agg: pd.DataFrame) -> np.ndarray:
    """Resolve training rows in the all-groups matrix and reject ambiguous animal
    identities.
    """
    if len(df_for_training) == len(df_for_shap_and_agg):
        same_order = False
        if "individual" in df_for_training.columns and "individual" in df_for_shap_and_agg.columns:
            try:
                same_order = bool(
                    np.array_equal(
                        df_for_training["individual"].astype(str).to_numpy(copy=False),
                        df_for_shap_and_agg["individual"].astype(str).to_numpy(copy=False),
                    )
                )
            except Exception:
                same_order = False
        if same_order or "individual" not in df_for_training.columns:
            return np.arange(len(df_for_training), dtype=np.int64)

    if "individual" in df_for_training.columns and "individual" in df_for_shap_and_agg.columns:
        train_ids = df_for_training["individual"].astype(str)
        all_ids = df_for_shap_and_agg["individual"].astype(str)
        if bool(train_ids.is_unique) and bool(all_ids.is_unique):
            all_pos = {str(v): idx for idx, v in enumerate(all_ids.tolist())}
            positions: List[int] = []
            for item in train_ids.tolist():
                if str(item) not in all_pos:
                    raise ValueError(
                        "FeatureImp baseline could not map a training-row individual into df_for_shap_and_agg: "
                        f"{item!r}."
                    )
                positions.append(int(all_pos[str(item)]))
            return np.asarray(positions, dtype=np.int64)

    train_idx = list(df_for_training.index)
    all_idx = list(df_for_shap_and_agg.index)
    if len(train_idx) > 0 and len(set(all_idx)) == len(all_idx):
        all_pos = {idx: pos for pos, idx in enumerate(all_idx)}
        if all(item in all_pos for item in train_idx):
            return np.asarray([int(all_pos[item]) for item in train_idx], dtype=np.int64)

    raise ValueError(
        "FeatureImp baseline could not map training rows into the all-groups frame. "
        "Provide df_for_shap_and_agg with unique 'individual' identifiers or matching row indices."
    )

def _build_predictions_dataframe(
    *,
    df_for_training: pd.DataFrame,
    df_for_shap_and_agg: pd.DataFrame,
    train_to_all_positions: np.ndarray,
    training_group_a: str,
    training_group_b: str,
    oof_prob_group_b: np.ndarray,
    insample_prob_group_b: np.ndarray,
    all_prob_group_b: np.ndarray,
) -> pd.DataFrame:
    """Build prediction tables with out-of-fold values for training rows and averaged
    values for other groups.
    """
    prob_cols = [f"Prob_{training_group_a}", f"Prob_{training_group_b}"]
    train_base = df_for_training[["individual", "group"]].reset_index(drop=True).copy() if "individual" in df_for_training.columns else df_for_training[["group"]].reset_index(drop=True).copy()
    if "individual" not in train_base.columns:
        train_base.insert(0, "individual", np.arange(len(train_base), dtype=np.int64))

    train_insample = pd.DataFrame({
        f"Prob_{training_group_a}_insample": 1.0 - np.asarray(insample_prob_group_b, dtype=np.float64),
        f"Prob_{training_group_b}_insample": np.asarray(insample_prob_group_b, dtype=np.float64),
    })
    train_oof = pd.DataFrame({
        f"Prob_{training_group_a}_outofsample": 1.0 - np.asarray(oof_prob_group_b, dtype=np.float64),
        f"Prob_{training_group_b}_outofsample": np.asarray(oof_prob_group_b, dtype=np.float64),
    })
    train_main = pd.DataFrame({
        prob_cols[0]: 1.0 - np.asarray(oof_prob_group_b, dtype=np.float64),
        prob_cols[1]: np.asarray(oof_prob_group_b, dtype=np.float64),
    })
    training_predictions_df = pd.concat([train_base, train_insample, train_oof, train_main], axis=1)

    all_base = df_for_shap_and_agg[["individual", "group"]].reset_index(drop=True).copy() if "individual" in df_for_shap_and_agg.columns else df_for_shap_and_agg[["group"]].reset_index(drop=True).copy()
    if "individual" not in all_base.columns:
        all_base.insert(0, "individual", np.arange(len(all_base), dtype=np.int64))
    train_pos_set = set(int(v) for v in np.asarray(train_to_all_positions, dtype=np.int64).tolist())
    non_training_mask = np.asarray([idx not in train_pos_set for idx in range(len(all_base))], dtype=bool)
    non_training_base = all_base.loc[non_training_mask].reset_index(drop=True)
    non_training_prob_group_b = np.asarray(all_prob_group_b, dtype=np.float64)[non_training_mask]

    non_training_insample = pd.DataFrame(
        np.nan,
        index=np.arange(len(non_training_base)),
        columns=[f"{c}_insample" for c in prob_cols],
    )
    non_training_oof = pd.DataFrame(
        np.nan,
        index=np.arange(len(non_training_base)),
        columns=[f"{c}_outofsample" for c in prob_cols],
    )
    non_training_main = pd.DataFrame({
        prob_cols[0]: 1.0 - non_training_prob_group_b,
        prob_cols[1]: non_training_prob_group_b,
    })
    non_training_predictions_df = pd.concat(
        [non_training_base, non_training_insample, non_training_oof, non_training_main],
        axis=1,
    )

    if len(non_training_predictions_df) > 0:
        return pd.concat([training_predictions_df, non_training_predictions_df], axis=0, ignore_index=True)
    return training_predictions_df.reset_index(drop=True)

def _build_shap_summary_frames(
    *,
    feature_names: Sequence[str],
    df_for_training: pd.DataFrame,
    df_for_shap_and_agg: pd.DataFrame,
    training_group_a: str,
    training_group_b: str,
    oof_shap_values: np.ndarray,
    insample_shap_values: np.ndarray,
    all_groups_avg_shap_values: np.ndarray,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Summarize signed and absolute positive-class SHAP by training group and all-group
    view.
    """
    oof_shap = np.asarray(oof_shap_values, dtype=np.float64)
    insample_shap = np.asarray(insample_shap_values, dtype=np.float64)
    all_shap = np.asarray(all_groups_avg_shap_values, dtype=np.float64)

    flat_rows: Dict[str, Any] = {
        "feature_name": list(feature_names),
        "Global_Importance": np.mean(np.abs(oof_shap), axis=0).astype(float),
        f"MeanSHAP_{training_group_a}": np.mean(oof_shap[df_for_training["group"].astype(str) == str(training_group_a), :], axis=0).astype(float),
        f"MeanSHAP_{training_group_b}": np.mean(oof_shap[df_for_training["group"].astype(str) == str(training_group_b), :], axis=0).astype(float),
        f"MeanAbsSHAP_{training_group_a}": np.mean(np.abs(oof_shap[df_for_training["group"].astype(str) == str(training_group_a), :]), axis=0).astype(float),
        f"MeanAbsSHAP_{training_group_b}": np.mean(np.abs(oof_shap[df_for_training["group"].astype(str) == str(training_group_b), :]), axis=0).astype(float),
    }

    all_groups = [str(g) for g in pd.unique(df_for_shap_and_agg["group"].astype(str))]
    training_groups = {str(training_group_a), str(training_group_b)}
    for group_name in all_groups:
        if group_name in training_groups:
            continue
        mask = df_for_shap_and_agg["group"].astype(str) == str(group_name)
        if int(mask.sum()) == 0:
            continue
        flat_rows[f"MeanSHAP_{group_name}"] = np.mean(all_shap[mask.to_numpy(copy=False), :], axis=0).astype(float)
        flat_rows[f"MeanAbsSHAP_{group_name}"] = np.mean(np.abs(all_shap[mask.to_numpy(copy=False), :]), axis=0).astype(float)

    summary_df = pd.DataFrame(flat_rows)
    if len(summary_df) > 0:
        summary_df = summary_df.sort_values(
            by=["Global_Importance", "feature_name"],
            ascending=[False, True],
        ).reset_index(drop=True)

    combined_rows: Dict[str, Any] = {
        "feature_name": list(feature_names),
        "Global_Importance_INSAMPLE": np.mean(np.abs(insample_shap), axis=0).astype(float),
        f"MeanSHAP_{training_group_a}_INSAMPLE": np.mean(insample_shap[df_for_training["group"].astype(str) == str(training_group_a), :], axis=0).astype(float),
        f"MeanSHAP_{training_group_b}_INSAMPLE": np.mean(insample_shap[df_for_training["group"].astype(str) == str(training_group_b), :], axis=0).astype(float),
        f"MeanAbsSHAP_{training_group_a}_INSAMPLE": np.mean(np.abs(insample_shap[df_for_training["group"].astype(str) == str(training_group_a), :]), axis=0).astype(float),
        f"MeanAbsSHAP_{training_group_b}_INSAMPLE": np.mean(np.abs(insample_shap[df_for_training["group"].astype(str) == str(training_group_b), :]), axis=0).astype(float),
        "Global_Importance_OUTOFSAMPLE": np.mean(np.abs(oof_shap), axis=0).astype(float),
        f"MeanSHAP_{training_group_a}_OUTOFSAMPLE": np.mean(oof_shap[df_for_training["group"].astype(str) == str(training_group_a), :], axis=0).astype(float),
        f"MeanSHAP_{training_group_b}_OUTOFSAMPLE": np.mean(oof_shap[df_for_training["group"].astype(str) == str(training_group_b), :], axis=0).astype(float),
        f"MeanAbsSHAP_{training_group_a}_OUTOFSAMPLE": np.mean(np.abs(oof_shap[df_for_training["group"].astype(str) == str(training_group_a), :]), axis=0).astype(float),
        f"MeanAbsSHAP_{training_group_b}_OUTOFSAMPLE": np.mean(np.abs(oof_shap[df_for_training["group"].astype(str) == str(training_group_b), :]), axis=0).astype(float),
    }

    for group_name in all_groups:
        if group_name in training_groups:
            continue
        mask = df_for_shap_and_agg["group"].astype(str) == str(group_name)
        if int(mask.sum()) == 0:
            continue
        combined_rows[f"MeanSHAP_{group_name}_POOLED"] = np.mean(all_shap[mask.to_numpy(copy=False), :], axis=0).astype(float)
        combined_rows[f"MeanAbsSHAP_{group_name}_POOLED"] = np.mean(np.abs(all_shap[mask.to_numpy(copy=False), :]), axis=0).astype(float)

    combined_df = pd.DataFrame(combined_rows)
    if len(combined_df) > 0:
        combined_df = combined_df.sort_values(
            by=["Global_Importance_OUTOFSAMPLE", "feature_name"],
            ascending=[False, True],
        ).reset_index(drop=True)

    return summary_df, combined_df

def _write_shap_outputs(
    *,
    out_dir: str,
    summary_df: pd.DataFrame,
    combined_df: pd.DataFrame,
) -> Tuple[str, str]:
    """Write consolidated and in-sample/out-of-fold SHAP workbooks."""
    summary_path = os.path.join(out_dir, SHAP_SUMMARY_FILENAME)
    combined_path = os.path.join(out_dir, SHAP_COMBINED_FILENAME)
    save_consolidated_summary_with_formatting(summary_df, summary_path)
    save_consolidated_summary_with_formatting(
        combined_df,
        combined_path,
        is_combined_insample_outofsample=True,
    )
    return summary_path, combined_path
