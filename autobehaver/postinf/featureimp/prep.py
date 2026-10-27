"""Name FeatureImp runs, filter sex strata, parse feature families, and apply
completeness and variance criteria.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


def get_analysis_folder_name(
    analysis_run_name: str = "",
    sex_stratum: Optional[str] = None,
) -> str:
    """Generate the analysis folder name based on run name and sex stratum."""

    base_name = "FEATURE_ANALYSIS"

    if sex_stratum:
        s = str(sex_stratum).lower()
        if s == "male":
            base_name += "_MALE"
        elif s == "female":
            base_name += "_FEMALE"

    if analysis_run_name:
        sanitized_run_name = re.sub(r"[<>:\"/\\|?*]", "_", str(analysis_run_name))
        return f"{base_name}_{sanitized_run_name}"
    return base_name


def filter_data_by_sex(df: pd.DataFrame, sex_value: Any) -> pd.DataFrame:
    """Filter dataframe to include only the specified sex."""

    if "sex" not in df.columns:
        raise ValueError("Dataframe does not contain 'sex' column")

    sex_value_upper = str(sex_value).upper()
    if sex_value_upper in ["M", "MALE"]:
        filtered_df = df[df["sex"].astype(str).str.upper().isin(["M", "MALE"])].copy()
    elif sex_value_upper in ["F", "FEMALE"]:
        filtered_df = df[df["sex"].astype(str).str.upper().isin(["F", "FEMALE"])].copy()
    else:
        filtered_df = df[df["sex"].astype(str).str.upper() == sex_value_upper].copy()

    if len(filtered_df) == 0:
        available_sexes = df["sex"].unique() if "sex" in df.columns else []
        raise ValueError(
            f"No data found for sex '{sex_value}'. Available sexes: {available_sexes}"
        )

    return filtered_df



def _parse_feature_name(feature_name: str) -> Dict[str, Any]:
    """Parse feature name to extract source, category, smoothing, and measurement components."""

    # Bout-based kinematic features carry an explicit smoothing token.
    # Bout-based matrices may use Kinematic_[Category]_[Smoothing]_[Measurement].
    kin_match = re.match(r"(KinematicBB|Kinematic)_(.+?)_(inst|5|9)_(.+)", feature_name)
    if kin_match:
        return {
            "source": kin_match.group(1),
            "category": kin_match.group(2),
            "smoothing": kin_match.group(3),
            "measurement": kin_match.group(4),
            "grouping_key": f"{kin_match.group(1)}_{kin_match.group(2)}_{kin_match.group(4)}",  # ignore smoothing for grouping
        }

    # For other features, try to extract source
    other_match = re.match(r"([^_]+)_(.+)", feature_name)
    if other_match:
        return {
            "source": other_match.group(1),
            "category": None,
            "smoothing": None,
            "measurement": other_match.group(2),
            "grouping_key": feature_name,
        }

    # Fallback
    return {
        "source": "Unknown",
        "category": None,
        "smoothing": None,
        "measurement": feature_name,
        "grouping_key": feature_name,
    }


def _filter_sparse_features_by_group(
    df: pd.DataFrame,
    training_group_a: str,
    training_group_b: str,
    min_completeness: float,
    feature_cols: Sequence[str],
) -> List[str]:
    """Filter features based on minimum completeness in BOTH training groups."""

    if min_completeness <= 0:
        return list(feature_cols)

    # Create training subset
    training_df = df[df["group"].isin([training_group_a, training_group_b])].copy()

    if training_df.empty:
        print("  WARNING: No training group data found. Skipping sparsity filtering.")
        return list(feature_cols)

    # Check group presence
    groups_present = training_df["group"].unique()
    if training_group_a not in groups_present:
        print(
            f"  WARNING: Training group '{training_group_a}' not found in data. Skipping sparsity filtering."
        )
        return list(feature_cols)
    if training_group_b not in groups_present:
        print(
            f"  WARNING: Training group '{training_group_b}' not found in data. Skipping sparsity filtering."
        )
        return list(feature_cols)

    group_a_df = training_df[training_df["group"] == training_group_a]
    group_b_df = training_df[training_df["group"] == training_group_b]

    n_a = len(group_a_df)
    n_b = len(group_b_df)

    if n_a == 0 or n_b == 0:
        print("  WARNING: One or both training groups are empty. Skipping sparsity filtering.")
        return list(feature_cols)

    # Calculate completeness for each feature in each group
    features_to_keep: List[str] = []
    dropped_features: List[Dict[str, Any]] = []

    for col in feature_cols:
        # Group A completeness
        completeness_a = group_a_df[col].notna().sum() / n_a

        # Group B completeness
        completeness_b = group_b_df[col].notna().sum() / n_b

        # Keep if meets threshold in both groups
        if completeness_a >= min_completeness and completeness_b >= min_completeness:
            features_to_keep.append(col)
        else:
            dropped_features.append(
                {
                    "feature": col,
                    "completeness_a": float(completeness_a),
                    "completeness_b": float(completeness_b),
                }
            )

    if dropped_features:
        print(
            f"  - Dropped {len(dropped_features)} features with insufficient representation:"
        )
        for i, info in enumerate(dropped_features[:5]):
            print(
                f"    {i+1}. {info['feature']}: {info['completeness_a']*100:.1f}% in {training_group_a}, {info['completeness_b']*100:.1f}% in {training_group_b}"
            )
        if len(dropped_features) > 5:
            print(f"    ... and {len(dropped_features) - 5} more")

    return features_to_keep


def _filter_low_variance_features(
    X_train_full: pd.DataFrame,
    y_train: pd.Series,
    training_group_a: str,
    training_group_b: str,
    feature_cols: Sequence[str],
    min_variance_threshold: float = 1e-6,
) -> Tuple[List[str], Optional[Dict[str, Any]]]:
    """Filter out features with very low variance in training data."""

    group_a_mask = (y_train == training_group_a)
    group_b_mask = (y_train == training_group_b)

    # Calculate std for each feature (data is already imputed)
    filtering_stats: Dict[str, Any] = {
        "zero_variance_count": 0,
        "near_zero_variance_count": 0,
        "zero_variance_examples": [],
        "near_zero_variance_examples": [],
    }

    features_to_keep: List[str] = []
    zero_variance_features: List[str] = []
    near_zero_variance_features: List[str] = []

    for col in feature_cols:
        feature_std = X_train_full[col].std()

        # Check for zero variance
        if (
            pd.isna(feature_std)
            or feature_std == 0
            or np.isclose(feature_std, 0, atol=1e-10)
        ):
            zero_variance_features.append(col)
            continue

        # Check for near-zero variance
        if float(feature_std) < float(min_variance_threshold):
            near_zero_variance_features.append(col)
            continue

        features_to_keep.append(col)

    # Record stats
    filtering_stats["zero_variance_count"] = len(zero_variance_features)
    filtering_stats["near_zero_variance_count"] = len(near_zero_variance_features)
    filtering_stats["zero_variance_examples"] = zero_variance_features[:10]
    filtering_stats["near_zero_variance_examples"] = near_zero_variance_features[:10]

    total_removed = (
        filtering_stats["zero_variance_count"]
        + filtering_stats["near_zero_variance_count"]
    )
    if total_removed > 0:
        print(f"  - Dropped {total_removed} low-variance features:")
        if filtering_stats["zero_variance_count"] > 0:
            print(
                f"    - Zero variance: {filtering_stats['zero_variance_count']} features"
            )
            if filtering_stats["zero_variance_examples"]:
                print(
                    f"      Examples: {', '.join(filtering_stats['zero_variance_examples'][:5])}"
                )

        if filtering_stats["near_zero_variance_count"] > 0:
            print(
                f"    - Near-zero variance (std < {min_variance_threshold}): {filtering_stats['near_zero_variance_count']} features"
            )
            if filtering_stats["near_zero_variance_examples"]:
                print(
                    f"      Examples: {', '.join(filtering_stats['near_zero_variance_examples'][:5])}"
                )

    return features_to_keep, filtering_stats
