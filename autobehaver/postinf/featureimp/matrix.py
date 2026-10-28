"""Load, validate, and normalize FeatureImp matrices.

Remove unavailable declared features and absent transitions before row-wise CLR
transformation, then standardize numeric features against the control subset. Save the
normalization parameters as run provenance.
"""

from __future__ import annotations

import gc
import csv
import hashlib
import json
import os
from collections import Counter
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from autobehaver.atomic_io import atomic_write_dataframe_csv, atomic_write_json, sha256_file
from .matrix_input import verify_upstream_receipt, validate_numeric_columns


DEFAULT_ID_COLUMNS = ['individual', 'group', 'sex']
OPTIONAL_ID_COLUMNS = ['source_files', 'total_frames_analyzed']


def validate_feature_matrix(
    matrix_path: str,
    required_id_cols: List[str] | None = None,
) -> Tuple[bool, Optional[str], Optional[pd.DataFrame]]:
    """Validate that a feature matrix has the expected structure."""
    if required_id_cols is None:
        required_id_cols = list(DEFAULT_ID_COLUMNS)

    try:
        with open(matrix_path, newline="", encoding="utf-8-sig") as handle:
            raw_headers = next(csv.reader(handle), [])
        if not raw_headers:
            return False, "Feature matrix has no header row", None
        blank_headers = [index for index, value in enumerate(raw_headers) if not str(value).strip()]
        if blank_headers:
            return False, f"Blank column name(s) at positions: {blank_headers[:20]}", None
        duplicate_headers = sorted(
            value for value, count in Counter(raw_headers).items() if count > 1
        )
        if duplicate_headers:
            return False, f"Duplicate column names: {duplicate_headers[:20]}", None

        from .matrix_input import read_feature_matrix
        df = read_feature_matrix(matrix_path)

        missing_cols = [col for col in required_id_cols if col not in df.columns]
        if missing_cols:
            return False, f"Missing required columns: {missing_cols}", None

        feature_cols = [col for col in df.columns if col not in required_id_cols]
        if len(feature_cols) == 0:
            return False, "No feature columns found", None

        if len(df) == 0:
            return False, "DataFrame is empty", None

        for column in required_id_cols:
            blank = df[column].isna() | df[column].astype(str).str.strip().eq("")
            if bool(blank.any()):
                return False, f"Required identity column {column!r} contains blank values", None

        individual_ids = df["individual"].astype(str).str.strip()
        if individual_ids.duplicated().any():
            duplicates = sorted(
                set(individual_ids[individual_ids.duplicated(keep=False)].tolist())
            )[:20]
            return False, f"Duplicate individual identifiers: {duplicates}", None

        candidate_feature_cols = [
            column
            for column in df.columns
            if column not in set(required_id_cols) | set(OPTIONAL_ID_COLUMNS)
        ]
        validate_numeric_columns(df, candidate_feature_cols)
        df.attrs['input_provenance'] = verify_upstream_receipt(matrix_path)
        numeric_feature_cols = df[candidate_feature_cols].select_dtypes(include=[np.number]).columns
        if len(numeric_feature_cols) == 0:
            return False, "No numeric feature columns found", None
        numeric_values = df[list(numeric_feature_cols)].to_numpy(dtype=np.float64, copy=False)
        if np.isinf(numeric_values).any():
            return False, "Numeric feature matrix contains infinite values", None

        if "total_frames_analyzed" in df.columns:
            try:
                total_frames = pd.to_numeric(df["total_frames_analyzed"], errors="raise")
            except Exception:
                return False, "total_frames_analyzed contains non-numeric values", None
            populated = total_frames.dropna()
            if bool((populated <= 0).any()):
                return False, "total_frames_analyzed must be positive when provided", None

        return True, None, df

    except Exception as e:
        return False, f"Error loading matrix: {str(e)}", None



def compute_feature_schema_sha256(feature_columns: Sequence[str]) -> str:
    """Return a stable SHA-256 digest for an ordered feature list."""
    payload = "\n".join([str(c) for c in feature_columns]).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()



def _is_timebudget_feature(feature_name: str) -> bool:
    name = str(feature_name)
    return 'TimeBudget' in name



def _is_transition_probability_feature(feature_name: str) -> bool:
    name = str(feature_name)
    return name.startswith('Transition_') and '_to_' in name



def _is_compositional_feature(feature_name: str) -> bool:
    return _is_timebudget_feature(feature_name) or _is_transition_probability_feature(feature_name)



def infer_compositional_sets(feature_columns: Sequence[str]) -> Dict[str, List[str]]:
    """Group time-budget columns and true transition-probability columns for CLR
    transformation.

    Other Transition-prefixed summaries remain ordinary numeric features.
    """
    compositional_cols = [
        str(col) for col in feature_columns
        if _is_compositional_feature(str(col))
    ]

    compositional_sets: Dict[str, List[str]] = {}
    for col in compositional_cols:
        parts = col.split('_')
        if _is_timebudget_feature(col):
            if col.startswith('TimeBudget_PoseGivenOrient_'):
                key = 'TimeBudget_PoseGivenOrient_' + parts[-1]
            elif col.startswith('TimeBudget_OrientGivenPose_'):
                key = 'TimeBudget_OrientGivenPose_' + parts[-1]
            else:
                key = '_'.join(parts[:2])
        elif _is_transition_probability_feature(col):
            key = '_'.join(parts[:-1]) + '_'
        else:
            continue

        compositional_sets.setdefault(key, []).append(col)

    return compositional_sets



def _filter_transition_probability_features(
    df: pd.DataFrame,
    transition_cols: Sequence[str],
    *,
    control_group_tag: Optional[str],
    training_group_a: Optional[str],
    training_group_b: Optional[str],
    min_training_positive_fraction: float = 0.10,
) -> Tuple[List[str], List[Dict[str, Any]]]:
    """Return surviving transition-probability columns and drop metadata.

    Filtering happens on the *raw* master matrix before CLR normalization so
    all-zero / effectively absent transitions do not acquire meaningless CLR
    values from epsilon replacement.
    """
    transition_cols = [str(col) for col in transition_cols]
    if len(transition_cols) == 0:
        return [], []

    group_values = df['group'].fillna('').astype(str) if 'group' in df.columns else pd.Series('', index=df.index)

    if control_group_tag:
        control_mask = group_values.str.contains(str(control_group_tag), case=False, na=False, regex=False)
    else:
        control_mask = pd.Series(False, index=df.index)
    control_n = int(control_mask.sum())

    training_groups = {
        str(g) for g in (training_group_a, training_group_b)
        if g is not None and str(g) != ''
    }
    training_mask = group_values.isin(training_groups) if training_groups else pd.Series(False, index=df.index)
    training_n = int(training_mask.sum())

    kept_cols: List[str] = []
    dropped_records: List[Dict[str, Any]] = []

    for col in transition_cols:
        numeric_values = pd.to_numeric(df[col], errors='coerce')

        control_values = numeric_values.loc[control_mask]
        control_arr = control_values.to_numpy(dtype=float, copy=False)
        control_present_mask = np.isfinite(control_arr) & (control_arr > 0)
        control_positive_count = int(control_present_mask.sum())

        training_values = numeric_values.loc[training_mask]
        training_arr = training_values.to_numpy(dtype=float, copy=False)
        training_present_mask = np.isfinite(training_arr) & (training_arr > 0)
        training_positive_count = int(training_present_mask.sum())
        training_positive_fraction = (float(training_positive_count) / float(training_n)) if training_n > 0 else None

        drop_reason: Optional[str] = None
        if control_n > 0 and control_positive_count == 0:
            drop_reason = 'control_group_all_zero_or_nan'
        elif training_n > 0 and training_positive_fraction is not None and training_positive_fraction < float(min_training_positive_fraction):
            drop_reason = 'low_training_group_prevalence'

        if drop_reason is None:
            kept_cols.append(col)
            continue

        dropped_records.append(
            {
                'feature': str(col),
                'drop_reason': str(drop_reason),
                'control_group_tag_used': '' if control_group_tag is None else str(control_group_tag),
                'control_row_count': int(control_n),
                'control_positive_count': int(control_positive_count),
                'training_group_a': '' if training_group_a is None else str(training_group_a),
                'training_group_b': '' if training_group_b is None else str(training_group_b),
                'training_row_count': int(training_n),
                'training_positive_count': int(training_positive_count),
                'training_positive_fraction': (None if training_positive_fraction is None else float(training_positive_fraction)),
                'min_training_positive_fraction': float(min_training_positive_fraction),
            }
        )

    return kept_cols, dropped_records



def _clr_transform(values: np.ndarray, epsilon: float = 1e-12) -> np.ndarray:
    """Apply centered log-ratios after replacing exact zeros with positive epsilon.

    A vector containing NaN remains entirely unavailable; negative or infinite
    components are rejected.
    """
    arr = np.asarray(values, dtype=np.float64)
    if not np.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("CLR epsilon must be positive and finite")
    if np.isinf(arr).any() or np.any(arr[np.isfinite(arr)] < 0):
        raise ValueError("Compositions must contain nonnegative finite values or unavailable NaNs")
    if np.isnan(arr).any():
        return np.full_like(arr, np.nan)
    arr = np.where(arr == 0, epsilon, arr)
    gmean = np.exp(np.mean(np.log(arr)))
    return np.log(arr / gmean)



def _apply_clr_to_dataframe(
    df: pd.DataFrame,
    compositional_sets: Mapping[str, Sequence[str]],
) -> pd.DataFrame:
    """Transform each compositional set row-wise in place and return the same DataFrame."""
    for _key, cols in compositional_sets.items():
        cols_list = [str(c) for c in cols]
        if len(cols_list) == 0:
            continue
        df[cols_list] = df[cols_list].apply(
            lambda row: _clr_transform(row.values),
            axis=1,
            result_type='expand',
        )
    return df



def load_normalization_params(params_path: str) -> Dict[str, Any]:
    """Load and minimally validate a saved normalization-params JSON."""
    with open(params_path, 'r') as f:
        params = json.load(f)

    if not isinstance(params, dict):
        raise ValueError(f"Normalization params must be a JSON object: {params_path}")
    if 'means' not in params or 'stds' not in params:
        raise ValueError(
            f"Normalization params missing required 'means'/'stds' keys: {params_path}"
        )
    if not isinstance(params['means'], dict) or not isinstance(params['stds'], dict):
        raise ValueError(
            f"Normalization params 'means'/'stds' must be JSON objects: {params_path}"
        )

    if 'numeric_feature_columns' not in params:
        # Derive feature columns when the saved params omit the explicit list.
        numeric_feature_columns = list(params['means'].keys())
        params['numeric_feature_columns'] = numeric_feature_columns
    if 'compositional_sets' not in params:
        params['compositional_sets'] = infer_compositional_sets(params['numeric_feature_columns'])
    if 'scalar_feature_columns' not in params:
        comp = {c for cols in params['compositional_sets'].values() for c in cols}
        params['scalar_feature_columns'] = [
            str(c) for c in params['numeric_feature_columns'] if str(c) not in comp
        ]
    if 'id_columns' not in params:
        params['id_columns'] = list(DEFAULT_ID_COLUMNS)

    return params



def normalize_feature_matrix(
    results_dir: str,
    raw_matrix_path: Optional[str] = None,
    *,
    skip_if_normalized_exists: bool = False,
    analysis_folder_path: Optional[str] = None,
    control_group_tag: Optional[str] = None,
    training_group_a: Optional[str] = None,
    training_group_b: Optional[str] = None,
) -> Optional[str]:
    """Normalize the master feature matrix.

    Loads the raw feature matrix, applies CLR transformation to compositional
    features and Z-score standardization to all numeric features, then saves
    the result.
    """
    print("\n--- Normalizing Feature Matrix ---")

    save_folder = analysis_folder_path if analysis_folder_path else results_dir

    normalized_output_path = os.path.join(save_folder, 'NORMALIZED_FEATURE_MATRIX.csv')
    if raw_matrix_path is None:
        raw_matrix_path = os.path.join(results_dir, 'MASTER_FEATURE_MATRIX.csv')
    receipt_path = os.path.join(save_folder, 'NORMALIZATION_RECEIPT.json')
    upstream_input = verify_upstream_receipt(raw_matrix_path)
    identity = {'raw_sha256': sha256_file(raw_matrix_path), 'source_sha256': sha256_file(__file__),
                'matrix_input': upstream_input,
                'control_group_tag': control_group_tag or 'CONTROL', 'training_group_a': training_group_a, 'training_group_b': training_group_b}
    if skip_if_normalized_exists and os.path.exists(normalized_output_path):
        if not os.path.isfile(receipt_path):
            raise ValueError('Normalized matrix lacks a verified receipt; use a fresh analysis folder')
        receipt = json.loads(open(receipt_path, encoding='utf-8').read())
        if receipt.get('identity') != identity or receipt.get('normalized_sha256') != sha256_file(normalized_output_path) or receipt.get('params_sha256') != sha256_file(os.path.join(save_folder, 'NORMALIZATION_PARAMS.json')):
            raise ValueError('Normalized matrix inputs/settings/outputs changed; reuse rejected')
        return normalized_output_path

    if not os.path.exists(raw_matrix_path):
        print(f"Error: '{raw_matrix_path}' not found. Cannot normalize.")
        parent_dir = os.path.dirname(raw_matrix_path) or "."
        print(f"Looked in directory: {parent_dir}")
        if os.path.isdir(parent_dir):
            try:
                csvs = sorted([p for p in os.listdir(parent_dir) if p.lower().endswith('.csv')])
                print(f"CSV files in that folder ({len(csvs)}):")
                for p in csvs[:30]:
                    print(f"  - {p}")
                if len(csvs) > 30:
                    print("  ...")
            except Exception as e:
                print(f"Could not list directory contents: {e}")
        return None

    print(f"Loading and validating feature matrix: {raw_matrix_path}")
    is_valid, error_msg, df = validate_feature_matrix(raw_matrix_path)
    if not is_valid or df is None:
        print(f"ERROR: Feature matrix validation failed: {error_msg}")
        return None

    print("Loaded raw feature matrix.")
    print(f"  - Shape: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"  - Groups: {sorted(df['group'].unique())}")

    id_cols_for_feature_selection = ['individual', 'group', 'sex']
    candidate_feature_cols = [col for col in df.columns if col not in set(id_cols_for_feature_selection) | set(OPTIONAL_ID_COLUMNS)]

    # Work on the analysis copy only. Empty declared master placeholders must
    # leave before CLR, otherwise one missing component poisons a whole set.
    globally_empty_features = [c for c in candidate_feature_cols if df[c].isna().all()]
    df.drop(columns=globally_empty_features, inplace=True)
    candidate_feature_cols = [c for c in candidate_feature_cols if c not in globally_empty_features]
    numeric_cols = df[candidate_feature_cols].select_dtypes(include=[np.number]).columns.tolist()
    non_numeric_cols = [col for col in candidate_feature_cols if col not in numeric_cols]

    if non_numeric_cols:
        print(f"  Warning: Excluding {len(non_numeric_cols)} non-numeric columns from normalization:")
        for col in non_numeric_cols[:10]:
            print(f"    - {col}")
        if len(non_numeric_cols) > 10:
            print(f"    ... and {len(non_numeric_cols) - 10} more")

    control_tag_to_use = control_group_tag if control_group_tag else "CONTROL"

    feature_cols = list(numeric_cols)
    transition_probability_cols = [col for col in feature_cols if _is_transition_probability_feature(col)]
    _kept_transition_cols, dropped_transition_records = _filter_transition_probability_features(
        df,
        transition_probability_cols,
        control_group_tag=control_tag_to_use,
        training_group_a=training_group_a,
        training_group_b=training_group_b,
        min_training_positive_fraction=0.10,
    )
    dropped_transition_cols = [str(rec['feature']) for rec in dropped_transition_records]
    if dropped_transition_cols:
        dropped_transition_set = set(dropped_transition_cols)
        df.drop(columns=dropped_transition_cols, inplace=True, errors='ignore')
        feature_cols = [col for col in feature_cols if col not in dropped_transition_set]
        reason_counts = pd.Series([rec['drop_reason'] for rec in dropped_transition_records]).value_counts().to_dict()
        print(
            f"Filtered {len(dropped_transition_cols)} transition probability features before CLR/Z-score normalization: "
            f"{reason_counts}"
        )
    else:
        print("No transition probability features required pre-normalization filtering.")

    compositional_cols = [col for col in feature_cols if _is_compositional_feature(col)]
    scalar_cols = [col for col in feature_cols if col not in compositional_cols]

    print(f"Identified {len(compositional_cols)} compositional features and {len(scalar_cols)} scalar features.")

    compositional_sets = infer_compositional_sets(compositional_cols)
    print(f"\nFound {len(compositional_sets)} distinct compositional sets to transform.")

    df = _apply_clr_to_dataframe(df, compositional_sets)
    print("CLR transformation applied to all compositional sets.")

    print("\nVerifying CLR transformation (each compositional set should sum to ~0 per row)...")
    failed_sets = []
    failed_max_dev: Dict[str, float] = {}
    for key, cols in compositional_sets.items():
        values = df[list(cols)]
        applicable = np.isfinite(values.to_numpy(dtype=float)).all(axis=1)
        sums = values.loc[applicable].sum(axis=1)
        max_dev = float(np.max(np.abs(sums))) if len(sums) else 0.0
        if not np.allclose(sums, 0):
            failed_sets.append(key)
            failed_max_dev[key] = max_dev

    if not failed_sets:
        print(f"All {len(compositional_sets)} compositional sets passed verification.")
    else:
        print(
            f"{len(failed_sets)}/{len(compositional_sets)} compositional sets FAILED verification. "
            "Please check the data / CLR transform for these sets:"
        )
        for key in failed_sets:
            print(f"  - FAILED: {key}* (Max deviation from zero: {failed_max_dev.get(key)})")

    print("\nApplying Z-score standardization...")

    numeric_features = scalar_cols + compositional_cols

    unique_groups = df['group'].dropna().unique()
    print(f"\nAvailable groups in dataset: {sorted(unique_groups)}")

    print(f"Searching for groups containing the control_tag_to_use: '{control_tag_to_use}'")
    control_df = df[df['group'].str.contains(control_tag_to_use, case=False, na=False, regex=False)]

    if control_df.empty:
        print(
            f"Error: No group containing '{control_tag_to_use}' found to calculate normalization parameters. "
            "Skipping Z-score."
        )
        print(f"Available groups were: {sorted(unique_groups)}")
        return None

    control_means = control_df[numeric_features].mean()
    control_stds = control_df[numeric_features].std()
    control_counts = control_df[numeric_features].count()
    insufficient = (control_counts < 2) | ~np.isfinite(control_stds)
    unsupported_reference = [str(c) for c in control_counts.index[insufficient]]
    if unsupported_reference:
        df.drop(columns=unsupported_reference, inplace=True)
        numeric_features = [c for c in numeric_features if c not in unsupported_reference]
        control_means = control_means[numeric_features]
        control_stds = control_stds[numeric_features]
        print(f"Excluded {len(unsupported_reference)} features with fewer than two finite control observations.")

    # CLR transforms can make a mathematically constant component differ only
    # by floating-point roundoff (typically around 1e-16).  Treating that
    # residue as a real standard deviation amplifies platform-specific noise
    # into order-one Z-scores.  Use an absolute floor comfortably above double
    # precision roundoff; a reference-constant feature then normalizes to its
    # centered value without division by numerical noise.
    zscore_std_floor = 1e-12
    near_zero_control_std = control_stds.abs() <= zscore_std_floor
    near_zero_control_std_features = [str(c) for c in control_stds.index[near_zero_control_std]]
    control_stds.loc[near_zero_control_std] = 1.0
    if near_zero_control_std_features:
        print(
            f"Stabilized {len(near_zero_control_std_features)} control-reference "
            f"standard deviation(s) <= {zscore_std_floor:g}."
        )

    zscore_round_decimals = 12
    normalized_numeric = (
        (df[numeric_features] - control_means) / control_stds
    ).round(decimals=zscore_round_decimals)
    # IEEE-754 signed zero is numerically equal but serializes differently on
    # different platforms; canonicalize it at the reproducibility boundary.
    df[numeric_features] = normalized_numeric.mask(normalized_numeric == 0, 0.0)
    print("Z-score normalization applied based on control group parameters.")

    print("\nChecking for and removing fully empty columns before saving...")
    original_feature_count = len(df.columns)
    df.dropna(axis=1, how='all', inplace=True)
    new_feature_count = len(df.columns)
    if original_feature_count != new_feature_count:
        print(f"  -> Permanently dropped {original_feature_count - new_feature_count} empty feature columns from the file.")

    kept_numeric_features = [str(c) for c in numeric_features if str(c) in df.columns]
    kept_feature_set = set(kept_numeric_features)
    id_columns_present = [
        str(col) for col in (list(DEFAULT_ID_COLUMNS) + OPTIONAL_ID_COLUMNS)
        if str(col) in df.columns and str(col) not in kept_feature_set
    ]
    kept_scalar_cols = [str(c) for c in scalar_cols if str(c) in kept_feature_set]
    kept_compositional_cols = [str(c) for c in compositional_cols if str(c) in kept_feature_set]
    kept_compositional_sets = {
        str(k): [str(c) for c in v if str(c) in kept_feature_set]
        for k, v in compositional_sets.items()
        if any(str(c) in kept_feature_set for c in v)
    }
    norm_params = {
        'means': {str(k): float(v) for k, v in control_means[kept_numeric_features].to_dict().items()},
        'stds': {str(k): float(v) for k, v in control_stds[kept_numeric_features].to_dict().items()},
        'control_group_tag_used': str(control_tag_to_use),
        'id_columns': id_columns_present,
        'numeric_feature_columns': kept_numeric_features,
        'scalar_feature_columns': kept_scalar_cols,
        'compositional_feature_columns': kept_compositional_cols,
        'compositional_sets': kept_compositional_sets,
        'clr_input_compositional_sets': {str(k): list(v) for k,v in compositional_sets.items()},
        'non_numeric_columns_ignored': [str(c) for c in non_numeric_cols],
        'globally_empty_features_removed_before_normalization': globally_empty_features,
        'matrix_input_provenance': df.attrs.get('input_provenance', {}),
        'transition_filtering': {
            'enabled': True,
            'control_zero_or_nan_rule': True,
            'min_training_positive_fraction': 0.10,
            'training_group_a_used': '' if training_group_a is None else str(training_group_a),
            'training_group_b_used': '' if training_group_b is None else str(training_group_b),
            'dropped_transition_feature_count': int(len(dropped_transition_records)),
            'dropped_transition_features': [str(rec['feature']) for rec in dropped_transition_records],
        },
        'zscore_std_floor': float(zscore_std_floor),
        'zscore_std_floor_feature_count': int(len(near_zero_control_std_features)),
        'zscore_std_floor_features': near_zero_control_std_features,
        'zscore_round_decimals': int(zscore_round_decimals),
        'control_finite_counts': {str(k): int(v) for k, v in control_counts.items()},
        'unsupported_control_reference_features': unsupported_reference,
        'unavailable_composition_policy': 'retain_entire_incomplete_set_as_nan',
        'normalization_method': 'clr_then_zscore_v4_canonical_rounding',
        'numeric_feature_schema_sha256': compute_feature_schema_sha256(kept_numeric_features),
    }
    params_path = os.path.join(save_folder, 'NORMALIZATION_PARAMS.json')
    atomic_write_json(norm_params, params_path)
    print(f"Normalization parameters saved to {params_path}")

    output_path = os.path.join(save_folder, 'NORMALIZED_FEATURE_MATRIX.csv')
    atomic_write_dataframe_csv(df, output_path, index=False)
    atomic_write_json({'identity': identity, 'normalized_sha256': sha256_file(output_path), 'params_sha256': sha256_file(params_path)}, receipt_path)
    print(f"Normalized feature matrix saved to {output_path}")

    gc.collect()

    return output_path
