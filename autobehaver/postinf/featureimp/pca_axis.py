"""Fit PCA on the two reference groups, project all individuals, and score distance
along and away from their centroid axis.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Mapping, Optional

import numpy as np
import pandas as pd

from .pca_utils import get_cached_pca_projection
from .postanalysis_logging import write_stage_manifest


def _safe_group_str(x: Any) -> str:
    if x is None:
        return "NA"
    try:
        if pd.isna(x):
            return "NA"
    except Exception:
        pass
    return str(x)


def _json_dump(path: str, payload: Mapping[str, Any]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(dict(payload), f, indent=2, sort_keys=True, default=str)
        f.write("\n")


def _compute_axis_scores(
    X_pca: np.ndarray,
    groups: np.ndarray,
    *,
    training_group_a: str,
    training_group_b: str,
) -> Optional[Dict[str, Any]]:
    """Compute reference-axis progress and orthogonal deviation.

    ``axis_t`` is 0 at ``training_group_b`` and 1 at ``training_group_a``.
    """

    arr = np.asarray(X_pca, dtype=float)
    if arr.ndim != 2 or arr.size == 0:
        return None

    g = np.asarray([_safe_group_str(v) for v in groups], dtype=object)
    mask_a = g == str(training_group_a)
    mask_b = g == str(training_group_b)
    if not np.any(mask_a) or not np.any(mask_b):
        return None

    mu_a = np.nanmean(arr[mask_a, :], axis=0)
    mu_b = np.nanmean(arr[mask_b, :], axis=0)
    d = mu_a - mu_b
    denom = float(np.dot(d, d))
    if not np.isfinite(denom) or denom <= 0:
        return None

    centered = arr - mu_b
    t = (centered @ d) / denom
    projected = np.outer(t, d)
    residual = centered - projected
    r = np.linalg.norm(residual, axis=1)

    training_sep = float(np.linalg.norm(d))
    if not np.isfinite(training_sep) or training_sep <= 0:
        return None
    r_norm = r / training_sep

    return {
        "t": np.asarray(t, dtype=float),
        "r": np.asarray(r, dtype=float),
        "r_norm": np.asarray(r_norm, dtype=float),
        "training_sep": training_sep,
        "mu_a": np.asarray(mu_a, dtype=float),
        "mu_b": np.asarray(mu_b, dtype=float),
        "axis_vector": np.asarray(d, dtype=float),
    }


def _quantile(series: pd.Series, q: float) -> float:
    try:
        return float(series.astype(float).quantile(float(q)))
    except Exception:
        return float("nan")


def _summarize_axis_scores_by_group(axis_df: pd.DataFrame) -> pd.DataFrame:
    """Summarize axis progress and orthogonal distance for each group."""
    columns = [
        "group",
        "n",
        "axis_t_mean",
        "axis_t_median",
        "axis_t_q25",
        "axis_t_q75",
        "axis_r_mean",
        "axis_r_median",
        "axis_r_q25",
        "axis_r_q75",
        "axis_r_norm_mean",
        "axis_r_norm_median",
        "axis_r_norm_q25",
        "axis_r_norm_q75",
    ]
    if axis_df.empty:
        return pd.DataFrame(columns=columns)

    rows = []
    for group, gdf in axis_df.groupby("group", dropna=False):
        t = gdf["axis_t"].astype(float)
        r = gdf["axis_r"].astype(float)
        rn = gdf["axis_r_norm"].astype(float)
        rows.append(
            {
                "group": _safe_group_str(group),
                "n": int(len(gdf)),
                "axis_t_mean": float(np.nanmean(t)) if len(t) else float("nan"),
                "axis_t_median": float(np.nanmedian(t)) if len(t) else float("nan"),
                "axis_t_q25": _quantile(t, 0.25),
                "axis_t_q75": _quantile(t, 0.75),
                "axis_r_mean": float(np.nanmean(r)) if len(r) else float("nan"),
                "axis_r_median": float(np.nanmedian(r)) if len(r) else float("nan"),
                "axis_r_q25": _quantile(r, 0.25),
                "axis_r_q75": _quantile(r, 0.75),
                "axis_r_norm_mean": float(np.nanmean(rn)) if len(rn) else float("nan"),
                "axis_r_norm_median": float(np.nanmedian(rn)) if len(rn) else float("nan"),
                "axis_r_norm_q25": _quantile(rn, 0.25),
                "axis_r_norm_q75": _quantile(rn, 0.75),
            }
        )
    return pd.DataFrame(rows, columns=columns).sort_values("group", kind="stable").reset_index(drop=True)


def run_pca_analysis(
    *,
    X_train: Any,
    X_all_groups: Any,
    df_train: pd.DataFrame,
    df_all_groups: pd.DataFrame,
    all_feature_names: Any,
    paths: Dict[str, Any],
    GLOBAL_RANDOM_SEED: Optional[int] = None,
    pca_fixed_k: Optional[int] = None,
    n_pca_components_to_save: int = 0,
    training_group_a: Optional[str] = None,
    training_group_b: Optional[str] = None,
) -> Dict[str, Any]:
    """Fit reference PCA and export individual axis scores and group summaries.

    Axis progress is 0 at group B and 1 at group A.
    """

    del df_train  # Parameter kept to match the caller signature.

    if training_group_a is None or training_group_b is None:
        raise ValueError("PCA analysis requires training_group_a and training_group_b.")
    for col in ("individual", "group", "sex"):
        if col not in df_all_groups.columns:
            raise ValueError(f"PCA analysis requires df_all_groups column {col!r}.")

    analysis_folder = os.path.abspath(str(paths.get("analysis_folder", ".") or "."))
    try:
        feature_names_list = [str(f) for f in list(all_feature_names)]
    except Exception:
        feature_names_list = []
    pca_dir = os.path.join(analysis_folder, "PCA")
    os.makedirs(pca_dir, exist_ok=True)

    cache_paths = dict(paths)
    cache_paths["analysis_folder"] = analysis_folder

    print("\n" + "=" * 80)
    print("Running PCA reference-axis analysis")
    print("=" * 80)

    effective_fixed_k = 50 if pca_fixed_k is None else int(pca_fixed_k)
    _scaler, _pca, _X_ref_pca, X_all_pca, k_selected, explained_var_ratio = get_cached_pca_projection(
        X_train,
        X_all_groups,
        paths=cache_paths,
        cache_key="pca_axis_reference",
        pca_fixed_k=int(effective_fixed_k),
        random_state=GLOBAL_RANDOM_SEED,
    )
    k_selected = int(k_selected)
    explained_var_ratio = np.asarray(explained_var_ratio, dtype=float)

    explained_var_df = pd.DataFrame(
        {
            "Component": [f"PC{i + 1}" for i in range(k_selected)],
            "Explained_Variance": explained_var_ratio,
            "Cumulative_Explained_Variance": np.cumsum(explained_var_ratio),
        }
    )
    explained_path = os.path.join(pca_dir, "PCA_EXPLAINED_VARIANCE.csv")
    explained_var_df.to_csv(explained_path, index=False)

    try:
        n_save = int(n_pca_components_to_save)
    except Exception:
        n_save = 0
    if n_save <= 0:
        n_save = k_selected
    n_save = int(max(1, min(n_save, k_selected)))
    coords_df = df_all_groups[["individual", "group", "sex"]].copy()
    for i in range(n_save):
        coords_df[f"PC{i + 1}"] = X_all_pca[:, i]
    coords_path = os.path.join(pca_dir, "PCA_PROJECTED_COORDINATES.csv")
    coords_df.to_csv(coords_path, index=False)

    scores = _compute_axis_scores(
        X_all_pca,
        df_all_groups["group"].values,
        training_group_a=str(training_group_a),
        training_group_b=str(training_group_b),
    )
    if scores is None:
        groups_present = sorted(set(str(g) for g in df_all_groups["group"].dropna().unique().tolist()))
        raise ValueError(
            "Could not compute PCA axis scores. One or both training groups may be missing, "
            "or their PCA-space centroids may be identical. "
            f"training_group_a={training_group_a!r}; training_group_b={training_group_b!r}; "
            f"groups_present={groups_present[:50]}"
        )

    axis_df = df_all_groups[["individual", "group", "sex"]].copy()
    axis_df["axis_t"] = scores["t"]
    axis_df["axis_r"] = scores["r"]
    axis_df["axis_r_norm"] = scores["r_norm"]
    axis_df["axis_training_sep"] = float(scores["training_sep"])
    axis_df["axis_pca_k"] = int(k_selected)
    axis_scores_path = os.path.join(pca_dir, "PCA_AXIS_SCORES.csv")
    axis_df.to_csv(axis_scores_path, index=False)

    summary_df = _summarize_axis_scores_by_group(axis_df)
    summary_path = os.path.join(pca_dir, "PCA_AXIS_GROUP_SUMMARY.csv")
    summary_df.to_csv(summary_path, index=False)

    reference_path = os.path.join(pca_dir, "PCA_AXIS_REFERENCE.json")
    _json_dump(
        reference_path,
        {
            "training_group_a": str(training_group_a),
            "training_group_b": str(training_group_b),
            "axis_t_definition": f"0={training_group_b}, 1={training_group_a}",
            "axis_training_sep": float(scores["training_sep"]),
            "axis_pca_k": int(k_selected),
            "mu_a": np.asarray(scores["mu_a"], dtype=float).tolist(),
            "mu_b": np.asarray(scores["mu_b"], dtype=float).tolist(),
            "axis_vector": np.asarray(scores["axis_vector"], dtype=float).tolist(),
            "explained_variance_ratio": explained_var_ratio.tolist(),
            "feature_count": int(len(feature_names_list)),
            "features": list(feature_names_list),
        },
    )

    manifest_path = os.path.join(pca_dir, "MANIFEST_PCA.json")
    write_stage_manifest(
        manifest_path,
        stage="pca",
        analysis_folder=analysis_folder,
        output_dir=pca_dir,
        inputs={
            "normalized_matrix_csv": "NORMALIZED_FEATURE_MATRIX.csv",
            "selected_feature_count": int(len(feature_names_list)),
        },
        settings={
            "training_group_a": str(training_group_a),
            "training_group_b": str(training_group_b),
            "pca_fixed_k": int(effective_fixed_k),
            "n_pca_components_saved": int(n_save),
            "random_seed": int(GLOBAL_RANDOM_SEED) if GLOBAL_RANDOM_SEED is not None else None,
        },
        outputs={
            "explained_variance_csv": os.path.basename(explained_path),
            "projected_coordinates_csv": os.path.basename(coords_path),
            "pca_axis_scores_csv": os.path.basename(axis_scores_path),
            "pca_axis_group_summary_csv": os.path.basename(summary_path),
            "pca_axis_reference_json": os.path.basename(reference_path),
        },
        summary={
            "n_samples": int(len(axis_df)),
            "pca_k": int(k_selected),
            "axis_training_sep": float(scores["training_sep"]),
        },
    )

    artifact_paths = [
        explained_path,
        coords_path,
        axis_scores_path,
        summary_path,
        reference_path,
        manifest_path,
    ]

    print(f"PCA outputs saved to: {pca_dir}")
    print(f"  axis scores: {axis_scores_path}")
    print(f"  axis summary: {summary_path}")
    print("=" * 80 + "\n")
    return {
        "output_dir": pca_dir,
        "artifact_paths": artifact_paths,
        "summary": "PCA reference-axis analysis completed.",
    }


__all__ = [
    "run_pca_analysis",
    "_compute_axis_scores",
    "_summarize_axis_scores_by_group",
]
