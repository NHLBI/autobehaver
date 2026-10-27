"""Weight FeatureImp subsets by out-of-fold log-loss improvement over a fold-aware
constant baseline.

When available, permuted subset skills set the dead-zone threshold; incomplete subsets
receive zero weight.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from .integrity import strict_bool
from autobehaver.atomic_io import atomic_write_dataframe_csv, atomic_write_json


@dataclass(frozen=True)
class SubsetWeightingConfig:
    """Configuration for subset weighting."""

    tau_quantile: float = 0.95
    gamma: float = 2.0
    metric_col: str = "delta_norm"  # normalized log-loss improvement
    subset_complete_col: str = "subset_complete"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _effective_ensemble_size(weights: np.ndarray) -> float:
    """Return effective ensemble size as (sum w)^2 / sum(w^2)."""

    w = np.asarray(weights, dtype=np.float64)
    w = w[np.isfinite(w)]
    w = w[w > 0]
    if w.size == 0:
        return 0.0
    s1 = float(np.sum(w))
    s2 = float(np.sum(w * w))
    if s2 <= 0:
        return 0.0
    return (s1 * s1) / s2


def _load_subset_performance_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Subset performance CSV not found: {path}")
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"Subset performance CSV is empty: {path}")
    return df


def compute_tau_from_permuted(
    permuted_df: pd.DataFrame,
    cfg: SubsetWeightingConfig,
) -> Tuple[float, Dict[str, Any]]:
    """Compute tau (dead-zone threshold) from permuted subset performance.

    Returns
    -------
    (tau, stats)
    """

    metric = cfg.metric_col
    complete_col = cfg.subset_complete_col

    if metric not in permuted_df.columns:
        raise KeyError(f"Permuted subset performance missing column: {metric}")

    df = permuted_df.copy()
    if complete_col in df.columns:
        df = df[df[complete_col].map(strict_bool)]

    vals = pd.to_numeric(df[metric], errors="coerce").to_numpy(dtype=np.float64)
    vals = vals[np.isfinite(vals)]

    if vals.size == 0:
        raise ValueError("Requested permutation threshold has no finite complete metrics")

    q = float(cfg.tau_quantile)
    if not (0.0 < q < 1.0):
        raise ValueError(f"tau_quantile must be between 0 and 1 (exclusive). Got: {q}")

    tau_raw = float(np.quantile(vals, q))
    tau = max(0.0, tau_raw)

    stats = {
        "tau_source": "permutation_quantile",
        "tau_quantile": q,
        "tau_raw": tau_raw,
        "tau": tau,
        "n_values": int(vals.size),
        "perm_metric_min": float(np.min(vals)),
        "perm_metric_median": float(np.median(vals)),
        "perm_metric_p95": float(np.quantile(vals, 0.95)),
        "perm_metric_max": float(np.max(vals)),
    }
    return tau, stats


def compute_subset_weights(
    real_df: pd.DataFrame,
    tau: float,
    cfg: SubsetWeightingConfig,
) -> pd.DataFrame:
    """Compute subset weights for the real run.

    Weight function:
        w = max(0, delta_norm - tau) ** gamma

    Incomplete subsets or missing metrics receive weight 0.
    """

    metric = cfg.metric_col
    complete_col = cfg.subset_complete_col
    gamma = float(cfg.gamma)
    if not np.isfinite(gamma) or gamma <= 0 or not np.isfinite(tau) or tau < 0:
        raise ValueError("gamma must be positive finite and tau nonnegative finite")

    if metric not in real_df.columns:
        raise KeyError(f"Real subset performance missing column: {metric}")

    df = real_df.copy()

    if complete_col not in df.columns:
        df[complete_col] = True

    df[complete_col] = df[complete_col].map(strict_bool)
    is_complete = df[complete_col].to_numpy(dtype=bool)
    identity = [c for c in ('permutation_idx', 'h_hit', 'subset_idx') if c in df.columns]
    if identity and df.duplicated(subset=identity).any():
        raise ValueError("Duplicate subset identities in performance input")
    vals = pd.to_numeric(df[metric], errors="coerce").to_numpy(dtype=np.float64)
    vals_ok = np.isfinite(vals)

    w = np.zeros(len(df), dtype=np.float64)
    idx = np.where(is_complete & vals_ok)[0]
    if idx.size > 0:
        base = np.maximum(0.0, vals[idx] - float(tau))
        w[idx] = np.power(base, gamma)

    df["tau_used"] = float(tau)
    df["gamma_used"] = float(gamma)
    df["subset_weight"] = w
    df["subset_weight_nonzero"] = df["subset_weight"] > 0
    df["subset_weight_reason"] = np.where(
        ~is_complete,
        "incomplete_subset",
        np.where(~vals_ok, "missing_metric", np.where(w > 0, "ok", "below_tau")),
    )

    return df


def compute_and_write_subset_weights(
    *,
    analysis_folder: str,
    permuted_base_folder: Optional[str],
    cfg: SubsetWeightingConfig,
    verbose: bool = True,
) -> Optional[Dict[str, Any]]:
    """Calculate subset weights and publish their table and manifest.

    Return the manifest; raise if real metrics or explicitly requested permutation
    metrics are missing.
    """

    analysis_folder = str(analysis_folder or ".")
    real_perf_path = os.path.join(analysis_folder, "COMMITTEE_SUBSETS", "SUBSET_PERFORMANCE_REAL.csv")
    if not os.path.exists(real_perf_path):
        raise FileNotFoundError(real_perf_path)

    real_df = _load_subset_performance_csv(real_perf_path)

    tau = 0.0
    tau_stats: Dict[str, Any] = {
        "tau_source": "default_zero",
        "tau_quantile": cfg.tau_quantile,
        "tau": 0.0,
        "note": "No permutation subset performance found; tau set to 0.",
    }

    perm_perf_path = None
    if permuted_base_folder:
        permuted_base_folder = str(permuted_base_folder)
        candidate = os.path.join(permuted_base_folder, "COMMITTEE_SUBSETS", "SUBSET_PERFORMANCE_PERMUTED_ALL.csv")
        if os.path.exists(candidate):
            perm_perf_path = candidate
            perm_df = _load_subset_performance_csv(candidate)
            tau, tau_stats = compute_tau_from_permuted(perm_df, cfg)
        else:
            raise FileNotFoundError(f"Requested permutation metrics missing: {candidate}")

    weights_df = compute_subset_weights(real_df, tau=tau, cfg=cfg)

    out_dir = os.path.join(analysis_folder, "COMMITTEE_SUBSETS")
    os.makedirs(out_dir, exist_ok=True)
    weights_path = os.path.join(out_dir, "SUBSET_WEIGHTS_REAL.csv")
    atomic_write_dataframe_csv(weights_df, weights_path, index=False)

    w = pd.to_numeric(weights_df["subset_weight"], errors="coerce").to_numpy(dtype=np.float64)
    w = w[np.isfinite(w)]
    w_pos = w[w > 0]

    summary = {
        "n_subsets_total": int(len(weights_df)),
        "n_subsets_complete": int(np.sum(weights_df.get(cfg.subset_complete_col, True).astype(bool))),
        "n_nonzero_weights": int(w_pos.size),
        "frac_nonzero_weights": float(w_pos.size / len(weights_df)) if len(weights_df) else 0.0,
        "sum_weights": float(np.sum(w_pos)) if w_pos.size else 0.0,
        "mean_weight_nonzero": float(np.mean(w_pos)) if w_pos.size else 0.0,
        "median_weight_nonzero": float(np.median(w_pos)) if w_pos.size else 0.0,
        "max_weight": float(np.max(w_pos)) if w_pos.size else 0.0,
        "effective_ensemble_size": float(_effective_ensemble_size(w_pos)),
    }

    manifest: Dict[str, Any] = {
        "manifest_version": 1,
        "stage": "subset_weighting_phase2",
        "created_at": _utc_now_iso(),
        "analysis_folder": analysis_folder,
        "inputs": {
            "real_subset_performance_csv": real_perf_path,
            "permuted_subset_performance_csv": perm_perf_path,
        },
        "settings": {
            "metric_col": cfg.metric_col,
            "subset_complete_col": cfg.subset_complete_col,
            "tau_quantile": float(cfg.tau_quantile),
            "gamma": float(cfg.gamma),
            "tau": float(tau),
            **{k: v for k, v in tau_stats.items() if k not in {"tau"}},
        },
        "outputs": {
            "subset_weights_csv": weights_path,
        },
        "summary": summary,
    }

    manifest_path = os.path.join(out_dir, "WEIGHTING_MANIFEST.json")
    atomic_write_json(manifest, manifest_path)

    if verbose:
        print("\n" + "=" * 80)
        print("SUBSET WEIGHTING")
        print("=" * 80)
        print(f"Real subset performance: {real_perf_path}")
        if perm_perf_path:
            print(f"Permuted subset performance: {perm_perf_path}")
        print(f"Weight metric: {cfg.metric_col}")
        print(f"Tau source: {tau_stats.get('tau_source')}, tau={tau:.6g} (quantile={cfg.tau_quantile})")
        print(f"Gamma: {cfg.gamma}")
        print(f"Wrote subset weights: {weights_path}")
        print(f"Wrote manifest: {manifest_path}")
        print(
            f"Nonzero weights: {summary['n_nonzero_weights']}/{summary['n_subsets_total']} "
            f"({summary['frac_nonzero_weights']*100:.2f}%) | "
            f"N_eff={summary['effective_ensemble_size']:.2f}"
        )

    return manifest
