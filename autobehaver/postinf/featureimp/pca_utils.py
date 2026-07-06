"""PCA utilities for the retained FeatureImp PCA reference-axis analysis."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import hashlib
import json
import os

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def _feature_name_hash(X: Any) -> str:
    if isinstance(X, pd.DataFrame):
        try:
            payload = "\n".join(str(c) for c in X.columns).encode("utf-8")
            return hashlib.sha1(payload).hexdigest()
        except Exception:
            return "unknown"
    return "no_columns"


def _as_float32_matrix(X: Any) -> np.ndarray:
    if isinstance(X, pd.DataFrame):
        arr = X.to_numpy(dtype=np.float32, copy=True)
    else:
        arr = np.asarray(X, dtype=np.float32)
    if np.isnan(arr).any():
        arr = np.nan_to_num(arr, nan=0.0)
    return arr


def _pca_disk_cache_path(analysis_folder: str, cache_key: str, meta: Dict[str, Any]) -> str:
    cache_dir = os.path.join(str(analysis_folder or "."), ".pca_cache")
    try:
        os.makedirs(cache_dir, exist_ok=True)
    except Exception:
        return ""
    meta_json = json.dumps(meta, sort_keys=True, default=str)
    h = hashlib.sha1(meta_json.encode("utf-8")).hexdigest()[:16]
    safe_key = "".join(c if c.isalnum() or c in ("_", "-") else "_" for c in str(cache_key or "pca"))
    return os.path.join(cache_dir, f"pca_model_{safe_key}_{h}.joblib")


def _joblib_load(path: str) -> Any:
    try:
        import joblib  # type: ignore
        return joblib.load(path)
    except Exception:
        return None


def _joblib_dump(obj: Any, path: str) -> bool:
    try:
        import joblib  # type: ignore
        joblib.dump(obj, path)
        return True
    except Exception:
        return False


def _fit_project_reference_pca(
    X_ref_array: np.ndarray,
    X_all_array: np.ndarray,
    *,
    pca_fixed_k: Optional[int],
    random_state: Optional[int],
) -> Tuple[StandardScaler, PCA, np.ndarray, np.ndarray, int, np.ndarray]:
    """Fit StandardScaler+fixed-k PCA on reference rows and project all rows."""

    scaler = StandardScaler()
    X_ref_scaled = scaler.fit_transform(X_ref_array)
    X_all_scaled = scaler.transform(X_all_array)

    n_ref_samples, n_features = X_ref_scaled.shape
    max_possible = int(min(n_ref_samples, n_features))
    if max_possible <= 0:
        raise ValueError("PCA requires at least one reference sample and one feature.")

    try:
        requested_k = int(pca_fixed_k) if pca_fixed_k is not None else 50
    except Exception:
        requested_k = 50
    k_selected = int(max(1, min(requested_k, max_possible)))

    pca = PCA(n_components=k_selected, random_state=random_state)
    X_ref_pca = pca.fit_transform(X_ref_scaled)
    X_all_pca = pca.transform(X_all_scaled)
    explained_var_ratio = np.asarray(pca.explained_variance_ratio_, dtype=float)
    return scaler, pca, X_ref_pca, X_all_pca, k_selected, explained_var_ratio


def get_cached_pca_projection(
    X_ref: Any,
    X_all: Any,
    *,
    paths: Dict[str, Any],
    cache_key: str,
    pca_fixed_k: Optional[int],
    random_state: Optional[int],
) -> Tuple[StandardScaler, PCA, np.ndarray, np.ndarray, int, np.ndarray]:
    """Compute or reuse the fixed-k reference PCA projection."""

    analysis_folder = str(paths.get("analysis_folder", "") or "")
    try:
        fixed_k_i = int(pca_fixed_k) if pca_fixed_k is not None else 50
    except Exception:
        fixed_k_i = 50

    rs = None
    if random_state is not None:
        try:
            rs = int(random_state)
        except Exception:
            rs = None

    try:
        n_ref = int(getattr(X_ref, "shape")[0])
        n_all = int(getattr(X_all, "shape")[0])
        n_feat = int(getattr(X_ref, "shape")[1])
    except Exception:
        n_ref = n_all = n_feat = -1

    meta = {
        "analysis_folder": analysis_folder,
        "cache_key": str(cache_key or "pca"),
        "pca_fixed_k": fixed_k_i,
        "random_state": rs,
        "n_ref": n_ref,
        "n_all": n_all,
        "n_feat": n_feat,
        "feature_name_hash": _feature_name_hash(X_ref),
    }
    key = tuple(sorted(meta.items()))
    cache = paths.setdefault("_pca_cache", {})
    if key in cache:
        return cache[key]

    disk_path = _pca_disk_cache_path(analysis_folder, str(cache_key or "pca"), meta)
    if disk_path and os.path.exists(disk_path):
        payload = _joblib_load(disk_path)
        if isinstance(payload, dict) and payload.get("meta") == meta:
            scaler = payload.get("scaler")
            pca = payload.get("pca")
            k_selected = payload.get("k_selected")
            if scaler is not None and pca is not None and k_selected is not None:
                try:
                    k_i = int(k_selected)
                    X_ref_arr = _as_float32_matrix(X_ref)
                    X_all_arr = _as_float32_matrix(X_all)
                    X_ref_pca_full = pca.transform(scaler.transform(X_ref_arr))
                    X_all_pca_full = pca.transform(scaler.transform(X_all_arr))
                    result = (
                        scaler,
                        pca,
                        X_ref_pca_full[:, :k_i],
                        X_all_pca_full[:, :k_i],
                        k_i,
                        np.asarray(payload.get("explained_var_ratio"), dtype=float)[:k_i],
                    )
                    cache[key] = result
                    return result
                except Exception:
                    pass

    result = _fit_project_reference_pca(
        _as_float32_matrix(X_ref),
        _as_float32_matrix(X_all),
        pca_fixed_k=fixed_k_i,
        random_state=rs,
    )
    cache[key] = result
    if disk_path:
        try:
            scaler, pca, _X_ref_pca, _X_all_pca, k_selected, explained_var_ratio = result
            _joblib_dump(
                {
                    "meta": meta,
                    "scaler": scaler,
                    "pca": pca,
                    "k_selected": int(k_selected),
                    "explained_var_ratio": np.asarray(explained_var_ratio, dtype=float),
                },
                disk_path,
            )
        except Exception:
            pass
    return result
