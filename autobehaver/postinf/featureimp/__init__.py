"""Feature-importance (post-inference) analysis.

This package contains both:

* **Lightweight** utilities and post-analysis runners (FSEA, PCA, etc.), and
* A **heavy** end-to-end feature importance pipeline (xgboost/shap/etc.).

Important
---------
We intentionally avoid importing the heavy pipeline at package import time so
that lightweight runners and ``--help`` remain fast and do not require optional
ML dependencies.

The primary public entrypoints are exposed lazily via ``__getattr__``.
"""

from __future__ import annotations

from typing import Any

__all__ = ["run_feature_importance_analysis", "plan_feature_importance_analysis", "process_featureimp_shard", "merge_featureimp_shards", "plan_plain_xgboost_baseline_shards", "process_plain_xgboost_baseline_shard", "merge_plain_xgboost_baseline_shards", "run_featureimp_postanalysis_stage", "run_featureimp_global_fsea_from_file", "normalize_feature_matrix", "load_normalization_params"]


def __getattr__(name: str) -> Any:
    """Lazily resolve public API attributes.

    This keeps package import lightweight while preserving the historical
    convenience import:

        from autobehaver.postinf.featureimp import run_feature_importance_analysis

    Notes
    -----
    Accessing these attributes may import heavy optional dependencies.
    """

    if name in {"run_feature_importance_analysis", "plan_feature_importance_analysis", "process_featureimp_shard", "merge_featureimp_shards", "plan_plain_xgboost_baseline_shards", "process_plain_xgboost_baseline_shard", "merge_plain_xgboost_baseline_shards", "run_featureimp_postanalysis_stage", "run_featureimp_global_fsea_from_file", "normalize_feature_matrix"}:
        from . import core  # local import by design (may be heavy)

        return getattr(core, name)

    if name in {"load_normalization_params"}:
        from . import matrix

        return getattr(matrix, name)

    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


def __dir__() -> list[str]:
    return sorted(set(list(globals().keys()) + __all__))
