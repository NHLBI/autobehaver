"""FeatureImp feature-importance and postanalysis entry points.

Public functions load lazily so package imports and command-line help do not require
optional analysis dependencies.
"""

from __future__ import annotations

from typing import Any

__all__ = ["run_feature_importance_analysis", "plan_feature_importance_analysis", "process_featureimp_shard", "merge_featureimp_shards", "plan_plain_xgboost_baseline_shards", "process_plain_xgboost_baseline_shard", "merge_plain_xgboost_baseline_shards", "run_featureimp_postanalysis_stage", "run_featureimp_global_fsea_from_file", "normalize_feature_matrix", "load_normalization_params"]


def __getattr__(name: str) -> Any:
    """Resolve public functions lazily so package imports do not load optional analysis
    dependencies.
    """

    if name in {"run_feature_importance_analysis", "plan_feature_importance_analysis", "process_featureimp_shard", "merge_featureimp_shards", "plan_plain_xgboost_baseline_shards", "process_plain_xgboost_baseline_shard", "merge_plain_xgboost_baseline_shards", "run_featureimp_postanalysis_stage", "run_featureimp_global_fsea_from_file", "normalize_feature_matrix"}:
        from . import core  # Load optional analysis dependencies only when this API is used.

        return getattr(core, name)

    if name in {"load_normalization_params"}:
        from . import matrix

        return getattr(matrix, name)

    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


def __dir__() -> list[str]:
    return sorted(set(list(globals().keys()) + __all__))
