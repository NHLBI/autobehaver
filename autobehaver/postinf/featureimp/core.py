"""Public FeatureImp settings and workflow entry points with lazy stage imports.

Single-job execution uses the same plan, process, and merge engine as sharded execution;
optional FSEA and PCA run after merging.
"""

from __future__ import annotations

import json
import os
from autobehaver.portable_paths import generated_path
import warnings
from datetime import datetime
from typing import Any, Dict, Iterable, Mapping, Optional, Set

from .config import DEFAULT_SETTINGS

# Keep track of the most recently applied *fully merged* settings. Staged
# helpers consult this snapshot when a caller omits explicit overrides.
_LAST_SETTINGS: Dict[str, Any] = dict(DEFAULT_SETTINGS)

# Keep track of the most recently applied *explicit* settings so standalone
# postanalysis stages can distinguish user-supplied overrides from merged
# defaults. This avoids a later stage accidentally overwriting saved analysis
# context just because a default happens to be present in the active config.
_LAST_EXPLICIT_SETTINGS: Dict[str, Any] = {}
_LAST_EXPLICIT_SETTING_KEYS: Set[str] = set()

# The saved run-settings snapshot stores the analysis-defining FeatureImp
# settings plus retained reference postanalysis controls needed to complete a
# sharded run later without repeating the full config. Runtime machine knobs stay
# out of this file on purpose.
FEATUREIMP_RUN_SETTINGS_FROZEN_KEYS: tuple[str, ...] = (
    "control_group_tag",
    "training_group_a",
    "training_group_b",
    "include_feature_categories",
    "sex_filter",
    "min_feature_completeness_per_group",
    "MIN_VARIANCE_THRESHOLD",
    "model_subsample",
    "n_estimators",
    "learning_rate",
    "max_depth",
    "n_splits",
    "H_hits_per_feature",
    "n_features_per_model",
    "GLOBAL_RANDOM_SEED",
    "run_permutation_test",
    "n_permutations",
    "subset_weight_tau_quantile",
    "subset_weight_gamma",
)

FEATUREIMP_RUN_SETTINGS_CONTEXT_KEYS: tuple[str, ...] = (
    "analysis_run_name",
    "main_dir",
    "matrix_dir",
    "results_dir",
    "matrix_path",
)

# Lightweight postanalyses (FSEA + PCA) are normally
# launched after merge_shards, often in a separate process from stage=plan. Keep
# their settings in the snapshot so a planned sharded run remains reproducible
# without requiring the user to repeat the same config at merge time. Explicit
# merge-time CLI/config values still override these saved values.
FEATUREIMP_RUN_SETTINGS_POSTANALYSIS_KEYS: tuple[str, ...] = (
    "run_fsea_analysis",
    "run_pca_analysis",
    "postanalysis_strict",
    "postanalysis_debug",
    "pca_fixed_k",
    "n_pca_components_to_save",
    "fsea_permutations",
    "fsea_min_size",
    "fsea_max_size",
    "fsea_fdr_threshold",
    "fsea_viz_top_n",
    "fsea_viz_plot_all_groups",
    "fsea_viz_color_percentile",
)

FEATUREIMP_RUN_SETTINGS_METADATA_KEYS: tuple[str, ...] = (
    "__analysis_folder__",
    "__analysis_folder_name__",
    "__settings_scope__",
    "__timestamp_tag__",
    "__autobehaver_version__",
)

FEATUREIMP_RUN_SETTINGS_ALLOWED_KEYS: tuple[str, ...] = tuple(
    dict.fromkeys(
        FEATUREIMP_RUN_SETTINGS_FROZEN_KEYS
        + FEATUREIMP_RUN_SETTINGS_CONTEXT_KEYS
        + FEATUREIMP_RUN_SETTINGS_POSTANALYSIS_KEYS
    )
)


def filter_saved_settings_snapshot(payload: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    """Return only the allowed FEATUREIMP_RUN_SETTINGS.json fields."""

    if not isinstance(payload, Mapping):
        return {}

    filtered: Dict[str, Any] = {}
    for key in FEATUREIMP_RUN_SETTINGS_METADATA_KEYS:
        if key in payload:
            filtered[key] = payload[key]
    for key in FEATUREIMP_RUN_SETTINGS_ALLOWED_KEYS:
        if key in payload:
            filtered[key] = payload[key]
    return filtered



def allocate_analysis_folder(results_dir: str, name: str) -> str:
    """Exclusively reserve a portable run folder without overwriting another run."""
    os.makedirs(results_dir, exist_ok=True)
    for attempt in range(10000):
        candidate_name = name if attempt == 0 else f"{name}_{attempt:03d}"
        candidate = str(generated_path(results_dir, candidate_name, reserve=100))
        try:
            os.mkdir(candidate)
        except FileExistsError:
            continue
        return candidate
    raise RuntimeError("Could not allocate a unique FeatureImp analysis folder")

def build_settings_snapshot_payload(*, analysis_folder_path: str = "", analysis_folder_name: str = "", timestamp_tag: str = "", settings_map: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    """Build the persisted FEATUREIMP_RUN_SETTINGS.json payload.

    The staged FeatureImp workflow treats the snapshot as a curated record of
    analysis-defining committee settings, light context, and the retained
    reference postanalysis controls (FSEA/PCA) needed after merge_shards.
    Runtime machine knobs and unrelated defaults are intentionally excluded.
    """

    payload: Dict[str, Any] = {
        "__analysis_folder__": str(analysis_folder_path),
        "__analysis_folder_name__": str(analysis_folder_name or os.path.basename(str(analysis_folder_path).rstrip(os.sep))),
        "__settings_scope__": "analysis_folder_defining",
    }
    if timestamp_tag:
        payload["__timestamp_tag__"] = str(timestamp_tag)
    try:
        from autobehaver import __version__ as _ab_version
        payload["__autobehaver_version__"] = str(_ab_version)
    except Exception:
        payload["__autobehaver_version__"] = "unknown"

    merged: Dict[str, Any] = dict(DEFAULT_SETTINGS)
    merged.update(dict(settings_map or _LAST_SETTINGS))
    merged["include_feature_categories"] = _normalize_include_feature_categories(
        merged.get("include_feature_categories")
    )
    for key in FEATUREIMP_RUN_SETTINGS_ALLOWED_KEYS:
        payload[key] = merged.get(key, DEFAULT_SETTINGS.get(key))
    return payload


def _normalize_include_feature_categories(val: Any) -> Any:
    """Normalize include_feature_categories into canonical form.

    Special values (case-insensitive): NONE/ALL/*/NULL -> None (use the default recognized feature categories).
    An empty list is treated as None.
    """

    if val is None:
        return None

    if isinstance(val, str):
        tokens = [val]
    else:
        try:
            tokens = list(val)
        except Exception:
            tokens = [str(val)]

    if len(tokens) == 0:
        return None

    if len(tokens) == 1:
        t0 = str(tokens[0]).strip().lower()
        if t0 in ("none", "all", "*", "null"):
            return None

    return tokens


# Settings


def apply_settings(settings_dict: Dict[str, Any], *, propagate: bool = True, explicit_keys: Optional[Iterable[str]] = None) -> None:
    """Apply a settings dictionary to this facade module."""

    global _LAST_SETTINGS, _LAST_EXPLICIT_SETTINGS, _LAST_EXPLICIT_SETTING_KEYS

    # Merge user settings on top of defaults so downstream modules always
    # receive a complete config.
    merged: Dict[str, Any] = dict(DEFAULT_SETTINGS)
    merged.update(settings_dict)
    from .integrity import strict_bool
    for key, default in DEFAULT_SETTINGS.items():
        if isinstance(default, bool):
            merged[key] = strict_bool(merged[key])
    import math
    for key in ('min_feature_completeness_per_group', 'MIN_VARIANCE_THRESHOLD', 'subset_weight_gamma', 'subset_weight_tau_quantile', 'fsea_fdr_threshold'):
        if merged.get(key) is not None and not math.isfinite(float(merged[key])):
            raise ValueError(f"{key} must be finite")
    for key in ('min_feature_completeness_per_group', 'fsea_fdr_threshold'):
        if not 0 <= float(merged[key]) <= 1:
            raise ValueError(f"{key} must be in [0,1]")
    if float(merged['MIN_VARIANCE_THRESHOLD']) < 0 or float(merged['subset_weight_gamma']) <= 0 or not 0 < float(merged['subset_weight_tau_quantile']) < 1:
        raise ValueError("Invalid variance/weighting settings")


    # Normalize include_feature_categories tokens like NONE/ALL/* into None so
    # callers can request all recognized feature categories from the API or CLI.
    merged["include_feature_categories"] = _normalize_include_feature_categories(
        merged.get("include_feature_categories")
    )

    # Path semantics (FeatureImp)
    #   - matrix_dir  : where MASTER_FEATURE_MATRIX.csv lives (MMGen output folder)
    #   - results_dir : where FeatureImp run folders are written
    matrix_dir = str(merged.get("matrix_dir") or "").strip()
    results_dir = str(merged.get("results_dir") or "").strip()

    # Auto-derive results_dir when not provided.
    if not results_dir:
        if matrix_dir:
            parent = os.path.dirname(os.path.normpath(matrix_dir))
            if parent == "":
                parent = "."
            results_dir = os.path.join(parent, "9_feature_importance")
        else:
            results_dir = "9_feature_importance"
        merged["results_dir"] = results_dir

    merged["results_dir"] = results_dir

    # Auto-derive matrix_path if not provided.
    matrix_path = str(merged.get("matrix_path") or "").strip()
    if not matrix_path:
        matrix_path = os.path.join(matrix_dir, "MASTER_FEATURE_MATRIX.csv") if matrix_dir else "MASTER_FEATURE_MATRIX.csv"
        merged["matrix_path"] = matrix_path

    _LAST_SETTINGS = merged

    explicit_key_set = {str(k) for k in (settings_dict.keys() if explicit_keys is None else explicit_keys) if str(k)}
    _LAST_EXPLICIT_SETTING_KEYS = explicit_key_set
    _LAST_EXPLICIT_SETTINGS = {k: merged.get(k) for k in explicit_key_set if k in merged}

    # Apply each resolved runtime default to the public settings facade.
    for key in DEFAULT_SETTINGS.keys():
        globals()[key] = merged.get(key)

    if propagate:
        pass


# Initialize defaults immediately (module-level globals)
#
# Keep optional model dependencies out of package initialization.
# We only initialize this module's globals here; propagation to heavy
# submodules happens when the CLI (or caller) explicitly applies settings.
apply_settings(DEFAULT_SETTINGS, propagate=False, explicit_keys=())


def get_current_settings() -> Dict[str, Any]:
    """Return a shallow copy of the active FeatureImp runtime settings."""

    return dict(_LAST_SETTINGS)


def get_explicit_settings() -> Dict[str, Any]:
    """Return the most recently supplied explicit FeatureImp settings."""

    return dict(_LAST_EXPLICIT_SETTINGS)


def get_explicit_setting_keys() -> Set[str]:
    """Return the keys explicitly supplied in the latest settings apply."""

    return set(_LAST_EXPLICIT_SETTING_KEYS)


# Warnings / logging helpers


def apply_warning_filters() -> None:
    """Apply global warning filters based on settings.

    This is called early by the CLI.
    """

    if globals().get("SUPPRESS_RUNTIME_WARNINGS", True):
        warnings.filterwarnings("ignore", category=RuntimeWarning)

    if globals().get("SUPPRESS_FUTURE_WARNINGS", True):
        warnings.filterwarnings("ignore", category=FutureWarning)

    # Suppress configured layout warnings in batch logs.
    if globals().get("SUPPRESS_TIGHT_LAYOUT_WARNINGS", True):
        warnings.filterwarnings(
            "ignore",
            message=r"Tight layout not applied.*",
            category=UserWarning,
        )


# Small public helpers


def get_analysis_folder_name(analysis_run_name: str, sex_stratum: str | None = None) -> str:
    """Return standardized analysis folder name for a run.

    Parameters
    ----------
    analysis_run_name:
        User-provided run name.
    sex_stratum:
        Optional sex stratum label. Supported values are defined by
        :func:`autobehaver.postinf.featureimp.prep.get_analysis_folder_name`.
        When None (default), this returns the standard pooled-sex folder name.
    """

    # Avoid importing pandas-heavy prep module at import time.
    from .prep import get_analysis_folder_name as _get

    return _get(analysis_run_name=analysis_run_name, sex_stratum=sex_stratum)


def normalize_feature_matrix(
    results_dir: str,
    raw_matrix_path: str,
    analysis_folder_path: str,
    *,
    skip_if_normalized_exists: bool = True,
) -> Optional[str]:
    """Normalize the raw feature matrix and save NORMALIZED_FEATURE_MATRIX.csv.

    Returns
    -------
    str | None
        Path to the normalized matrix, or None if normalization was skipped/failed.
    """

    from .matrix import normalize_feature_matrix as _normalize

    return _normalize(
        raw_matrix_path=raw_matrix_path,
        results_dir=results_dir,
        analysis_folder_path=analysis_folder_path,
        control_group_tag=globals().get("control_group_tag"),
        training_group_a=globals().get("training_group_a"),
        training_group_b=globals().get("training_group_b"),
        skip_if_normalized_exists=skip_if_normalized_exists,
    )


# Staged execution helpers


def plan_feature_importance_analysis(*args: Any, **kwargs: Any):
    """Create a frozen staged/sharded execution plan for FeatureImp.

    This lightweight facade imports :mod:`featureimp.plan` only when needed, so
    the planner can run without importing the heavy XGBoost/SHAP pipeline.
    """

    from . import plan as _plan

    for key in ("analysis_run_name", "include_feature_categories", "control_group_tag",
                "training_group_a", "training_group_b", "n_splits",
                "H_hits_per_feature", "n_features_per_model"):
        kwargs.setdefault(key, _LAST_SETTINGS.get(key, DEFAULT_SETTINGS.get(key)))
    kwargs.setdefault("sex_filter", _LAST_SETTINGS.get("sex_filter", "all"))
    kwargs.setdefault(
        "min_feature_completeness_per_group",
        _LAST_SETTINGS.get("min_feature_completeness_per_group", DEFAULT_SETTINGS.get("min_feature_completeness_per_group")),
    )
    kwargs.setdefault(
        "min_variance_threshold",
        _LAST_SETTINGS.get("MIN_VARIANCE_THRESHOLD", DEFAULT_SETTINGS.get("MIN_VARIANCE_THRESHOLD")),
    )
    kwargs.setdefault(
        "run_permutation_test",
        _LAST_SETTINGS.get("run_permutation_test", DEFAULT_SETTINGS.get("run_permutation_test")),
    )
    kwargs.setdefault(
        "n_permutations",
        _LAST_SETTINGS.get("n_permutations", DEFAULT_SETTINGS.get("n_permutations")),
    )
    kwargs.setdefault(
        "num_shards",
        _LAST_SETTINGS.get("num_shards", DEFAULT_SETTINGS.get("num_shards", 1)),
    )
    kwargs.setdefault(
        "GLOBAL_RANDOM_SEED",
        _LAST_SETTINGS.get("GLOBAL_RANDOM_SEED", DEFAULT_SETTINGS.get("GLOBAL_RANDOM_SEED")),
    )

    return _plan.plan_featureimp_run(*args, **kwargs)



def process_featureimp_shard(*args: Any, **kwargs: Any):
    """Run one staged/sharded FeatureImp processing shard."""

    from . import shard_runner as _shard_runner

    kwargs.setdefault(
        "shard_index",
        _LAST_SETTINGS.get("shard_index", DEFAULT_SETTINGS.get("shard_index", 0)),
    )
    return _shard_runner.process_featureimp_shard_run(*args, **kwargs)



def merge_featureimp_shards(*args: Any, **kwargs: Any):
    """Merge completed training shards and run enabled FSEA and PCA postanalyses."""

    from . import merge_shards as _merge_shards

    return _merge_shards.merge_featureimp_shards_run(*args, **kwargs)



def plan_plain_xgboost_baseline_shards(*args: Any, **kwargs: Any):
    """Create a frozen shard plan for the full-feature XGBoost comparison."""
    from . import plain_xgboost_staged as _plain_xgboost_staged

    return _plain_xgboost_staged.plan_plain_xgboost_baseline_shards(*args, **kwargs)


def process_plain_xgboost_baseline_shard(*args: Any, **kwargs: Any):
    """Process one planned full-feature XGBoost comparison shard."""
    from . import plain_xgboost_staged as _plain_xgboost_staged

    return _plain_xgboost_staged.process_plain_xgboost_baseline_shard(*args, **kwargs)


def merge_plain_xgboost_baseline_shards(*args: Any, **kwargs: Any):
    """Merge completed full-feature XGBoost comparison shards."""
    from . import plain_xgboost_staged as _plain_xgboost_staged

    return _plain_xgboost_staged.merge_plain_xgboost_baseline_shards(*args, **kwargs)

def run_featureimp_postanalysis_stage(*args: Any, **kwargs: Any):
    """Run one standalone optional FeatureImp side-analysis stage."""

    from . import postanalysis_runner as _postanalysis_runner

    return _postanalysis_runner.run_featureimp_postanalysis_stage(*args, **kwargs)


def run_featureimp_global_fsea_from_file(
    *,
    ranked_list_path: str,
    output_dir: str,
    feature_col: str = "feature_name",
    rank_col: str = "Global_Importance",
    label: str = "",
) -> str:
    """Run a single global-importance FSEA directly from a ranking CSV.

    This exposes a lighter-weight helper for standalone preranked global FSEA runs.
    """

    from .fsea_runner import run_global_importance_fsea_from_file as _run_fsea_from_file
    from .runtime_settings import FSEARuntimeSettings

    settings = FSEARuntimeSettings.from_mapping(_LAST_SETTINGS)
    return _run_fsea_from_file(
        ranked_list_path=ranked_list_path,
        output_dir=output_dir,
        feature_col=feature_col,
        rank_col=rank_col,
        label=label,
        settings=settings,
    )


# Main API


def run_feature_importance_analysis(*args: Any, **kwargs: Any):
    """Run FeatureImp through the serial plan, process, and merge workflow.

    Use one shard while preserving the caller's settings for later stages.
    """

    if args:
        positional_names = ["normalized_matrix_path", "results_dir"]
        if len(args) > len(positional_names):
            raise TypeError(
                "run_feature_importance_analysis accepts at most two positional "
                "arguments: normalized_matrix_path, results_dir"
            )
        for name, value in zip(positional_names, args):
            if name in kwargs:
                raise TypeError(f"{name} was provided both positionally and by keyword")
            kwargs[name] = value

    normalized_matrix_path = kwargs.pop("normalized_matrix_path", None)
    results_dir = kwargs.pop("results_dir", None)
    if not normalized_matrix_path:
        raise ValueError("normalized_matrix_path is required")
    if not results_dir:
        raise ValueError("results_dir is required")

    analysis_folder_path = kwargs.pop("analysis_folder_path", None)
    unexpected = sorted(k for k in kwargs.keys() if k not in DEFAULT_SETTINGS)
    if unexpected:
        raise TypeError(
            "run_feature_importance_analysis() got unexpected keyword argument(s): "
            + ", ".join(unexpected)
        )
    analysis_run_name = str(kwargs.get("analysis_run_name", _LAST_SETTINGS.get("analysis_run_name", "")) or "")
    sex_filter = str(kwargs.get("sex_filter", _LAST_SETTINGS.get("sex_filter", "all")) or "all")

    if analysis_folder_path:
        resolved_analysis_folder = str(analysis_folder_path)
        os.makedirs(resolved_analysis_folder, exist_ok=True)
    else:
        sex_filter_norm = sex_filter.strip().lower()
        sex_stratum = sex_filter_norm if sex_filter_norm in {"male", "female"} else None
        analysis_folder_name = get_analysis_folder_name(analysis_run_name, sex_stratum=sex_stratum)
        timestamp_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
        resolved_analysis_folder = allocate_analysis_folder(results_dir, f"{analysis_folder_name}_{timestamp_tag}")

    requested_num_shards = kwargs.pop("num_shards", _LAST_SETTINGS.get("num_shards", 1))
    try:
        requested_num_shards_int = int(requested_num_shards)
    except Exception:
        requested_num_shards_int = 1
    if requested_num_shards_int != 1:
        warnings.warn(
            "FeatureImp stage=all now runs through the staged planner/shard/finalize pipeline serially; "
            f"forcing num_shards=1 (received {requested_num_shards_int}). Use stage=plan/process_shard/merge_shards for sharded execution with a serial merge tail.",
            RuntimeWarning,
            stacklevel=2,
        )

    requested_shard_index = kwargs.pop("shard_index", 0)
    try:
        requested_shard_index_int = int(requested_shard_index)
    except Exception:
        requested_shard_index_int = 0
    if requested_shard_index_int != 0:
        warnings.warn(
            "FeatureImp stage=all always runs shard_index=0 in serial staged mode; "
            f"ignoring shard_index={requested_shard_index_int}.",
            RuntimeWarning,
            stacklevel=2,
        )

    # Preserve the caller's intended settings for later staged phases while
    # forcing the serial one-shard interpretation used by stage=all.
    merged_settings = dict(_LAST_SETTINGS)
    merged_settings.update({k: v for k, v in kwargs.items() if k in DEFAULT_SETTINGS})
    merged_settings.update(
        {
            "num_shards": 1,
            "shard_index": 0,
            "stage": "all",
            "results_dir": results_dir,
        }
    )
    explicit_programmatic_keys = {k for k in kwargs.keys() if k in DEFAULT_SETTINGS}
    explicit_programmatic_keys.update({
        "stage",
        "num_shards",
        "shard_index",
        })
    apply_settings(
        merged_settings,
        propagate=False,
        explicit_keys=explicit_programmatic_keys,
    )

    plan_allowed = {
        "analysis_run_name",
        "include_feature_categories",
        "control_group_tag",
        "training_group_a",
        "training_group_b",
        "sex_filter",
        "n_splits",
        "H_hits_per_feature",
        "n_features_per_model",
        "min_feature_completeness_per_group",
        "min_variance_threshold",
        "run_permutation_test",
        "n_permutations",
        "GLOBAL_RANDOM_SEED",
    }

    plan_kwargs = {k: v for k, v in kwargs.items() if k in plan_allowed}
    plan_kwargs.update(
        {
            "normalized_matrix_path": normalized_matrix_path,
            "results_dir": results_dir,
            "analysis_folder_path": resolved_analysis_folder,
            "num_shards": 1,
        }
    )

    planned_analysis_folder = plan_feature_importance_analysis(**plan_kwargs)
    process_featureimp_shard(
        analysis_folder_path=str(planned_analysis_folder),
        shard_index=0,
    )
    merge_featureimp_shards(
        analysis_folder_path=str(planned_analysis_folder),
    )

    return str(planned_analysis_folder)


__all__ = [
    "apply_settings",
    "get_current_settings",
    "get_explicit_settings",
    "get_explicit_setting_keys",
    "apply_warning_filters",
    "get_analysis_folder_name",
    "normalize_feature_matrix",
    "plan_feature_importance_analysis",
    "process_featureimp_shard",
    "merge_featureimp_shards",
    "plan_plain_xgboost_baseline_shards",
    "process_plain_xgboost_baseline_shard",
    "merge_plain_xgboost_baseline_shards",
    "run_feature_importance_analysis",
    "run_featureimp_postanalysis_stage",
    "run_featureimp_global_fsea_from_file",
    "build_settings_snapshot_payload",
    "filter_saved_settings_snapshot",
    "FEATUREIMP_RUN_SETTINGS_FROZEN_KEYS",
    "FEATUREIMP_RUN_SETTINGS_CONTEXT_KEYS",
    "FEATUREIMP_RUN_SETTINGS_POSTANALYSIS_KEYS",
    "FEATUREIMP_RUN_SETTINGS_METADATA_KEYS",
    "FEATUREIMP_RUN_SETTINGS_ALLOWED_KEYS",
]
