"""Standalone FeatureImp side-analysis runner.

This module lets retained downstream analyses run as independent jobs
or as automatic reference post-merge steps.

Design goals
------------
- Keep the staged core path lean.
- Reuse existing analysis implementations with minimal behavioral drift.
- Record persistent stage status in ``POSTANALYSIS_INDEX.json``.
- Fail loudly for standalone jobs while still writing an error log.
"""

from __future__ import annotations

import os
import traceback
from datetime import datetime
from typing import Any, Dict, Mapping, Optional, Sequence, Set

import numpy as np

from .baseline_outputs import (
    PLAIN_XGBOOST_STAGE_NAME,
    POSTANALYSIS_INDEX_FILENAME,
    upsert_postanalysis_stage_record,
    write_baseline_outputs_summary,
    load_featureimp_baseline_context,
)
from .plain_xgboost_baseline import rerun_plain_xgboost_baseline_from_analysis_folder
from .postanalysis_logging import append_postanalysis_error_log
from .shard_runner import _json_load, _load_settings_snapshot
from .final_merge import MERGE_DIRNAME, FINAL_MERGE_DONE_FILENAME, FINAL_POSTANALYSIS_BUNDLE_FILENAME
from .fsea_runner import _perform_fsea_analysis
from .runtime_settings import FSEARuntimeSettings

PLAN_ONLY_POSTANALYSIS_STAGES: Set[str] = {PLAIN_XGBOOST_STAGE_NAME}

POST_FINAL_MERGE_POSTANALYSIS_STAGES: Set[str] = {
    "fsea",
    "pca",
}

ALL_POSTANALYSIS_STAGES: Set[str] = PLAN_ONLY_POSTANALYSIS_STAGES | POST_FINAL_MERGE_POSTANALYSIS_STAGES



def _preflight_postanalysis_dependencies(analysis_folder_path: str, stage_name: str, settings_snapshot: Mapping[str, Any]) -> None:
    """Validate dependencies for retained post-analysis stages.

    Removed optional side branches are not preflighted here; retained stages perform their own input validation when run.
    """
    return None



def _safe_relpath(path: Optional[str], start: str) -> Optional[str]:
    if not path:
        return None
    try:
        return os.path.relpath(str(path), start=str(start))
    except Exception:
        return str(path)


def _postanalysis_output_dir(stage_name: str, analysis_folder_path: str, settings_snapshot: Mapping[str, Any]) -> str:
    stage = str(stage_name or "").strip().lower()
    if stage == PLAIN_XGBOOST_STAGE_NAME:
        return os.path.join(analysis_folder_path, "PLAIN_XGBOOST_BASELINE")
    if stage == "fsea":
        return os.path.join(analysis_folder_path, "FSEA")
    if stage == "pca":
        return os.path.join(analysis_folder_path, "PCA")
    return analysis_folder_path


def _require_analysis_folder(analysis_folder_path: str) -> str:
    analysis_dir = os.path.abspath(str(analysis_folder_path))
    if not os.path.isdir(analysis_dir):
        raise FileNotFoundError(f"FeatureImp analysis folder does not exist: {analysis_dir}")
    return analysis_dir


def _require_final_merge_done(analysis_folder_path: str) -> Dict[str, Any]:
    merge_done_path = os.path.join(str(analysis_folder_path), MERGE_DIRNAME, FINAL_MERGE_DONE_FILENAME)
    if not os.path.exists(merge_done_path):
        raise RuntimeError(
            f"Post-merge stage requires completed final merge, but completion marker was not found: {merge_done_path}"
        )
    payload = _json_load(merge_done_path)
    if not isinstance(payload, dict) or str(payload.get("status", "")).strip().lower() != "completed":
        raise RuntimeError(f"final-merge completion marker is invalid: {merge_done_path}")
    return payload


_POSTANALYSIS_IGNORED_EXPLICIT_KEYS: Set[str] = {
    "stage",
    "analysis_folder_path",
}


def _apply_saved_settings_snapshot(analysis_folder_path: str) -> Dict[str, Any]:
    """Apply effective settings for a standalone postanalysis stage.

    FEATUREIMP_RUN_SETTINGS.json stores only analysis-folder-defining context.
    Standalone follow-up stages start from that saved context and then layer on
    only the settings explicitly supplied for the current invocation. Merged
    defaults should not override the saved folder definition by accident.
    """

    saved_snapshot = dict(_load_settings_snapshot(str(analysis_folder_path)) or {})
    from . import core as _core

    current_explicit = dict(_core.get_explicit_settings() or {})
    effective_settings = dict(saved_snapshot)
    for key, value in current_explicit.items():
        key_str = str(key or "").strip()
        if (not key_str) or key_str in _POSTANALYSIS_IGNORED_EXPLICIT_KEYS or key_str.startswith("__"):
            continue
        effective_settings[key_str] = value

    _core.apply_settings(
        effective_settings,
        propagate=False,
        explicit_keys=current_explicit.keys(),
    )
    return effective_settings


def _build_paths(analysis_folder_path: str, *, run_permutation_test: bool = False) -> Dict[str, Any]:
    pipeline_folder = os.path.join(analysis_folder_path, "SAVED_PIPELINE")
    shap_values_folder = os.path.join(pipeline_folder, "temp_shap_values")
    models_folder = os.path.join(pipeline_folder, "models")
    os.makedirs(shap_values_folder, exist_ok=True)
    return {
        "analysis_folder": str(analysis_folder_path),
        "pipeline_folder": pipeline_folder,
        "shap_values_folder": shap_values_folder,
        "models_folder": models_folder,
        "file_suffix": "",
        "permuted_base_folder": os.path.join(analysis_folder_path, "PERMUTED_RESULTS") if bool(run_permutation_test) else None,
    }


def _record_and_raise_stage_failure(
    *,
    analysis_folder_path: str,
    stage_name: str,
    started_at: str,
    output_dir: str,
    error: BaseException,
    extra_context: Optional[Mapping[str, Any]] = None,
) -> None:
    tb = traceback.format_exc()
    log_path = append_postanalysis_error_log(
        analysis_folder=str(analysis_folder_path),
        stage_name=str(stage_name),
        error=error,
        traceback_str=tb,
        context={
            "invoked_by": "standalone_stage",
            "output_dir": str(output_dir),
            **dict(extra_context or {}),
        },
    )
    stage_record = {
        "stage": str(stage_name),
        "status": "failed",
        "started_at": started_at,
        "failed_at": datetime.now().isoformat(timespec="seconds"),
        "output_dir": _safe_relpath(output_dir, str(analysis_folder_path)),
        "error_type": type(error).__name__,
        "error": str(error),
        "error_log": _safe_relpath(log_path, str(analysis_folder_path)),
        "context": {"invoked_by": "standalone_stage", **dict(extra_context or {})},
    }
    upsert_postanalysis_stage_record(str(analysis_folder_path), stage_record)
    if stage_name == PLAIN_XGBOOST_STAGE_NAME:
        write_baseline_outputs_summary(str(analysis_folder_path))
    raise error


def _run_logged_stage(
    stage_name: str,
    func,
    *,
    analysis_folder_path: str,
    settings_snapshot: Mapping[str, Any],
    output_dir: Optional[str] = None,
    extra_context: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    started_at = datetime.now().isoformat(timespec="seconds")
    out_dir = str(output_dir or _postanalysis_output_dir(stage_name, analysis_folder_path, settings_snapshot))
    try:
        result = func()
        stage_record: Dict[str, Any] = {
            "stage": str(stage_name),
            "status": "completed",
            "started_at": started_at,
            "completed_at": datetime.now().isoformat(timespec="seconds"),
            "output_dir": _safe_relpath(out_dir, str(analysis_folder_path)),
            "context": {"invoked_by": "standalone_stage", **dict(extra_context or {})},
        }
        if isinstance(result, str) and result.strip():
            stage_record["result_path"] = _safe_relpath(result, str(analysis_folder_path))
        elif isinstance(result, Mapping):
            result_path = result.get("result_path")
            if isinstance(result_path, str) and result_path.strip():
                stage_record["result_path"] = _safe_relpath(result_path, str(analysis_folder_path))
            artifact_paths = result.get("artifact_paths")
            if isinstance(artifact_paths, (list, tuple)):
                stage_record["artifact_paths"] = [
                    _safe_relpath(str(p), str(analysis_folder_path))
                    for p in artifact_paths
                    if p
                ]
            summary = result.get("summary")
            if isinstance(summary, str) and summary.strip():
                stage_record["summary"] = summary
            output_dir_override = result.get("output_dir")
            if isinstance(output_dir_override, str) and output_dir_override.strip():
                stage_record["output_dir"] = _safe_relpath(output_dir_override, str(analysis_folder_path))
        upsert_postanalysis_stage_record(str(analysis_folder_path), stage_record)
        if stage_name == PLAIN_XGBOOST_STAGE_NAME:
            write_baseline_outputs_summary(str(analysis_folder_path), settings_snapshot=settings_snapshot)
        return stage_record
    except Exception as e:
        _record_and_raise_stage_failure(
            analysis_folder_path=str(analysis_folder_path),
            stage_name=str(stage_name),
            started_at=started_at,
            output_dir=out_dir,
            error=e,
            extra_context=extra_context,
        )
        raise AssertionError("unreachable")


def _load_core_postanalysis_bundle(analysis_folder_path: str) -> Dict[str, Any]:
    bundle_path = os.path.join(str(analysis_folder_path), MERGE_DIRNAME, FINAL_POSTANALYSIS_BUNDLE_FILENAME)
    if not os.path.exists(bundle_path):
        raise FileNotFoundError(
            f"Final postanalysis bundle not found. Run merge_shards first: {bundle_path}"
        )
    payload = _json_load(bundle_path)
    if not isinstance(payload, dict) or str(payload.get("status", "")).strip().lower() != "completed":
        raise RuntimeError(f"Final postanalysis bundle is missing or invalid: {bundle_path}")
    return payload


def _run_fsea_from_analysis_folder(
    *,
    analysis_folder_path: str,
    settings_override: Optional[Mapping[str, Any]] = None,
) -> Mapping[str, Any]:
    loaded = load_featureimp_baseline_context(str(analysis_folder_path))
    settings_snapshot = dict(loaded.get("settings_snapshot") or {})
    effective_settings = dict(settings_snapshot)
    effective_settings.update(dict(settings_override or {}))
    context = dict(loaded["context"])
    manifest = dict(loaded["manifest"])
    training_group_a = str(loaded["training_group_a"])
    training_group_b = str(loaded["training_group_b"])
    bundle = _load_core_postanalysis_bundle(str(analysis_folder_path))
    rel = str(bundle.get("hybrid_shap_mmap") or "").strip()
    if not rel:
        raise RuntimeError("Final postanalysis bundle does not record a hybrid SHAP memmap path.")
    mmap_path = os.path.join(str(analysis_folder_path), rel)
    if not os.path.exists(mmap_path):
        raise FileNotFoundError(f"Hybrid SHAP memmap recorded in final bundle was not found: {mmap_path}")
    n_train = int(bundle.get("n_train", 0) or 0)
    n_all = int(bundle.get("n_all", 0) or 0)
    n_features = int(bundle.get("n_features", 0) or 0)
    if n_all <= 0 or n_features <= 0 or n_train < 0:
        raise RuntimeError(f"Final postanalysis bundle has invalid shape metadata: {bundle}")
    hybrid = np.memmap(mmap_path, dtype=np.float32, mode="r", shape=(n_all, n_features))
    oof_shap_values = np.asarray(hybrid[:n_train, :], dtype=np.float32)
    all_groups_avg_shap_values = np.asarray(hybrid, dtype=np.float32)

    paths = _build_paths(
        str(analysis_folder_path),
        run_permutation_test=bool(effective_settings.get("run_permutation_test", manifest.get("run_permutation_test", False))),
    )
    _perform_fsea_analysis(
        oof_shap_values=oof_shap_values,
        df_for_training=context["df_for_training"],
        all_feature_names=context["selected_features"],
        training_group_a=training_group_a,
        training_group_b=training_group_b,
        df_for_shap_and_agg=context["df_for_shap_and_agg"],
        paths=paths,
        le=context["le"],
        num_classes=int(context["num_classes"]),
        all_groups_avg_shap_values=all_groups_avg_shap_values,
        settings=FSEARuntimeSettings.from_mapping(effective_settings),
    )
    out_dir = os.path.join(str(analysis_folder_path), "FSEA")
    artifact_paths = [out_dir] if os.path.isdir(out_dir) else []
    return {
        "output_dir": out_dir,
        "artifact_paths": artifact_paths,
        "summary": "FSEA outputs updated from saved final SHAP bundle.",
    }


def run_featureimp_postanalysis_stage(*, analysis_folder_path: str, stage_name: str) -> str:
    analysis_dir = _require_analysis_folder(analysis_folder_path)
    stage = str(stage_name or "").strip().lower()
    if stage not in ALL_POSTANALYSIS_STAGES:
        raise ValueError(
            f"Unsupported FeatureImp postanalysis stage: {stage_name!r}. Supported stages: {sorted(ALL_POSTANALYSIS_STAGES)}"
        )

    settings_snapshot = _apply_saved_settings_snapshot(analysis_dir)

    if stage in POST_FINAL_MERGE_POSTANALYSIS_STAGES:
        _require_final_merge_done(analysis_dir)

    _preflight_postanalysis_dependencies(analysis_dir, stage, settings_snapshot)

    # Baseline jobs already have their own rerun helpers and stage-index updates.
    if stage == PLAIN_XGBOOST_STAGE_NAME:
        rerun_plain_xgboost_baseline_from_analysis_folder(
            analysis_folder_path=str(analysis_dir),
            settings_override=settings_snapshot,
        )
        return analysis_dir

    paths = _build_paths(analysis_dir, run_permutation_test=bool(settings_snapshot.get("run_permutation_test", False)))

    if stage == "pca":
        loaded = load_featureimp_baseline_context(str(analysis_dir))
        context = dict(loaded["context"])
        training_group_a = str(loaded["training_group_a"])
        training_group_b = str(loaded["training_group_b"])
        X_train = context["X_train_full"]
        X_all = context["X_shap_full"]
        df_train = context["df_for_training"]
        df_all = context["df_for_shap_and_agg"]
        selected_features = list(context["selected_features"])

        from .pca_axis import run_pca_analysis

        _run_logged_stage(
            "pca",
            lambda: run_pca_analysis(
                X_train=X_train,
                X_all_groups=X_all,
                df_train=df_train,
                df_all_groups=df_all,
                all_feature_names=selected_features,
                paths=paths,
                GLOBAL_RANDOM_SEED=int(settings_snapshot.get("GLOBAL_RANDOM_SEED", 7) or 7),
                pca_fixed_k=settings_snapshot.get("pca_fixed_k"),
                n_pca_components_to_save=int(settings_snapshot.get("n_pca_components_to_save", 0) or 0),
                training_group_a=training_group_a,
                training_group_b=training_group_b,
            ),
            analysis_folder_path=analysis_dir,
            settings_snapshot=settings_snapshot,
        )
        return analysis_dir

    if stage == "fsea":
        _run_logged_stage(
            "fsea",
            lambda: _run_fsea_from_analysis_folder(analysis_folder_path=analysis_dir, settings_override=settings_snapshot),
            analysis_folder_path=analysis_dir,
            settings_snapshot=settings_snapshot,
        )
        return analysis_dir



    raise AssertionError(f"Unhandled postanalysis stage: {stage}")


__all__ = [
    "PLAN_ONLY_POSTANALYSIS_STAGES",
    "POST_FINAL_MERGE_POSTANALYSIS_STAGES",
    "ALL_POSTANALYSIS_STAGES",
    "run_featureimp_postanalysis_stage",
]
