"""Helpers for FeatureImp comparison-baseline reruns and surfacing.

This module stays lightweight and focuses on two tasks:
- loading the staged analysis context needed to rerun the comparison baselines;
- maintaining a small root-level summary note and postanalysis-index entries so
  the baseline outputs are easy to find after merge/finalize or standalone reruns.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any, Dict, List, Mapping, Optional

from .config import DEFAULT_SETTINGS
from .plan import load_featureimp_plan_manifest
from .shard_runner import _load_featureimp_context, _load_settings_snapshot

POSTANALYSIS_INDEX_FILENAME = "POSTANALYSIS_INDEX.json"
BASELINE_OUTPUTS_SUMMARY_FILENAME = "FEATUREIMP_BASELINE_OUTPUTS.txt"

PLAIN_XGBOOST_STAGE_NAME = "plain_xgboost_baseline"

PLAIN_XGBOOST_DIRNAME = "PLAIN_XGBOOST_BASELINE"
PLAIN_XGBOOST_FILES = [
    "PLAIN_XGBOOST_FOLD_PERFORMANCE.csv",
    "PLAIN_XGBOOST_REPEAT_PERFORMANCE.csv",
    "PLAIN_XGBOOST_POOLED_PERFORMANCE.csv",
    "PLAIN_XGBOOST_CONFIG.json",
    "PLAIN_XGBOOST_SHAP_MANIFEST.json",
    "AVERAGED_PREDICTIONS_FULL.xlsx",
    "AVERAGED_PREDICTIONS_FORMATTED_FOR_GRAPHPAD.xlsx",
    "SHAP_CONSOLIDATED_SUMMARY.xlsx",
    "SHAP_CONSOLIDATED_SUMMARY_INSAMPLE_VS_OUTOFSAMPLE.xlsx",
    "SHAP_FEATURE_STABILITY_ACROSS_REPEATS.csv",
    "SHAP_PER_INDIVIDUAL_SUMMARY.xlsx",
    "SHAP_CONSOLIDATED_SUMMARY_MALE_VS_FEMALE.xlsx",
    "GLOBAL_IMPORTANCE_RANKING.csv",
]


def collect_plain_xgboost_artifact_paths(analysis_folder_path: str) -> List[str]:
    analysis_dir = str(analysis_folder_path)
    plain_dir = os.path.join(analysis_dir, PLAIN_XGBOOST_DIRNAME)
    artifact_paths: List[str] = [os.path.join(plain_dir, name) for name in PLAIN_XGBOOST_FILES]
    perm_dirs = [name for name in os.listdir(plain_dir)] if os.path.isdir(plain_dir) else []
    for dirname in sorted(perm_dirs):
        if not str(dirname).startswith("PERMUTED_") or not str(dirname).endswith("_RESULTS"):
            continue
        perm_dir = os.path.join(plain_dir, dirname)
        artifact_paths.append(perm_dir)
        for root, _, files in os.walk(perm_dir):
            for filename in sorted(files):
                artifact_paths.append(os.path.join(root, filename))
    fsea_dir = os.path.join(plain_dir, "FSEA")
    if os.path.isdir(fsea_dir):
        artifact_paths.append(fsea_dir)
        for root, _, files in os.walk(fsea_dir):
            for filename in sorted(files):
                artifact_paths.append(os.path.join(root, filename))
    return artifact_paths


def _safe_relpath(path: Optional[str], start: str) -> Optional[str]:
    if not path:
        return None
    try:
        return os.path.relpath(str(path), start=str(start))
    except Exception:
        return str(path)


def load_featureimp_baseline_context(analysis_folder_path: str) -> Dict[str, Any]:
    analysis_dir = str(analysis_folder_path)
    manifest = load_featureimp_plan_manifest(analysis_dir)
    settings_snapshot = _load_settings_snapshot(analysis_dir)
    context = _load_featureimp_context(analysis_dir, manifest)

    training_groups = list(manifest.get("training_groups") or [])
    if len(training_groups) != 2:
        raise ValueError(
            f"Expected exactly two training groups in manifest, got: {training_groups!r}"
        )

    raw_seed = settings_snapshot.get("GLOBAL_RANDOM_SEED", DEFAULT_SETTINGS.get("GLOBAL_RANDOM_SEED", 7))
    try:
        global_random_seed = int(raw_seed) if raw_seed is not None else int(DEFAULT_SETTINGS.get("GLOBAL_RANDOM_SEED", 7) or 7)
    except Exception:
        global_random_seed = int(DEFAULT_SETTINGS.get("GLOBAL_RANDOM_SEED", 7) or 7)

    n_splits_raw = manifest.get("n_splits", settings_snapshot.get("n_splits", DEFAULT_SETTINGS.get("n_splits", 5)))
    try:
        n_splits = int(n_splits_raw) if n_splits_raw is not None else int(DEFAULT_SETTINGS.get("n_splits", 5) or 5)
    except Exception:
        n_splits = int(DEFAULT_SETTINGS.get("n_splits", 5) or 5)

    return {
        "analysis_folder_path": analysis_dir,
        "manifest": manifest,
        "settings_snapshot": settings_snapshot,
        "context": context,
        "training_group_a": str(training_groups[0]),
        "training_group_b": str(training_groups[1]),
        "n_splits": int(n_splits),
        "global_random_seed": int(global_random_seed),
    }


def _load_postanalysis_index(analysis_folder_path: str) -> Dict[str, Any]:
    path = os.path.join(str(analysis_folder_path), POSTANALYSIS_INDEX_FILENAME)
    if not os.path.exists(path):
        return {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "analysis_folder": str(analysis_folder_path),
            "stage_count": 0,
            "stages": [],
        }
    try:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        if not isinstance(payload, dict):
            raise ValueError("Invalid postanalysis index payload type")
    except Exception:
        payload = {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "analysis_folder": str(analysis_folder_path),
            "stage_count": 0,
            "stages": [],
        }
    payload.setdefault("analysis_folder", str(analysis_folder_path))
    payload.setdefault("stages", [])
    return payload


def upsert_postanalysis_stage_record(analysis_folder_path: str, stage_record: Mapping[str, Any]) -> str:
    analysis_dir = str(analysis_folder_path)
    payload = _load_postanalysis_index(analysis_dir)
    stages = [dict(r) for r in list(payload.get("stages") or []) if isinstance(r, Mapping)]
    stage_name = str(stage_record.get("stage", "")).strip()
    replaced = False
    for idx, row in enumerate(stages):
        if str(row.get("stage", "")).strip() == stage_name:
            stages[idx] = dict(stage_record)
            replaced = True
            break
    if not replaced:
        stages.append(dict(stage_record))

    payload["generated_at"] = datetime.now().isoformat(timespec="seconds")
    payload["analysis_folder"] = analysis_dir
    payload["stage_count"] = int(len(stages))
    payload["stages"] = stages

    out_path = os.path.join(analysis_dir, POSTANALYSIS_INDEX_FILENAME)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)
    return out_path


def write_baseline_outputs_summary(
    analysis_folder_path: str,
    *,
    settings_snapshot: Optional[Mapping[str, Any]] = None,
) -> str:
    analysis_dir = str(analysis_folder_path)
    settings = dict(settings_snapshot or _load_settings_snapshot(analysis_dir) or {})
    index_payload = _load_postanalysis_index(analysis_dir)
    stage_map = {
        str(rec.get("stage", "")).strip(): dict(rec)
        for rec in list(index_payload.get("stages") or [])
        if isinstance(rec, Mapping)
    }

    plain_xgb_dir = os.path.join(analysis_dir, PLAIN_XGBOOST_DIRNAME)
    plain_xgb_paths = collect_plain_xgboost_artifact_paths(analysis_dir)

    lines = [
        "FeatureImp baseline comparison outputs",
        f"Generated at: {datetime.now().isoformat(timespec='seconds')}",
        f"Analysis folder: {analysis_dir}",
        "",
        "Plain XGBoost baseline:",
        f"  Status: {str(stage_map.get(PLAIN_XGBOOST_STAGE_NAME, {}).get('status', 'not_recorded'))}",
        f"  Output directory: {_safe_relpath(plain_xgb_dir, analysis_dir)}{' [present]' if os.path.isdir(plain_xgb_dir) else ' [missing]'}",
    ]
    for path in plain_xgb_paths:
        rel = _safe_relpath(path, analysis_dir)
        prefix = "  Directory:" if os.path.isdir(path) else "  File:"
        lines.append(f"{prefix} {rel}{' [present]' if os.path.exists(path) else ' [missing]'}")

    out_path = os.path.join(analysis_dir, BASELINE_OUTPUTS_SUMMARY_FILENAME)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return out_path
