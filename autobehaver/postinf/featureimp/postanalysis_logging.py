"""Write FeatureImp postanalysis manifests and persistent error logs using
standard-library dependencies.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any, Dict, Optional

import platform
import sys


def _now_timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _iso_timestamp() -> str:
    try:
        return datetime.now().isoformat(timespec="seconds")
    except Exception:
        return _now_timestamp()


def _get_pkg_version() -> str:
    try:
        from autobehaver import __version__ as _v

        return str(_v)
    except Exception:
        return "unknown"


def build_stage_manifest(
    *,
    stage: str,
    analysis_folder: str,
    output_dir: str,
    file_suffix: str = "",
    inputs: Optional[Dict[str, Any]] = None,
    settings: Optional[Dict[str, Any]] = None,
    outputs: Optional[Dict[str, Any]] = None,
    summary: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build a standardized MANIFEST dict for a post-analysis stage.

    This intentionally uses a stable schema so that downstream tooling can rely
    on common keys across stages.
    """

    _inputs = dict(inputs or {})
    _settings = dict(settings or {})
    _outputs = dict(outputs or {})
    _summary = dict(summary or {})

    # Global importance source bookkeeping
    # Many downstream post-analysis stages depend (directly or indirectly) on the canonical
    # feature-importance ranking in GLOBAL_IMPORTANCE_RANKING{file_suffix}.csv.
    # We record which semantics the Global_Importance column represents.
    ranking_csv_path = None
    try:
        ranking_csv_path = os.path.join(str(analysis_folder or '.'), f"GLOBAL_IMPORTANCE_RANKING{str(file_suffix or '')}.csv")
    except Exception:
        ranking_csv_path = None

    importance_source = None
    weighting_manifest_path = None
    try:
        weighting_manifest_path = os.path.join(str(analysis_folder or '.'), 'COMMITTEE_SUBSETS', 'SHAP_WEIGHTING_MANIFEST.json')
    except Exception:
        weighting_manifest_path = None

    if ranking_csv_path and os.path.exists(ranking_csv_path):
        try:
            with open(ranking_csv_path, 'r', encoding='utf-8') as f:
                header_line = f.readline().strip('\n')
            cols = header_line.split(',') if header_line else []
            if ('Global_Importance_PermBaseline' in cols) and ('Global_Importance_WeightedRaw' in cols):
                importance_source = 'weighted_excess'
            elif weighting_manifest_path and os.path.exists(weighting_manifest_path):
                try:
                    with open(weighting_manifest_path, 'r', encoding='utf-8') as f:
                        weighting_manifest = json.load(f)
                    if weighting_manifest.get('weighted_shap'):
                        importance_source = 'weighted_excess'
                    else:
                        importance_source = 'unweighted'
                except Exception:
                    importance_source = 'unknown'
            else:
                importance_source = 'unweighted'
        except Exception:
            importance_source = 'unknown'
    else:
        importance_source = 'unknown'

    _settings.setdefault('importance_column', 'Global_Importance')
    _settings.setdefault('importance_source', importance_source)
    if ranking_csv_path:
        _settings.setdefault('global_importance_ranking_csv_detected', ranking_csv_path)

    return {
        "manifest_version": 1,
        "stage": str(stage or "unknown"),
        "created_at": _iso_timestamp(),
        "package": {
            "name": "autobehaver",
            "version": _get_pkg_version(),
        },
        "runtime": {
            "python": str(sys.version.split()[0]) if getattr(sys, "version", None) else "unknown",
            "platform": platform.platform(),
        },
        "analysis": {
            "analysis_folder": str(analysis_folder or "."),
            "output_dir": str(output_dir or "."),
            "file_suffix": str(file_suffix or ""),
        },
        "inputs": _inputs,
        "settings": _settings,
        "outputs": _outputs,
        "summary": _summary,
    }



def write_stage_manifest(
    path: str,
    *,
    stage: str,
    analysis_folder: str,
    output_dir: str,
    file_suffix: str = "",
    inputs: Optional[Dict[str, Any]] = None,
    settings: Optional[Dict[str, Any]] = None,
    outputs: Optional[Dict[str, Any]] = None,
    summary: Optional[Dict[str, Any]] = None,
) -> None:
    """Write a standardized stage manifest JSON to ``path``."""

    manifest = build_stage_manifest(
        stage=stage,
        analysis_folder=analysis_folder,
        output_dir=output_dir,
        file_suffix=file_suffix,
        inputs=inputs,
        settings=settings,
        outputs=outputs,
        summary=summary,
    )
    write_json(path, manifest)


def write_json(path: str, data: Dict[str, Any]) -> None:
    """Write a JSON file with stable formatting."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True, default=str)
        f.write("\n")


def append_postanalysis_error_log(
    *,
    analysis_folder: str,
    stage_name: str,
    error: BaseException,
    traceback_str: str,
    context: Optional[Dict[str, Any]] = None,
    log_filename: str = "POSTANALYSIS_ERROR_LOG.txt",
) -> str:
    """Append a post-analysis error record to a log file.

    Returns
    -------
    str
        Path to the error log.
    """

    analysis_folder = str(analysis_folder or ".")
    os.makedirs(analysis_folder, exist_ok=True)

    log_path = os.path.join(analysis_folder, log_filename)

    ctx = context or {}
    ts = _now_timestamp()

    with open(log_path, "a", encoding="utf-8") as f:
        f.write("=" * 100 + "\n")
        f.write(f"POST-ANALYSIS STAGE FAILURE: {stage_name}\n")
        f.write(f"Timestamp: {ts}\n")
        f.write(f"Error: {type(error).__name__}: {error}\n")
        if ctx:
            f.write("\nContext:\n")
            try:
                f.write(json.dumps(ctx, indent=2, sort_keys=True, default=str))
                f.write("\n")
            except Exception:
                # Best-effort: context should never block log writing.
                f.write(str(ctx) + "\n")
        f.write("\nTraceback:\n")
        f.write(traceback_str.rstrip() + "\n")
        f.write("\n")

    return log_path
