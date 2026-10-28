"""Merge completed FeatureImp shards, compute subset weights, and finalize reports.

Run reference FSEA and PCA analyses when enabled.
"""

from __future__ import annotations
from autobehaver.completion import verified_run

import os
from .integrity import strict_bool

from .subset_weight_merge import calculate_subset_weights_run
from .final_merge import finalize_featureimp_outputs_run
from .plan import load_featureimp_plan_manifest


def _run_default_reference_postanalysis(analysis_folder_path: str) -> None:
    """Run lightweight reference post-merge analyses when enabled.

    The public one-shot path (stage=all) and the sharded convenience tail
    (stage=merge_shards) both pass through this helper, so users get the same
    final FSEA + PCA outputs without a separate postanalysis command.
    """

    from . import core as _core
    from .postanalysis_runner import run_featureimp_postanalysis_stage
    from .shard_runner import _load_settings_snapshot
    import traceback

    # Prefer the analysis folder's saved settings for merge-tail behavior, then
    # layer on any settings explicitly supplied for this invocation. This keeps
    # a staged run planned with --no-run_pca_analysis or --no-run_fsea_analysis
    # consistent when the final merge is launched later as a separate command,
    # while still allowing an explicit merge-time override.
    settings = dict(_load_settings_snapshot(str(analysis_folder_path)) or {})
    if not settings:
        settings = dict(_core.get_current_settings() or {})
    current_explicit = dict(_core.get_explicit_settings() or {})
    for key, value in current_explicit.items():
        key_str = str(key or "").strip()
        if not key_str or key_str.startswith("__") or key_str in {"stage", "analysis_folder_path"}:
            continue
        settings[key_str] = value

    strict = strict_bool(settings.get("postanalysis_strict", False))
    debug = strict_bool(settings.get("postanalysis_debug", False))
    stages = [
        ("fsea", strict_bool(settings.get("run_fsea_analysis", True))),
        ("pca", strict_bool(settings.get("run_pca_analysis", True))),
    ]
    for stage_name, enabled in stages:
        if not enabled:
            print(f"[merge_shards] Skipping automatic {stage_name.upper()} postanalysis (--no-run_{stage_name}_analysis).")
            continue
        try:
            print(f"\n[merge_shards] Running automatic {stage_name.upper()} postanalysis")
            run_featureimp_postanalysis_stage(
                analysis_folder_path=str(analysis_folder_path),
                stage_name=str(stage_name),
            )
        except Exception as e:
            if debug:
                traceback.print_exc()
            raise RuntimeError(f'Requested automatic {stage_name.upper()} postanalysis failed') from e


@verified_run
def merge_featureimp_shards_run(analysis_folder_path: str) -> str:
    """Calculate subset weights, finalize completed shards, and run enabled reference
    postanalyses.
    """
    analysis_folder_path = os.path.abspath(str(analysis_folder_path))
    if not os.path.isdir(analysis_folder_path):
        raise FileNotFoundError(f"FeatureImp analysis folder not found: {analysis_folder_path}")

    manifest = load_featureimp_plan_manifest(analysis_folder_path)
    num_shards = int(manifest.get("active_num_shards", manifest.get("num_shards", manifest.get("requested_num_shards", 0))) or 0)
    if num_shards <= 0:
        raise RuntimeError(f"FeatureImp plan manifest has invalid num_shards={num_shards!r}")

    print("\n" + "=" * 80)
    print("MERGING FEATURE IMPORTANCE SHARDS")
    print("=" * 80 + "\n")

    calculate_subset_weights_run(analysis_folder_path)
    finalize_featureimp_outputs_run(analysis_folder_path)
    _run_default_reference_postanalysis(analysis_folder_path)
    return analysis_folder_path


__all__ = ["merge_featureimp_shards_run"]
