"""Command-line interface for FeatureImp planning, shard execution, merging, and
postanalysis.

Analysis dependencies load when a stage runs, allowing help in minimal environments.
"""

from __future__ import annotations

from autobehaver.cli_summary import reported_stage, completion_details

import argparse
import os
from autobehaver.portable_paths import generated_path
import sys
import time
from typing import Any, Dict, Sequence, Set

from autobehaver.argparse_utils import normalize_argv, parse_cli_args
from autobehaver.resource_monitor import add_monitor_arguments, monitored_cli, record_settings
from autobehaver.pipeline_layout import stage_paths_for_main_dir

from .config import DEFAULT_SETTINGS, build_arg_parser


def _import_core():
    """Import the FeatureImp public workflow module only when an analysis stage runs."""

    try:
        from . import core  # Defer analysis imports until a stage runs.
    except ModuleNotFoundError as e:
        missing = getattr(e, "name", None) or str(e)
        raise ModuleNotFoundError(
            "Feature importance analysis requires optional dependencies that are not installed. "
            f"Missing module: {missing}. "
            "Install the featureimp dependencies (e.g., xgboost, shap) "
            "and retry."
        ) from e

    return core



def _collect_explicit_cli_setting_keys(parser: argparse.ArgumentParser, argv_list: Sequence[str]) -> Set[str]:
    """Collect parser dest names that were explicitly supplied on the CLI."""

    option_to_dest: Dict[str, str] = {}
    for action in parser._actions:
        dest = str(getattr(action, "dest", "") or "")
        if not dest:
            continue
        for option_string in getattr(action, "option_strings", []) or []:
            option_to_dest[str(option_string)] = dest

    explicit: Set[str] = set()
    for raw_token in list(argv_list or []):
        token = str(raw_token)
        if not token.startswith("-"):
            continue
        option = token.split("=", 1)[0]
        dest = option_to_dest.get(option)
        if dest:
            explicit.add(dest)
    return explicit


def _create_timestamped_analysis_folder(core) -> tuple[str, str, str, str]:
    """Create a portable run directory from the run name, sex stratum, and timestamp."""
    sex_filter_norm = str(getattr(core, "sex_filter", "all") or "all").strip().lower()
    sex_stratum = sex_filter_norm if sex_filter_norm in {"male", "female"} else None
    timestamp_tag = time.strftime("%Y%m%d_%H%M%S")
    analysis_folder_name = core.get_analysis_folder_name(
        core.analysis_run_name,
        sex_stratum=sex_stratum,
    )
    analysis_folder_name = f"{analysis_folder_name}_{timestamp_tag}"
    analysis_folder_path = core.allocate_analysis_folder(core.results_dir, analysis_folder_name)
    analysis_folder_name = os.path.basename(analysis_folder_path)

    print(f"\nAnalysis folder created: {analysis_folder_name}")
    print(f"Matrix dir: {getattr(core, 'matrix_dir', '')}")
    print(f"Results dir: {core.results_dir}")
    print(f"Found raw matrix: {core.matrix_path}")
    print(f"Sex filter: {sex_filter_norm}")

    return analysis_folder_path, analysis_folder_name, timestamp_tag, sex_filter_norm



def _normalize_matrix_or_raise(core, analysis_folder_path: str) -> str:
    """Normalize the selected master matrix and require a resulting output path."""
    print("\n--- Normalizing Feature Matrix ---")
    normalized_matrix_path = core.normalize_feature_matrix(
        results_dir=core.results_dir,
        raw_matrix_path=core.matrix_path,
        skip_if_normalized_exists=False,
        analysis_folder_path=analysis_folder_path,
    )
    if normalized_matrix_path is None:
        raise FileNotFoundError(
            f"'{core.matrix_path}' not found. Cannot normalize. "
            "Confirm --matrix_dir and/or --matrix_path."
        )
    return str(normalized_matrix_path)



def _run_stage_all(core, analysis_folder_path: str, normalized_matrix_path: str) -> None:
    """Run the serial FeatureImp workflow and report its completion."""
    print("\n" + "=" * 80)
    print("RUNNING FEATURE IMPORTANCE ANALYSIS (K-Fold Committee Design)")
    print("=" * 80 + "\n")

    core.run_feature_importance_analysis(
        normalized_matrix_path=normalized_matrix_path,
        results_dir=core.results_dir,
        analysis_run_name=core.analysis_run_name,
        include_feature_categories=core.include_feature_categories,
        analysis_folder_path=analysis_folder_path,
        control_group_tag=core.control_group_tag,
        training_group_a=core.training_group_a,
        training_group_b=core.training_group_b,
        n_splits=core.n_splits,
        H_hits_per_feature=core.H_hits_per_feature,
        n_features_per_model=core.n_features_per_model,
        sex_filter=core.sex_filter,
    )



def _run_stage_plan(core, analysis_folder_path: str, normalized_matrix_path: str) -> None:
    """Create the frozen FeatureImp plan and report its artifacts."""
    print("\n" + "=" * 80)
    print("PLANNING FEATURE IMPORTANCE SHARDED RUN")
    print("=" * 80 + "\n")

    core.plan_feature_importance_analysis(
        normalized_matrix_path=normalized_matrix_path,
        results_dir=core.results_dir,
        analysis_run_name=core.analysis_run_name,
        include_feature_categories=core.include_feature_categories,
        analysis_folder_path=analysis_folder_path,
        control_group_tag=core.control_group_tag,
        training_group_a=core.training_group_a,
        training_group_b=core.training_group_b,
        n_splits=core.n_splits,
        H_hits_per_feature=core.H_hits_per_feature,
        n_features_per_model=core.n_features_per_model,
        sex_filter=core.sex_filter,
        num_shards=core.num_shards,
    )



def _run_stage_process_shard(core, analysis_folder_path: str) -> None:
    print("\n" + "=" * 80)
    print("PROCESSING FEATURE IMPORTANCE SHARD")
    print("=" * 80 + "\n")

    core.process_featureimp_shard(
        analysis_folder_path=analysis_folder_path,
        shard_index=int(getattr(core, "shard_index", 0) or 0),
    )



def _run_stage_merge_shards(core, analysis_folder_path: str) -> None:
    print("\n" + "=" * 80)
    print("MERGING FEATURE IMPORTANCE SHARDS")
    print("=" * 80 + "\n")

    core.merge_featureimp_shards(
        analysis_folder_path=analysis_folder_path,
    )



def _run_stage_postanalysis(core, analysis_folder_path: str, stage_name: str) -> None:
    print("\n" + "=" * 80)
    print(f"RUNNING FEATUREIMP POSTANALYSIS STAGE: {stage_name}")
    print("=" * 80 + "\n")

    core.run_featureimp_postanalysis_stage(
        analysis_folder_path=analysis_folder_path,
        stage_name=str(stage_name),
    )


def _run_stage_fsea_from_ranking_file(
    core,
    *,
    ranked_list_path: str,
    output_dir: str,
    feature_col: str,
    rank_col: str,
    label: str,
) -> None:
    """Run standalone global-importance enrichment from the requested ranking file."""
    print("\n" + "=" * 80)
    print("RUNNING FEATUREIMP FSEA FROM RANKING FILE")
    print("=" * 80 + "\n")

    core.run_featureimp_global_fsea_from_file(
        ranked_list_path=ranked_list_path,
        output_dir=output_dir,
        feature_col=feature_col,
        rank_col=rank_col,
        label=label,
    )


def _run_stage_plain_xgboost_baseline_plan(core, analysis_folder_path: str) -> None:
    print("\n" + "=" * 80)
    print("PLANNING PLAIN-XGBOOST BASELINE SHARDED RUN")
    print("=" * 80 + "\n")

    core.plan_plain_xgboost_baseline_shards(
        analysis_folder_path=analysis_folder_path,
        num_shards=int(getattr(core, "num_shards", 1) or 1),
    )


def _run_stage_plain_xgboost_baseline_process_shard(core, analysis_folder_path: str) -> None:
    print("\n" + "=" * 80)
    print("PROCESSING PLAIN-XGBOOST BASELINE SHARD")
    print("=" * 80 + "\n")

    core.process_plain_xgboost_baseline_shard(
        analysis_folder_path=analysis_folder_path,
        shard_index=int(getattr(core, "shard_index", 0) or 0),
    )


def _run_stage_plain_xgboost_baseline_merge_shards(core, analysis_folder_path: str) -> None:
    print("\n" + "=" * 80)
    print("MERGING PLAIN-XGBOOST BASELINE SHARDS")
    print("=" * 80 + "\n")

    core.merge_plain_xgboost_baseline_shards(
        analysis_folder_path=analysis_folder_path,
    )


def _argv_has_option(argv_list: Sequence[str], option_names: Sequence[str]) -> bool:
    opts = set(option_names)
    for tok in argv_list:
        if not tok.startswith("--"):
            continue
        opt = tok.split("=", 1)[0]
        if opt in opts:
            return True
    return False


def _apply_featureimp_main_dir(args, *, argv_list: Sequence[str]) -> None:
    """Fill standard input and output directories without replacing explicitly supplied
    paths.
    """
    main_dir = str(getattr(args, "main_dir", "") or "").strip()
    if not main_dir:
        return

    derived = stage_paths_for_main_dir("featureimp", main_dir)

    explicit_matrix_dir = _argv_has_option(
        argv_list,
        ["--matrix_dir"],
    )
    explicit_results_dir = _argv_has_option(
        argv_list,
        ["--results_dir"],
    )

    if not explicit_matrix_dir:
        args.matrix_dir = derived["matrix_dir"]

    if not explicit_results_dir:
        args.results_dir = derived["results_dir"]



@monitored_cli("featureimp")
@reported_stage("FeatureImp", active=False)
def main(argv: Sequence[str] | None = None) -> None:
    """Run feature importance analysis from the command line."""

    argv_list = list(argv) if argv is not None else sys.argv[1:]
    parser = build_arg_parser()
    add_monitor_arguments(parser)
    argv_list = normalize_argv(argv_list, parser=parser)
    args = parse_cli_args(parser, argv_list)
    _apply_featureimp_main_dir(args, argv_list=argv_list)
    record_settings(args)

    stage = str(getattr(args, "stage", "all") or "all").strip().lower()

    completion_details(stage=f"FeatureImp {stage.replace('_', ' ')}")

    explicit_setting_keys = _collect_explicit_cli_setting_keys(parser, argv_list)

    core = _import_core()
    propagate_settings = stage == "all"
    core.apply_settings(
        {**DEFAULT_SETTINGS, **vars(args)},
        propagate=propagate_settings,
        explicit_keys=explicit_setting_keys,
    )
    core.apply_warning_filters()

    if stage in {"all", "plan"}:
        analysis_folder_path, analysis_folder_name, timestamp_tag, _ = _create_timestamped_analysis_folder(core)
        completion_details(output=analysis_folder_path)
        normalized_matrix_path = _normalize_matrix_or_raise(core, analysis_folder_path)

        if stage == "all":
            _run_stage_all(core, analysis_folder_path, normalized_matrix_path)
            completion_details(output=analysis_folder_path)
            return

        _run_stage_plan(core, analysis_folder_path, normalized_matrix_path)
        completion_details(output=analysis_folder_path)
        return

    plain_xgboost_baseline_staged_stages = {
        "plain_xgboost_baseline_plan",
        "plain_xgboost_baseline_process_shard",
        "plain_xgboost_baseline_merge_shards",
    }

    postanalysis_stages = {
        "plain_xgboost_baseline",
        "pca",
        "fsea",
    }

    if stage == "fsea":
        analysis_folder_path = str(getattr(args, "analysis_folder_path", "") or "").strip()
        ranked_list_path = str(getattr(args, "fsea_ranked_list_path", "") or "").strip()
        fsea_output_dir = str(getattr(args, "fsea_output_dir", "") or "").strip()
        feature_col = str(getattr(args, "fsea_feature_col", "feature_name") or "feature_name").strip() or "feature_name"
        rank_col = str(getattr(args, "fsea_rank_col", "Global_Importance") or "Global_Importance").strip() or "Global_Importance"
        label = str(getattr(args, "fsea_label", "") or "").strip()

        using_ranking_file_mode = bool(ranked_list_path or fsea_output_dir)
        if using_ranking_file_mode:
            if analysis_folder_path:
                raise SystemExit(
                    "Stage 'fsea' accepts either --analysis_folder_path or the pair "
                    "--fsea_ranked_list_path/--fsea_output_dir, not both."
                )
            if not ranked_list_path or not fsea_output_dir:
                raise SystemExit(
                    "Stage 'fsea' ranking-file mode requires both --fsea_ranked_list_path "
                    "and --fsea_output_dir."
                )

            _run_stage_fsea_from_ranking_file(
                core,
                ranked_list_path=ranked_list_path,
                output_dir=fsea_output_dir,
                feature_col=feature_col,
                rank_col=rank_col,
                label=label,
            )
            completion_details(output=fsea_output_dir)
            return

    if stage in {"process_shard", "merge_shards"} | plain_xgboost_baseline_staged_stages | postanalysis_stages:
        analysis_folder_path = str(getattr(args, "analysis_folder_path", "") or "").strip()
        if not analysis_folder_path:
            if stage == "fsea":
                raise SystemExit(
                    "Stage 'fsea' requires either --analysis_folder_path or the pair "
                    "--fsea_ranked_list_path/--fsea_output_dir."
                )
            raise SystemExit(
                f"Stage '{stage}' requires --analysis_folder_path pointing to an existing planned FeatureImp analysis folder."
            )
        if not os.path.isdir(analysis_folder_path):
            raise SystemExit(f"FeatureImp analysis folder does not exist: {analysis_folder_path}")

        if stage == "process_shard":
            _run_stage_process_shard(core, analysis_folder_path)
            completion_details(output=analysis_folder_path)
            return

        if stage == "merge_shards":
            _run_stage_merge_shards(core, analysis_folder_path)
            completion_details(output=analysis_folder_path)
            return

        if stage == "plain_xgboost_baseline_plan":
            _run_stage_plain_xgboost_baseline_plan(core, analysis_folder_path)
            completion_details(output=analysis_folder_path)
            return

        if stage == "plain_xgboost_baseline_process_shard":
            _run_stage_plain_xgboost_baseline_process_shard(core, analysis_folder_path)
            completion_details(output=analysis_folder_path)
            return

        if stage == "plain_xgboost_baseline_merge_shards":
            _run_stage_plain_xgboost_baseline_merge_shards(core, analysis_folder_path)
            completion_details(output=analysis_folder_path)
            return

        if stage in postanalysis_stages:
            _run_stage_postanalysis(core, analysis_folder_path, stage)
            completion_details(output=analysis_folder_path)
            return
        raise SystemExit(
            f"Stage '{stage}' is not implemented. The standard sharded workflow is: plan -> process_shard -> merge_shards."
        )

    raise SystemExit(f"Unsupported FeatureImp stage: {stage}")


if __name__ == "__main__":
    main()
