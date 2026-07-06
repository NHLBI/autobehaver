from autobehaver.postinf.featureimp import core, fsea_runner
from autobehaver.postinf.featureimp.runtime_settings import FSEARuntimeSettings, ValidationRuntimeSettings, SummaryRuntimeSettings


def test_core_apply_settings_does_not_mutate_fsea_runner_defaults():
    original = fsea_runner._DEFAULT_FSEA_SETTINGS
    try:
        fsea_runner._DEFAULT_FSEA_SETTINGS = FSEARuntimeSettings.from_mapping({"num_workers": 999})
        core.apply_settings({"num_workers": 3}, propagate=True)
        assert fsea_runner._DEFAULT_FSEA_SETTINGS.num_workers == 999
    finally:
        fsea_runner._DEFAULT_FSEA_SETTINGS = original


def test_runtime_settings_dataclasses_read_expected_values():
    fsea = FSEARuntimeSettings.from_mapping({
        "GLOBAL_RANDOM_SEED": 123,
        "num_workers": 4,
        "fsea_fdr_threshold": 0.2,
        "fsea_viz_plot_all_groups": True,
    })
    assert fsea.global_random_seed == 123
    assert fsea.num_workers == 4
    assert fsea.fsea_fdr_threshold == 0.2
    assert fsea.fsea_viz_plot_all_groups is True

    validation = ValidationRuntimeSettings.from_mapping({"GLOBAL_RANDOM_SEED": 77})
    assert validation.global_random_seed == 77

    summaries = SummaryRuntimeSettings.from_mapping({"VALIDATION_ERROR_ON_FAIL": True})
    assert summaries.validation_error_on_fail is True



def test_compat_apply_settings_forwards_into_core_snapshot():
    from autobehaver.postinf.featureimp import compat

    original = core.get_current_settings()
    try:
        compat.apply_settings({"GLOBAL_RANDOM_SEED": 321, "num_workers": 2})
        current = core.get_current_settings()
        assert current["GLOBAL_RANDOM_SEED"] == 321
        assert current["num_workers"] == 2
    finally:
        core.apply_settings(original, propagate=False)


def test_featureimp_stage_parser_exposes_retained_postanalysis_stages_only():
    from autobehaver.postinf.featureimp.config import build_arg_parser

    parser = build_arg_parser()
    stage_action = next(a for a in parser._actions if getattr(a, "dest", None) == "stage")
    assert set(stage_action.choices) == {
        "all",
        "plan",
        "process_shard",
        "merge_shards",
        "plain_xgboost_baseline",
        "plain_xgboost_baseline_plan",
        "plain_xgboost_baseline_process_shard",
        "plain_xgboost_baseline_merge_shards",
        "pca",
        "fsea",
    }


def test_featureimp_core_settings_snapshot_whitelists_analysis_and_retained_postanalysis_keys():
    from autobehaver.postinf.featureimp import core

    payload = core.build_settings_snapshot_payload(
        analysis_folder_path="/tmp/example",
        analysis_folder_name="FEATURE_ANALYSIS_example",
        timestamp_tag="20260416_000000",
        settings_map={
            "control_group_tag": "CTRL",
            "training_group_a": "A",
            "training_group_b": "B",
            "analysis_run_name": "example",
            "results_dir": "/tmp/results",
            "fsea_permutations": 25,
            "run_fsea_analysis": False,
            "run_pca_analysis": False,
        },
    )

    assert payload["control_group_tag"] == "CTRL"
    assert payload["training_group_a"] == "A"
    assert payload["training_group_b"] == "B"
    assert payload["analysis_run_name"] == "example"
    assert payload["results_dir"] == "/tmp/results"
    assert payload["__settings_scope__"] == "analysis_folder_defining"
    assert payload["fsea_permutations"] == 25
    assert payload["run_fsea_analysis"] is False
    assert payload["run_pca_analysis"] is False
    assert "run_plain_xgboost_baseline" not in payload
