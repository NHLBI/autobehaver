from __future__ import annotations

from dataclasses import fields
from importlib import import_module
from pathlib import Path

import pytest


STAGES = ("autocropper", "dlc", "keypre", "keyprenorm", "posepred_inference", "moviechecker")
CONFIG_CLASSES = {
    "autocropper": "AutocropperConfig",
    "dlc": "DLCInferencerConfig",
    "keypre": "KeypreConfig",
    "keyprenorm": "KeypreNormConfig",
    "posepred_inference": "PosePredInferenceConfig",
}


@pytest.mark.parametrize("stage", STAGES)
def test_preinf_has_direct_options_without_pipeline_yaml(stage, tmp_path):
    cli = import_module(f"autobehaver.preinf.{stage}.cli")
    parser = cli.build_arg_parser()
    assert "--config" not in parser._option_string_actions
    assert not hasattr(cli, "_load_config_defaults")
    args = cli.parse_cli_args(parser, ["--main-dir", str(tmp_path)])
    assert args.main_dir == str(tmp_path)
    assert args.resource_monitor is None
    with pytest.raises(SystemExit) as error:
        cli.parse_cli_args(parser, ["--config", "unused.yaml"])
    assert error.value.code == 2


@pytest.mark.parametrize("stage", CONFIG_CLASSES)
def test_preinf_parser_defaults_match_typed_runtime(stage):
    cli = import_module(f"autobehaver.preinf.{stage}.cli")
    config = import_module(f"autobehaver.preinf.{stage}.config")
    required = {"input_dir": "in"}
    if stage == "dlc":
        required["dlc_config"] = config.DEFAULT_DLC_CONFIG_PATH
    runtime = getattr(config, CONFIG_CLASSES[stage])(**required)
    args = cli.parse_cli_args(cli.build_arg_parser(), ["--input-dir", "in"])
    for field in fields(runtime):
        if field.name not in ("dlc_config", "inference_model") and hasattr(args, field.name):
            assert getattr(args, field.name) == getattr(runtime, field.name), field.name


@pytest.mark.parametrize("stage", STAGES)
def test_preinf_explicit_paths_override_main_dir_and_parse_once(stage, tmp_path, monkeypatch):
    cli = import_module(f"autobehaver.preinf.{stage}.cli")
    parsed = []
    captured = {}
    original_parse = cli.parse_cli_args

    def capture_parse(parser, argv):
        parsed.append(list(argv))
        return original_parse(parser, argv)

    monkeypatch.setattr(cli, "parse_cli_args", capture_parse)
    explicit_input = tmp_path / "explicit_input"
    explicit_output = tmp_path / "explicit_output"
    argv = ["--main-dir", str(tmp_path / "project"), "--input-dir", str(explicit_input)]
    if stage == "moviechecker":
        exclusions = tmp_path / "explicit_exclusions.csv"
        argv += ["--exclusions-csv", str(exclusions), "--review-all", "--playback-stride", "2"]

        class FakeApp:
            def __init__(self, **kwargs):
                captured.update(kwargs)

            def run(self):
                captured["ran"] = True

        monkeypatch.setattr(cli, "MovieCheckerApp", FakeApp)
    else:
        argv += ["--output-dir", str(explicit_output), "--registry-enabled", "--registry-lock-timeout-sec", "17"]

        def fake_run(*args, **kwargs):
            captured["cfg"] = kwargs.get("cfg", args[0] if args else None)
            return 0

        monkeypatch.setattr(cli, f"run_{stage}", fake_run)
        if stage == "dlc":
            dlc_model = tmp_path / "model.yaml"
            dlc_model.write_text("bodyparts: [point]\n")
            argv += ["--dlc-config", str(dlc_model), "--scratch-dir", str(tmp_path / "scratch")]
        if stage == "posepred_inference":
            argv += ["--no-use-tta", "--inf-average", "3"]
    if stage == "dlc":
        with pytest.raises(SystemExit) as error:
            cli.main(argv)
        assert error.value.code == 0
    else:
        cli.main(argv)
    assert len(parsed) == 1
    if stage == "moviechecker":
        assert Path(captured["input_dir"]) == explicit_input
        assert Path(captured["exclusions_csv_path"]) == exclusions
        assert captured["playback_stride"] == 2
        assert captured["review_all"] is True
        assert captured["ran"] is True
    else:
        cfg = captured["cfg"]
        assert Path(cfg.input_dir) == explicit_input
        assert Path(cfg.output_dir) == explicit_output
        assert cfg.registry_enabled is True
        assert cfg.registry_lock_timeout_sec == 17
        assert Path(cfg.registry_csv) == tmp_path / "project" / "registry.csv"
        if stage == "dlc":
            assert Path(cfg.dlc_config) == dlc_model
            assert Path(cfg.scratch_dir) == tmp_path / "scratch"
        if stage == "posepred_inference":
            assert cfg.inference_model == cli.default_models()
            assert cfg.use_tta is False
            assert cfg.inf_average == 3
            assert cfg.write_run_log is True


def test_dlc_default_lists_are_independent_between_parsers():
    cli = import_module("autobehaver.preinf.dlc.cli")
    first = cli.build_arg_parser().parse_args([])
    first.exclude_bodyparts.append("other")
    second = cli.build_arg_parser().parse_args([])
    assert second.exclude_bodyparts == ["Front", "Back"]
