from pathlib import Path

import yaml


def test_pipeline_template_yaml_loads_and_exposes_stage_sections():
    cfg_path = Path(__file__).resolve().parents[1] / "configs" / "pipeline_template.yaml"
    data = yaml.safe_load(cfg_path.read_text())
    assert isinstance(data, dict)
    assert data["project"]["main_dir"] == "../my_autobehaver_project"
    for section in [
        "moviechecker",
        "autocropper",
        "dlc",
        "keypre",
        "keyprenorm",
        "posepred_inference",
        "infmovie",
        "mmgen",
        "featureimp",
        "posepred_training",
    ]:
        assert isinstance(data.get(section), dict), section


def test_global_main_dir_fallback_is_seen_by_stage_config_loaders():
    cfg = str(Path(__file__).resolve().parents[1] / "configs" / "pipeline_template.yaml")
    expected_suffix = "my_autobehaver_project"

    from autobehaver.preinf.moviechecker.cli import _load_config_defaults as moviechecker
    from autobehaver.preinf.autocropper.cli import _load_config_defaults as autocropper
    from autobehaver.preinf.dlc.cli import _load_config_defaults as dlc
    from autobehaver.preinf.keypre.cli import _load_config_defaults as keypre
    from autobehaver.preinf.keyprenorm.cli import _load_config_defaults as keyprenorm
    from autobehaver.preinf.posepred_inference.cli import _load_config_defaults as posepred_inference
    from autobehaver.postinf.infmovie.cli import _load_config_defaults as infmovie
    from autobehaver.postinf.mmgen.cli import _load_config_defaults as mmgen
    from autobehaver.postinf.featureimp.cli import _load_config_yaml as featureimp

    for loader in [
        moviechecker,
        autocropper,
        dlc,
        keypre,
        keyprenorm,
        posepred_inference,
        infmovie,
        mmgen,
        featureimp,
    ]:
        defaults = loader(cfg)
        assert str(defaults.get("main_dir", "")).replace("\\", "/").endswith(expected_suffix), loader


def test_infmovie_removed_noop_flags_are_not_exposed():
    from autobehaver.postinf.infmovie.cli import build_parser

    parser = build_parser()
    help_text = parser.format_help()
    option_strings = [opt for action in parser._actions for opt in action.option_strings]
    assert "--num_movies" not in option_strings
    assert option_strings.count("--skip_first_n_frames") == 1
    assert "--skip_first_n_frames" in help_text
