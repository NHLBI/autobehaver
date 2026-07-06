from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace


def test_stage_paths_for_main_dir(tmp_path: Path) -> None:
    from autobehaver.pipeline_layout import stage_paths_for_main_dir

    root = tmp_path / "project"
    got = {
        "moviechecker": stage_paths_for_main_dir("moviechecker", str(root)),
        "autocropper": stage_paths_for_main_dir("autocropper", str(root)),
        "dlc": stage_paths_for_main_dir("dlc", str(root)),
        "keypre": stage_paths_for_main_dir("keypre", str(root)),
        "keyprenorm": stage_paths_for_main_dir("keyprenorm", str(root)),
        "posepred_inference": stage_paths_for_main_dir("posepred_inference", str(root)),
        "infmovie": stage_paths_for_main_dir("infmovie", str(root)),
        "mmgen": stage_paths_for_main_dir("mmgen", str(root)),
        "featureimp": stage_paths_for_main_dir("featureimp", str(root)),
    }

    assert got["moviechecker"]["input_dir"].endswith("1_full_movies")
    assert got["moviechecker"]["exclusions_csv"].endswith("1_full_movies/exclusions.csv")
    assert got["autocropper"]["output_dir"].endswith("2_cropped_movies")
    assert got["dlc"]["output_dir"].endswith("3_dlc")
    assert got["keypre"]["output_dir"].endswith("4_keypoint_features")
    assert got["keyprenorm"]["output_dir"].endswith("5_normalized_features")
    assert got["posepred_inference"]["output_dir"].endswith("6_posepred_inference")
    assert got["infmovie"]["output_dir"].endswith("7_infmovies")
    assert got["mmgen"]["output_dir"].endswith("8_master_matrix")
    assert got["featureimp"]["results_dir"].endswith("9_feature_importance")


def test_moviechecker_cli_main_dir(monkeypatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}

    class _FakeApp:
        def __init__(self, *, input_dir, exclusions_csv_path, review_all, playback_stride):
            captured["input_dir"] = input_dir
            captured["exclusions_csv_path"] = exclusions_csv_path
            captured["review_all"] = review_all
            captured["playback_stride"] = playback_stride

        def run(self) -> None:
            captured["ran"] = True

    from autobehaver.preinf.moviechecker import cli as mc_cli

    monkeypatch.setattr(mc_cli, "MovieCheckerApp", _FakeApp)
    mc_cli.main(["--main_dir", str(tmp_path)])

    assert captured["input_dir"] == str((tmp_path / "1_full_movies").resolve())
    assert captured["exclusions_csv_path"] == str((tmp_path / "1_full_movies" / "exclusions.csv").resolve())
    assert captured["ran"] is True


def test_autocropper_cli_main_dir_sets_registry_default(monkeypatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}

    def _fake_run(*, cfg, args) -> None:
        captured["cfg"] = cfg
        captured["args"] = args

    from autobehaver.preinf.autocropper import cli as ac_cli

    monkeypatch.setattr(ac_cli, "run_autocropper", _fake_run)
    ac_cli.main(["--main_dir", str(tmp_path), "--registry_enabled"])

    cfg = captured["cfg"]
    assert cfg.input_dir == str((tmp_path / "1_full_movies").resolve())
    assert cfg.output_dir == str((tmp_path / "2_cropped_movies").resolve())
    assert cfg.registry_csv == str((tmp_path / "registry.csv").resolve())


def test_dlc_cli_main_dir(monkeypatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}
    dlc_cfg = tmp_path / "dlc_config.yaml"
    dlc_cfg.write_text("dlc: true\n")

    def _fake_run(cfg):
        captured["cfg"] = cfg
        return 0

    from autobehaver.preinf.dlc import cli as dlc_cli

    monkeypatch.setattr(dlc_cli, "run_dlc", _fake_run)
    try:
        dlc_cli.main(["--main_dir", str(tmp_path), "--dlc_config", str(dlc_cfg)])
    except SystemExit as e:
        assert int(getattr(e, "code", 1) or 0) == 0

    cfg = captured["cfg"]
    assert cfg.input_dir == str((tmp_path / "2_cropped_movies").resolve())
    assert cfg.output_dir == str((tmp_path / "3_dlc").resolve())


def test_keypre_cli_main_dir(monkeypatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}

    def _fake_run(*, cfg, args) -> None:
        captured["cfg"] = cfg

    from autobehaver.preinf.keypre import cli as kp_cli

    monkeypatch.setattr(kp_cli, "run_keypre", _fake_run)
    kp_cli.main(["--main_dir", str(tmp_path)])

    cfg = captured["cfg"]
    assert cfg.input_dir == str((tmp_path / "3_dlc").resolve())
    assert cfg.output_dir == str((tmp_path / "4_keypoint_features").resolve())


def test_keyprenorm_cli_main_dir(monkeypatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}

    def _fake_run(*, cfg, args) -> None:
        captured["cfg"] = cfg

    from autobehaver.preinf.keyprenorm import cli as kpn_cli

    monkeypatch.setattr(kpn_cli, "run_keyprenorm", _fake_run)
    kpn_cli.main(["--main_dir", str(tmp_path)])

    cfg = captured["cfg"]
    assert cfg.input_dir == str((tmp_path / "4_keypoint_features").resolve())
    assert cfg.output_dir == str((tmp_path / "5_normalized_features").resolve())


def test_posepred_inference_cli_main_dir(monkeypatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}
    ckpt = tmp_path / "model.pth"
    ckpt.write_text("stub")

    def _fake_run(cfg) -> None:
        captured["cfg"] = cfg

    from autobehaver.preinf.posepred_inference import cli as pp_cli

    monkeypatch.setattr(pp_cli, "run_posepred_inference", _fake_run)
    pp_cli.main(["--main_dir", str(tmp_path), "--inference_model", str(ckpt)])

    cfg = captured["cfg"]
    assert cfg.input_dir == str((tmp_path / "5_normalized_features").resolve())
    assert cfg.output_dir == str((tmp_path / "6_posepred_inference").resolve())


def test_mmgen_cli_main_dir(monkeypatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}

    def _fake_run(cfg):
        captured["cfg"] = cfg

    from autobehaver.postinf.mmgen import cli as mm_cli
    from autobehaver.postinf.mmgen import core as mm_core

    monkeypatch.setattr(mm_core, "run_mmgen", _fake_run)
    mm_cli.main(["--main_dir", str(tmp_path)])

    cfg = captured["cfg"]
    assert cfg.inference_csv_dir == str((tmp_path / "6_posepred_inference").resolve())
    assert cfg.poscalc_dir == str((tmp_path / "4_keypoint_features").resolve())
    assert cfg.output_dir == str((tmp_path / "8_master_matrix").resolve())


def test_infmovie_cli_main_dir(monkeypatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}

    def _fake_run(cfg) -> None:
        captured["cfg"] = cfg

    from autobehaver.postinf.infmovie import cli as im_cli
    from autobehaver.postinf.infmovie import core as im_core

    monkeypatch.setattr(im_core, "run_infmovie", _fake_run)
    im_cli.main(["--main_dir", str(tmp_path)])

    cfg = captured["cfg"]
    assert cfg.inference_csv_dir == str((tmp_path / "6_posepred_inference").resolve())
    assert cfg.movie_dir == str((tmp_path / "2_cropped_movies").resolve())
    assert cfg.output_dir == str((tmp_path / "7_infmovies").resolve())


def test_featureimp_matrix_results_dirs_and_main_dir(tmp_path: Path) -> None:
    from autobehaver.postinf.featureimp.config import build_arg_parser, DEFAULT_SETTINGS
    from autobehaver.postinf.featureimp.cli import _apply_featureimp_main_dir

    parser = build_arg_parser(DEFAULT_SETTINGS)
    args = parser.parse_args(["--matrix_dir", str(tmp_path / "mm"), "--results_dir", str(tmp_path / "fi")])
    assert args.matrix_dir == str(tmp_path / "mm")
    assert args.results_dir == str(tmp_path / "fi")

    args2 = argparse.Namespace(
        main_dir=str(tmp_path),
        matrix_dir=DEFAULT_SETTINGS["matrix_dir"],
        results_dir=DEFAULT_SETTINGS["results_dir"],
    )
    _apply_featureimp_main_dir(args2, argv_list=["--main_dir", str(tmp_path)], config_defaults={})
    assert args2.matrix_dir == str((tmp_path / "8_master_matrix").resolve())
    assert args2.results_dir == str((tmp_path / "9_feature_importance").resolve())
