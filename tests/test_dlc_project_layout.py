"""Check relocated DLC projects using its native config-directory semantics."""

from pathlib import Path
import hashlib
import os
from types import SimpleNamespace

import pytest
import yaml


def _read_like_dlc(path):
    """Emulate DLC 2.3.9 read_config's project_path rewrite and YAML save."""
    path = Path(path)
    cfg = yaml.safe_load(path.read_text())
    directory = os.path.dirname(str(path))
    if cfg["project_path"] != directory:
        cfg["project_path"] = directory
        path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return cfg


def _project(tmp_path, relocated=False):
    root = tmp_path / "resource_project"
    model = root / "dlc-models/iteration-0/testNov29-trainset95shuffle1"
    for phase in ("train", "test"):
        folder = model / phase
        folder.mkdir(parents=True)
        (folder / "pose_cfg.yaml").write_text("net_type: resnet_50\n")
    for number in (7, 42):
        (model / f"train/snapshot-{number}.index").write_bytes(f"tiny index {number}".encode())
        (model / f"train/snapshot-{number}.data-00000-of-00001").write_bytes(f"tiny weights {number}".encode())
    config = root / "config.yaml"
    config.write_text(yaml.safe_dump({"project_path": "." if relocated else str(root.resolve()),
                                     "Task": "test", "date": "Nov29", "iteration": 0,
                                     "TrainingFraction": [.95], "snapshotindex": -1,
                                     "bodyparts": ["a", "b"], "batch_size": 8, "video_sets": {}}))
    return config, model


@pytest.mark.parametrize("relocated", [False, True])
def test_staged_config_directory_resolves_model_without_rewriting_source(tmp_path, relocated):
    from autobehaver.preinf.dlc.core import _prepare_portable_dlc_config
    config, model = _project(tmp_path, relocated)
    original = config.read_bytes()
    runtime = Path(_prepare_portable_dlc_config(str(config)))
    before_read = runtime.read_bytes()
    cfg = _read_like_dlc(runtime)
    selected = Path(cfg["project_path"]) / model.relative_to(config.parent)
    assert runtime.name == "config.yaml" and runtime.parent != config.parent
    assert selected.joinpath("test/pose_cfg.yaml").read_bytes() == model.joinpath("test/pose_cfg.yaml").read_bytes()
    assert os.path.samefile(selected / "train/snapshot-42.index", model / "train/snapshot-42.index")
    assert runtime.read_bytes() == before_read
    assert config.read_bytes() == original
    original_cfg = yaml.safe_load(original)
    assert {key: value for key, value in cfg.items() if key != "project_path"} == {
        key: value for key, value in original_cfg.items() if key != "project_path"}


def test_render_config_links_models_and_preserves_model_identity(tmp_path):
    from autobehaver.preinf.dlc.core import _prepare_portable_dlc_config, _write_temp_config
    from autobehaver.preinf.dlc.contract import model_identity
    config, model = _project(tmp_path)
    expected = model_identity(config, 1)
    assert set(expected["snapshot_files"]) == {
        hashlib.sha256(path.read_bytes()).hexdigest() for path in (model / "train").glob("snapshot-42.*")}
    runtime = Path(_prepare_portable_dlc_config(str(config)))
    assert model_identity(runtime, 1) == expected
    rendering_root = tmp_path / "rendering_project"
    rendering = Path(_write_temp_config(str(runtime), {"dotsize": 24, "pcutoff": .6}, str(rendering_root)))
    cfg = _read_like_dlc(rendering)
    selected = Path(cfg["project_path"]) / model.relative_to(config.parent)
    assert selected.joinpath("test/pose_cfg.yaml").read_bytes() == model.joinpath("test/pose_cfg.yaml").read_bytes()
    assert os.path.samefile(selected / "train/snapshot-42.index", model / "train/snapshot-42.index")
    assert cfg["dotsize"] == 24 and cfg["pcutoff"] == .6
    assert "dotsize" not in yaml.safe_load(runtime.read_text())
    assert "dotsize" not in yaml.safe_load(config.read_text())


def test_valid_configured_project_outside_config_directory_is_linked(tmp_path):
    from autobehaver.preinf.dlc.core import _prepare_portable_dlc_config
    from autobehaver.preinf.dlc.contract import model_identity
    original, model = _project(tmp_path)
    config_directory = tmp_path / "separate_configs"
    config_directory.mkdir()
    config = config_directory / "selected_model.yaml"
    config.write_bytes(original.read_bytes())
    expected = model_identity(config, 1)
    runtime = Path(_prepare_portable_dlc_config(str(config)))
    cfg = _read_like_dlc(runtime)
    selected = Path(cfg["project_path"]) / model.relative_to(original.parent)
    assert os.path.samefile(selected / "train/snapshot-42.index", model / "train/snapshot-42.index")
    assert model_identity(runtime, 1) == expected


def test_shared_backend_model_failure_stops_before_video_attempts(monkeypatch, tmp_path):
    from autobehaver.preinf.dlc import core
    from autobehaver.preinf.dlc.config import DLCInferencerConfig
    config, _model = _project(tmp_path)
    movies = tmp_path / "movies"
    movies.mkdir()
    (movies / "a.mp4").write_bytes(b"selection-only fixture")
    calls = []

    def prepare(path, shuffle):
        calls.append((path, shuffle))
        raise RuntimeError("shared model tree is unavailable")

    backend = SimpleNamespace(prepare_model=prepare,
                              analyze_videos=lambda *a, **k: pytest.fail("bad setup reached video retries"))
    monkeypatch.setattr(core, "_require_deeplabcut", lambda: backend)
    monkeypatch.setattr(core, "setup_gpu", lambda: None)
    cfg = DLCInferencerConfig(input_dir=str(movies), dlc_config=str(config), max_retries=10)
    with pytest.raises(RuntimeError, match="shared model tree is unavailable"):
        core.run_dlc(cfg)
    assert len(calls) == 1 and calls[0][1] == 1
