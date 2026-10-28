"""Check the isolated DLC boundary without loading models or running inference."""

import ast
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


def test_worker_is_standalone_python38():
    from autobehaver.preinf.dlc import _backend_worker
    source = Path(_backend_worker.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source, feature_version=(3, 8))
    imports = [item.module for item in ast.walk(tree) if isinstance(item, ast.ImportFrom)]
    imports += [alias.name for item in ast.walk(tree) if isinstance(item, ast.Import) for alias in item.names]
    assert all(not (name or "").startswith("autobehaver") for name in imports)


def test_worker_dispatch_preserves_supported_arguments_and_calls_once(monkeypatch):
    from autobehaver.preinf.dlc import _backend_worker as worker
    calls = []

    def analyze(config, videos, *, shuffle, save_as_csv, videotype, destfolder):
        calls.append((config, videos, shuffle, save_as_csv, videotype, destfolder))
        raise RuntimeError("native backend failure")

    backend = SimpleNamespace(analyze_videos=analyze)
    monkeypatch.setattr(worker, "_load_backend", lambda: (backend, {"python": "3.8"}))
    request = {"operation": "analyze_videos", "args": ["model path.yaml", ["a movie.mp4"]],
               "kwargs": {"shuffle": 1, "save_as_csv": True, "videotype": ".mp4",
                          "destfolder": "job scratch", "unsupported": 7}}
    with pytest.raises(RuntimeError, match="native backend failure"):
        worker.execute(request)
    assert calls == [("model path.yaml", ["a movie.mp4"], 1, True, ".mp4", "job scratch")]


def test_worker_main_returns_nonzero_and_backend_error(monkeypatch, tmp_path, capsys):
    from autobehaver.preinf.dlc import _backend_worker as worker
    request, response = tmp_path / "request.json", tmp_path / "response.json"
    request.write_text('{"operation":"unsupported"}', encoding="utf-8")
    monkeypatch.setattr(worker, "_load_backend", lambda: pytest.fail("invalid operation loaded backend"))
    assert worker.main([str(request), str(response)]) == 1
    payload = json.loads(response.read_text())
    assert payload["ok"] is False and "Unsupported DeepLabCut" in payload["error"]
    assert "ValueError" in capsys.readouterr().err


def test_host_dispatch_uses_actual_child_and_preserves_exit(monkeypatch, tmp_path):
    from autobehaver.preinf.dlc import backend
    worker = tmp_path / "_backend_worker.py"
    worker.write_text('''import json,sys
from pathlib import Path
request=json.loads(Path(sys.argv[1]).read_text())
response={"ok":True}
if request["operation"]=="probe":
    response.update(backend={"python":"fixture"},signatures={
        "analyze_videos":[{"name":"config","kind":1,"optional":False},
                          {"name":"videos","kind":1,"optional":False},
                          {"name":"shuffle","kind":3,"optional":True}],
        "create_labeled_video":[{"name":"config","kind":1,"optional":False},
                                {"name":"videos","kind":1,"optional":False},
                                {"name":"destfolder","kind":3,"optional":True}]})
elif request["operation"]=="analyze_videos":
    Path(request["args"][0]).write_text(json.dumps(request))
elif request["operation"]=="prepare_model":
    response.update(model={"model_folder":request["args"][0],"snapshot":"snapshot-10"})
elif request["kwargs"].get("force_failure"):
    response={"ok":False,"error":"fixture backend error"}
else:
    Path(request["kwargs"]["destfolder"]).write_text(Path(request["args"][0]).read_text())
Path(sys.argv[2]).write_text(json.dumps(response))
sys.exit(0 if response["ok"] else 17)
''', encoding="utf-8")
    monkeypatch.setattr(backend, "__file__", str(tmp_path / "backend.py"))
    monitor = SimpleNamespace(settings={})
    monkeypatch.setattr(backend, "current_monitor", lambda: monitor)
    proxy = backend.SubprocessDLC(sys.executable)
    assert monitor.settings["dlc_backend_python_version"] == "fixture"
    assert monitor.settings["dlc_backend_kind"] == "subprocess"
    assert proxy.prepare_model("model path with spaces", 1)["snapshot"] == "snapshot-10"
    assert monitor.settings["dlc_backend_model_folder"] == "model path with spaces"
    assert monitor.settings["dlc_backend_snapshot"] == "snapshot-10"
    target = tmp_path / "arguments with spaces.json"
    from autobehaver.preinf.dlc.core import _call_dlc_func
    _call_dlc_func(proxy.analyze_videos, str(target), ["movie; $literal.mp4"], shuffle=1, ignored=7)
    request = json.loads(target.read_text())
    assert request["args"] == [str(target), ["movie; $literal.mp4"]]
    assert request["kwargs"] == {"shuffle": 1}
    config = tmp_path / "model.yaml"
    config.write_text("bodyparts: [a, b]\nproject_path: /shared/model\n", encoding="utf-8")
    labeled = tmp_path / "labeled_config.json"
    from autobehaver.preinf.dlc.core import _write_temp_config
    temporary_config = _write_temp_config(str(config), {'dotsize': 24, 'pcutoff': .6, 'colormap': 'rainbow'},
                                         str(tmp_path / 'rendering_project'))
    proxy.create_labeled_video(temporary_config, ["movie.mp4"], destfolder=str(labeled))
    import yaml
    updated = yaml.safe_load(labeled.read_text())
    assert {key: updated[key] for key in ("dotsize", "pcutoff", "colormap")} == {
        "dotsize": 24, "pcutoff": .6, "colormap": "rainbow"}
    assert updated["project_path"] != "/shared/model"
    assert "/shared/model" in config.read_text()
    assert "dotsize" not in config.read_text()
    with pytest.raises(RuntimeError, match=r"exit 17.*fixture backend error"):
        proxy.create_labeled_video("config", [], force_failure=True)


def test_external_backend_does_not_import_host_tensorflow(monkeypatch, tmp_path):
    from autobehaver.preinf.dlc import backend, core
    from autobehaver.preinf.dlc.config import DLCInferencerConfig
    loaded = []
    monkeypatch.setattr(backend, "SubprocessDLC", lambda executable: loaded.append(executable) or SimpleNamespace())
    monkeypatch.setattr(core, "setup_gpu", lambda: pytest.fail("external backend initialized host TensorFlow"))
    monkeypatch.setattr(core, "_require_deeplabcut", lambda: pytest.fail("external backend imported host DLC"))
    monkeypatch.setattr(core, "_prepare_portable_dlc_config", lambda path: path)
    cfg = DLCInferencerConfig(input_dir=str(tmp_path), dlc_config="config.yaml", dlc_python="python-DLC")
    (tmp_path / 'movie.mp4').touch()
    class BackendSelected(Exception):
        pass
    def stop_after_backend_selection(*args):
        raise BackendSelected()
    monkeypatch.setattr(core, 'model_identity', stop_after_backend_selection)
    with pytest.raises(BackendSelected):
        core.run_dlc(cfg)
    assert loaded == ["python-DLC"]


@pytest.mark.parametrize("selected,expected", [(0, "snapshot-2"), (-1, "snapshot-10")])
def test_backend_model_check_selects_numeric_snapshot_without_inference(monkeypatch, tmp_path, selected, expected):
    from autobehaver.preinf.dlc import _backend_worker as worker
    import types
    model = tmp_path / "dlc-models/iteration-0/test-trainset95shuffle1"
    train = model / "train"
    train.mkdir(parents=True)
    (model / "test").mkdir()
    (model / "test/pose_cfg.yaml").write_text("dataset_type: imgaug\n")
    (train / "pose_cfg.yaml").write_text("net_type: resnet_50\n")
    for number in (10, 2):
        (train / f"snapshot-{number}.index").write_bytes(b"index")
        (train / f"snapshot-{number}.data-00000-of-00001").write_bytes(b"weights")
    calls = []
    auxiliary = SimpleNamespace(
        read_config=lambda path: {"project_path": str(tmp_path), "TrainingFraction": [.95], "snapshotindex": selected},
        get_model_folder=lambda fraction, shuffle, cfg: model.relative_to(tmp_path))
    utils = types.ModuleType("deeplabcut.utils")
    utils.auxiliaryfunctions = auxiliary
    config_module = types.ModuleType("deeplabcut.pose_estimation_tensorflow.config")
    config_module.load_config = lambda path: calls.append(Path(path).read_text())
    monkeypatch.setitem(sys.modules, "deeplabcut.utils", utils)
    monkeypatch.setitem(sys.modules, "deeplabcut.pose_estimation_tensorflow.config", config_module)
    result = worker._prepare_model("config.yaml", 1)
    assert result["snapshot"] == expected and len(calls) == 1
    assert result["checkpoint_files"] == [str(train / (expected + ".index")),
                                           str(train / (expected + ".data-00000-of-00001"))]
    (train / (expected + ".data-00000-of-00001")).unlink()
    with pytest.raises(FileNotFoundError, match="checkpoint weights are missing"):
        worker._prepare_model("config.yaml", 1)
    (model / "test/pose_cfg.yaml").unlink()
    with pytest.raises(FileNotFoundError):
        worker._prepare_model("config.yaml", 1)
