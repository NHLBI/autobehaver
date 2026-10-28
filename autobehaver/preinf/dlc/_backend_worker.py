"""Execute one DeepLabCut call without importing the main Autobehaver package.

Keep this standalone boundary compatible with Python 3.8 cluster containers.
Video selection, validation, retries, publication, and monitoring stay in the
supported main Python process.
"""

import inspect
import json
from pathlib import Path
import platform
import sys
import traceback


def _signature(function):
    try:
        return [{"name": item.name, "kind": int(item.kind),
                 "optional": item.default is not inspect.Parameter.empty}
                for item in inspect.signature(function).parameters.values()]
    except (TypeError, ValueError):
        return None


def _load_backend():
    import tensorflow as tf

    gpus = tf.config.experimental.list_physical_devices("GPU")
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as exc:
        print(exc, file=sys.stderr)
    import deeplabcut

    return deeplabcut, {
        "python": platform.python_version(),
        "tensorflow": getattr(tf, "__version__", None),
        "deeplabcut": getattr(deeplabcut, "__version__", None),
        "gpus": [gpu.name for gpu in gpus],
    }


def _prepare_model(config, shuffle):
    """Resolve and read the selected TensorFlow model in its actual environment."""
    from deeplabcut.utils import auxiliaryfunctions
    from deeplabcut.pose_estimation_tensorflow.config import load_config

    cfg = auxiliaryfunctions.read_config(config)
    folder = Path(cfg["project_path"]) / auxiliaryfunctions.get_model_folder(
        cfg["TrainingFraction"][0], shuffle, cfg)
    test_config = folder / "test" / "pose_cfg.yaml"
    load_config(str(test_config))
    train_config = folder / "train" / "pose_cfg.yaml"
    snapshots = sorted((folder / "train").glob("snapshot-*.index"),
                       key=lambda path: int(path.stem.split("-")[-1]))
    if not snapshots:
        raise FileNotFoundError("No DeepLabCut snapshots visible in " + str(folder / "train"))
    selected = cfg["snapshotindex"]
    if selected == "all":
        raise ValueError("DeepLabCut inference requires one selected snapshot")
    snapshot = snapshots[int(selected)]
    weights = sorted(snapshot.parent.glob(snapshot.stem + ".data-*"))
    if not weights:
        raise FileNotFoundError("DeepLabCut checkpoint weights are missing: " + str(snapshot))
    files = [train_config, snapshot] + weights
    for path in files:
        with path.open("rb") as stream:
            if not stream.read(1):
                raise ValueError("Empty DeepLabCut model file: " + str(path))
    return {"project_path": cfg["project_path"], "model_folder": str(folder),
            "snapshot": snapshot.stem, "test_pose_config": str(test_config),
            "checkpoint_files": [str(path) for path in [snapshot] + weights]}


def execute(request):
    """Dispatch only the supported backend operations with native API filtering."""
    operation = request["operation"]
    if operation not in ("probe", "prepare_model", "analyze_videos", "create_labeled_video"):
        raise ValueError("Unsupported DeepLabCut backend operation: " + str(operation))
    backend, info = _load_backend()
    if operation == "prepare_model":
        return {"ok": True, "backend": info,
                "model": _prepare_model(request["args"][0], request.get("kwargs", {}).get("shuffle", 1))}
    if operation == "probe":
        return {"ok": True, "backend": info, "signatures": {
            name: _signature(getattr(backend, name))
            for name in ("analyze_videos", "create_labeled_video")
        }}
    function = getattr(backend, operation)
    kwargs = request.get("kwargs", {})
    parameters = _signature(function)
    if parameters is not None and not any(item["kind"] == int(inspect.Parameter.VAR_KEYWORD)
                                          for item in parameters):
        allowed = {item["name"] for item in parameters}
        kwargs = {key: value for key, value in kwargs.items() if key in allowed}
    function(*request.get("args", []), **kwargs)
    return {"ok": True, "backend": info}


def main(argv=None):
    arguments = sys.argv[1:] if argv is None else argv
    if len(arguments) != 2:
        raise SystemExit("Expected a request JSON path and a response JSON path")
    request_path, response_path = map(Path, arguments)
    try:
        result = execute(json.loads(request_path.read_text(encoding="utf-8")))
        status = 0
    except Exception as exc:
        traceback.print_exc()
        result = {"ok": False, "error": type(exc).__name__ + ": " + str(exc)}
        status = 1
    response_path.write_text(json.dumps(result) + "\n", encoding="utf-8")
    return status


if __name__ == "__main__":
    raise SystemExit(main())
