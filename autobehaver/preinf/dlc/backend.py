"""Run DeepLabCut API calls in an existing separate Python environment."""

from __future__ import annotations

import os
import sys

import inspect
import json
from pathlib import Path
import shutil
import subprocess
import tempfile

from autobehaver.resource_monitor import current_monitor


class SubprocessDLC:
    """Expose the backend API while keeping stage bookkeeping in the main Python."""

    def __init__(self, executable: str):
        resolved = shutil.which(executable)
        if resolved is None:
            raise FileNotFoundError(f"DeepLabCut Python executable not found: {executable}")
        self.executable = resolved
        self.worker = Path(__file__).with_name("_backend_worker.py").resolve()
        response = self._request("probe")
        self.info = response["backend"]
        print("DeepLabCut backend: " + json.dumps(self.info, sort_keys=True), flush=True)
        monitor = current_monitor()
        if monitor is not None:
            monitor.settings.update({
                "dlc_backend_kind": "subprocess",
                "dlc_backend_python_version": self.info.get("python"),
                "dlc_backend_tensorflow_version": self.info.get("tensorflow"),
                "dlc_backend_version": self.info.get("deeplabcut"),
                "dlc_backend_gpu_count": len(self.info.get("gpus", [])),
            })
        self.analyze_videos = self._operation("analyze_videos", response["signatures"]["analyze_videos"])
        self.create_labeled_video = self._operation("create_labeled_video", response["signatures"]["create_labeled_video"])

    def prepare_model(self, config, shuffle):
        """Check model resolution inside the backend before per-video retries."""
        model = self._request("prepare_model", [str(config)], {"shuffle": int(shuffle)})["model"]
        print("DeepLabCut model: " + json.dumps(model, sort_keys=True), flush=True)
        monitor = current_monitor()
        if monitor is not None:
            monitor.settings.update({"dlc_backend_model_folder": model["model_folder"],
                                     "dlc_backend_snapshot": model["snapshot"]})
        return model

    def _request(self, operation, args=(), kwargs=None):
        with tempfile.TemporaryDirectory(prefix="autobehaver_dlc_call_") as directory:
            root = Path(directory)
            request = root / "request.json"
            response = root / "response.json"
            request.write_text(json.dumps({"operation": operation, "args": list(args), "kwargs": kwargs or {}}),
                               encoding="utf-8")
            environment = os.environ.copy()
            if not sys.stderr.isatty() or environment.get('AUTOBEHAVER_PROGRESS') in ('off', 'log'):
                environment['TQDM_DISABLE'] = '1'
            completed = subprocess.run([self.executable, str(self.worker), str(request), str(response)],
                                       check=False, shell=False, env=environment)
            payload = None
            if response.is_file():
                try:
                    payload = json.loads(response.read_text(encoding="utf-8"))
                except (OSError, ValueError) as exc:
                    raise RuntimeError(f"DeepLabCut backend returned an unreadable response: {exc}") from exc
            if completed.returncode != 0:
                detail = payload.get("error") if isinstance(payload, dict) else "see backend stderr"
                raise RuntimeError(
                    f"DeepLabCut backend {operation} failed (exit {completed.returncode}): {detail}"
                )
            if not isinstance(payload, dict) or payload.get("ok") is not True:
                detail = payload.get("error") if isinstance(payload, dict) else "missing response"
                raise RuntimeError(f"DeepLabCut backend exited without a successful response: {detail}")
            return payload

    def _operation(self, name, parameters):
        def call(*args, **kwargs):
            self._request(name, args, kwargs)

        call.__name__ = name
        if parameters is not None:
            call.__signature__ = inspect.Signature([
                inspect.Parameter(item["name"], item["kind"],
                                  default=None if item["optional"] else inspect.Parameter.empty)
                for item in parameters
            ])
        return call
