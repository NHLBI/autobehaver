import argparse
import ast
import contextlib
import csv
import json
import sys
import threading
import types
from pathlib import Path

import pytest

from autobehaver import resource_monitor as resources


class FakeProcesses:
    incomplete_samples = 0

    def __init__(self):
        self.samples = 0

    def hardware(self):
        return {"ram_total_bytes": 8192, "physical_cpu_cores": 4}

    def sample(self):
        self.samples += 1
        return {"tree_rss_bytes": self.samples * 1024,
                "tree_cpu_seconds_sampled": 0.01 * (self.samples - 1),
                "observed_processes": 2}


class FakeDevices:
    devices = ["mock"]
    closed = False

    def sample(self):
        return [{"uuid": "GPU-mock", "name": "Mock GPU", "memory_total_bytes": 4096,
                 "device_memory_used_bytes": 1024, "device_utilization_percent": 25}]

    def close(self):
        self.closed = True


@pytest.fixture
def fake_backends(monkeypatch):
    monkeypatch.setattr(resources, "_ProcessTree", FakeProcesses)
    monkeypatch.setattr(resources, "_NvidiaDevices", FakeDevices)
    monkeypatch.setattr(resources.importlib.metadata, "version", lambda package: "test-version")


def read_summary(monitor):
    return json.loads((monitor.run_dir / "resource_summary.json").read_text(encoding="utf-8"))


def test_disabled_monitor_is_inert(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Disabled monitoring must not initialize telemetry or threads")

    monkeypatch.setattr(resources, "_ProcessTree", forbidden)
    monkeypatch.setattr(resources, "_NvidiaDevices", forbidden)
    monkeypatch.setattr(resources.threading, "Thread", forbidden)
    monkeypatch.setattr(resources.importlib.metadata, "version", forbidden)
    with resources.ResourceMonitor("disabled", output_dir=tmp_path / "unused") as monitor:
        resources.record_work("files", completed=5)
        monitor.record_work("files", completed=5)
    assert not (tmp_path / "unused").exists()
    assert monitor.summary is None
    assert resources.current_monitor() is None


def test_reports_workload_samples_settings_and_cleanup(tmp_path, fake_backends):
    with resources.ResourceMonitor("example", enabled=True, output_dir=tmp_path) as monitor:
        resources.record_work("files", completed=2, skipped=8, failed=1)
        resources.record_settings({"batch": 32, "sequence_length": 41, "mode": "inference",
                                   "api_key": "private", "directory": "private-path",
                                   "access_token": 1234, "resource_monitor": True})
    summary = read_summary(monitor)
    assert summary["status"] == "partial_failure"
    assert summary["sampled_peak_tree_rss_bytes"] == 2048
    assert summary["sample_count"] == 2
    assert summary["workload"]["files"]["completed_per_wall_second"] == pytest.approx(2 / summary["wall_seconds"])
    assert summary["settings"] == {"batch": 32, "sequence_length": 41, "mode": "inference"}
    assert summary["gpus"][0]["mean_device_utilization_percent"] == 25
    assert "NOT attributed" in summary["gpu_scope"]
    assert monitor._gpus.closed
    assert not monitor._thread.is_alive()
    assert resources.current_monitor() is None
    with (monitor.run_dir / "resource_samples.csv").open(encoding="utf-8", newline="") as handle:
        samples = list(csv.DictReader(handle))
    assert len(samples) == 2
    assert samples[0]["observed_processes"] == "2"


@pytest.mark.parametrize("exception,status", [
    (RuntimeError("private exception message"), "failed"),
    (KeyboardInterrupt(), "interrupted"),
    (SystemExit(0), "completed"),
    (SystemExit(3), "failed"),
])
def test_exception_preserved_and_sampler_stopped(tmp_path, fake_backends, exception, status):
    monitor = resources.ResourceMonitor("failure", enabled=True, output_dir=tmp_path)
    with pytest.raises(type(exception)) as captured:
        with monitor:
            raise exception
    assert captured.value is exception
    assert read_summary(monitor)["status"] == status
    assert "private exception message" not in json.dumps(monitor.summary)
    assert not monitor._thread.is_alive()
    assert monitor._gpus.closed
    assert resources.current_monitor() is None


def test_optional_backends_missing_give_null_not_zero(tmp_path, monkeypatch):
    def unavailable():
        raise ImportError("not installed")

    monkeypatch.setattr(resources, "_ProcessTree", unavailable)
    monkeypatch.setattr(resources, "_NvidiaDevices", unavailable)
    with resources.ResourceMonitor("minimal", enabled=True, output_dir=tmp_path) as monitor:
        pass
    summary = read_summary(monitor)
    assert summary["status"] == "completed"
    assert summary["sampled_peak_tree_rss_bytes"] is None
    assert summary["tree_cpu_seconds_sampled"] is None
    assert summary["gpus"] == []
    assert len(summary["warnings"]) == 2
    assert summary["parent_cpu_seconds"] >= 0


def test_no_gpu_or_samples_when_disabled(tmp_path, fake_backends, monkeypatch):
    def forbidden():
        pytest.fail("GPU monitoring is disabled")

    monkeypatch.setattr(resources, "_NvidiaDevices", forbidden)
    with resources.ResourceMonitor("cpu", enabled=True, output_dir=tmp_path, gpu=False, samples=False) as monitor:
        pass
    assert not (monitor.run_dir / "resource_samples.csv").exists()
    assert read_summary(monitor)["gpus"] == []


def test_periodic_sampler_stops_promptly(tmp_path, fake_backends):
    sampled = threading.Event()

    class NotifyingProcesses(FakeProcesses):
        def sample(self):
            result = super().sample()
            if self.samples >= 2:
                sampled.set()
            return result

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(resources, "_ProcessTree", NotifyingProcesses)
        with resources.ResourceMonitor("periodic", enabled=True, output_dir=tmp_path,
                                       sample_interval_seconds=0.1) as monitor:
            assert sampled.wait(timeout=3)
    assert read_summary(monitor)["sample_count"] >= 3
    assert not monitor._thread.is_alive()


def test_nested_runs_and_export_never_overwrite(tmp_path, fake_backends):
    with resources.ResourceMonitor("same", enabled=True, output_dir=tmp_path) as outer:
        with resources.ResourceMonitor("same", enabled=True, output_dir=tmp_path) as inner:
            resources.record_work("files", completed=1)
        assert resources.current_monitor() is outer
        resources.record_work("files", completed=3)
    assert outer.run_dir != inner.run_dir
    output = tmp_path / "combined.csv"
    assert resources.export_summaries(tmp_path, output) == 2
    with output.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 2
    assert sorted(json.loads(row["workload"])["files"]["completed"] for row in rows) == [1, 3]
    with pytest.raises(FileExistsError):
        resources.export_summaries(tmp_path, output)


def test_write_failure_does_not_replace_pipeline_exception(tmp_path, fake_backends, monkeypatch):
    def fail_write(*args):
        raise OSError("disk full")

    monkeypatch.setattr(resources, "_write_json", fail_write)
    monitor = resources.ResourceMonitor("disk-full", enabled=True, output_dir=tmp_path)
    with pytest.raises(ValueError, match="science failure"):
        with monitor:
            raise ValueError("science failure")
    assert not monitor._thread.is_alive()
    assert resources.current_monitor() is None
    assert any("Summary write failed" in warning for warning in monitor.warnings)


def test_unwritable_output_does_not_stop_work(tmp_path, fake_backends):
    blocked = tmp_path / "file-not-directory"
    blocked.write_text("existing")
    with resources.ResourceMonitor("blocked", enabled=True, output_dir=blocked) as monitor:
        assert not monitor.enabled
    assert blocked.read_text() == "existing"


def test_validation_is_separate_from_processed_work(tmp_path, fake_backends):
    report = types.SimpleNamespace(stage="keypre", status="passed", expected_items=["cached"],
                                   outcome_counts=lambda: {"succeeded": 0, "reused": 1})
    with resources.ResourceMonitor("keypre", enabled=True, output_dir=tmp_path) as monitor:
        resources.record_validation(report)
    summary = read_summary(monitor)
    assert summary["workload"] == {}
    assert summary["validation"]["keypre"]["outcome_counts"]["reused"] == 1


def test_forked_worker_cannot_increment_parent_counters(tmp_path, fake_backends, monkeypatch):
    with resources.ResourceMonitor("parent", enabled=True, output_dir=tmp_path) as monitor:
        with monkeypatch.context() as patch:
            patch.setattr(resources.os, "getpid", lambda: monitor._owner_pid + 1)
            resources.record_work("files", completed=500)
            assert resources.current_monitor() is None
    assert monitor.summary["workload"] == {}


def test_direct_monitor_options_and_paths(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    options = resources._monitor_options(["--resource-monitor", "--resource-monitor-output", "reports",
                                          "--no-resource-monitor-gpu", "--resource-monitor-interval", "5"])
    assert options["output_dir"] == tmp_path / "reports"
    assert options["gpu"] is False
    options = resources._monitor_options(["--resource-monitor", "--resource-monitor-output", "cli-reports",
                                          "--resource-monitor-interval", "1", "--resource-monitor-gpu"])
    assert options["output_dir"] == tmp_path / "cli-reports"
    assert options["sample_interval_seconds"] == 1
    assert options["gpu"] is True
    assert resources._monitor_options(["--batch", "9", "--no-resource-monitor"]) == {"enabled": False}


@pytest.mark.parametrize("value", ["0", "-1", "nan", "inf", "0.05"])
def test_invalid_sampling_interval_rejected(value):
    with pytest.raises(SystemExit):
        resources._monitor_options(["--resource-monitor", "--resource-monitor-interval", value])


def test_monitor_boolean_values_require_explicit_flags():
    parser = resources.add_monitor_arguments(argparse.ArgumentParser())
    with pytest.raises(SystemExit):
        parser.parse_args(["--resource-monitor", "false"])
    assert resources._monitor_options([]) == {"enabled": False}
    assert resources._monitor_options(["--resource_monitor", "--no_resource_monitor_gpu"])["gpu"] is False


def test_wrapper_help_and_disabled_do_not_create_reports(tmp_path, fake_backends, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)

    @resources.monitored_cli("tiny")
    def tiny_main(argv=None):
        parser = resources.add_monitor_arguments(argparse.ArgumentParser())
        parser.add_argument("--batch", type=int, default=1)
        args = parser.parse_args(argv)
        resources.record_settings(args)
        return args.batch

    with pytest.raises(SystemExit) as captured:
        tiny_main(["--help", "--resource-monitor"])
    assert captured.value.code == 0
    assert "--resource-monitor" in capsys.readouterr().out
    assert tiny_main(["--batch", "7"]) == 7
    assert not list(tmp_path.iterdir())
    assert tiny_main(["--resource-monitor", "--batch", "9"]) == 9
    reports = list(tmp_path.rglob("resource_summary.json"))
    assert len(reports) == 1
    assert json.loads(reports[0].read_text())["settings"]["batch"] == 9


def test_process_tree_accounts_for_workers_and_pid_reuse(monkeypatch):
    class Process:
        def __init__(self, pid, created, cpu, rss):
            self.pid, self.created, self.cpu, self.rss = pid, created, cpu, rss
            self.descendants = []

        def oneshot(self):
            return contextlib.nullcontext()

        def children(self, recursive):
            assert recursive
            return self.descendants

        def create_time(self):
            return self.created

        def cpu_times(self):
            return types.SimpleNamespace(user=self.cpu, system=0)

        def memory_info(self):
            return types.SimpleNamespace(rss=self.rss)

    root = Process(1, 1, 20, 100)
    worker = Process(2, 2, 30, 200)
    root.descendants = [worker]
    fake_psutil = types.SimpleNamespace(Process=lambda: root, Error=OSError)
    monkeypatch.setitem(sys.modules, "psutil", fake_psutil)
    tree = resources._ProcessTree()
    tree.started = 5
    initial = tree.sample()
    assert initial["tree_cpu_seconds_sampled"] == 0
    assert initial["tree_rss_bytes"] == 300
    root.cpu += 1
    worker.cpu += 2
    assert tree.sample()["tree_cpu_seconds_sampled"] == 3
    root.descendants = [Process(2, 10, 4, 250)]
    assert tree.sample()["tree_cpu_seconds_sampled"] == 7
    root.descendants = []
    assert tree.sample()["tree_cpu_seconds_sampled"] == 7


def test_nvml_queries_are_device_wide_and_allow_unsupported_metrics(monkeypatch):
    class NvmlError(Exception):
        pass

    calls = []

    def unavailable(handle):
        raise NvmlError()

    fake_nvml = types.SimpleNamespace(
        NVMLError=NvmlError, nvmlInit=lambda: calls.append("init"),
        nvmlShutdown=lambda: calls.append("shutdown"), nvmlDeviceGetCount=lambda: 1,
        nvmlDeviceGetHandleByIndex=lambda index: "handle",
        nvmlDeviceGetName=lambda handle: b"Mock", nvmlDeviceGetUUID=lambda handle: b"GPU-mock",
        nvmlDeviceGetMemoryInfo=lambda handle: types.SimpleNamespace(total=4096, used=1024),
        nvmlDeviceGetUtilizationRates=unavailable,
    )
    monkeypatch.setitem(sys.modules, "pynvml", fake_nvml)
    backend = resources._NvidiaDevices()
    sample = backend.sample()[0]
    backend.close()
    assert sample["name"] == "Mock"
    assert sample["device_memory_used_bytes"] == 1024
    assert sample["device_utilization_percent"] is None
    assert calls == ["init", "shutdown"]


@pytest.mark.parametrize("relative_path", [
    "cli.py", "preinf/moviechecker/cli.py", "preinf/autocropper/cli.py",
    "preinf/dlc/cli.py", "preinf/keypre/cli.py", "preinf/keyprenorm/cli.py",
    "preinf/posepred_inference/cli.py", "postinf/mmgen/cli.py",
    "postinf/infmovie/cli.py", "postinf/featureimp/cli.py",
])
def test_all_stage_clis_have_common_hooks_without_importing_frameworks(relative_path):
    source = Path(__file__).resolve().parents[1] / "autobehaver" / relative_path
    tree = ast.parse(source.read_text(encoding="utf-8"))
    main = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main")
    assert any(isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
               and node.func.id == "monitored_cli" for node in main.decorator_list)
    calls = {node.func.id for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)}
    assert {"add_monitor_arguments", "record_settings"} <= calls


def test_keyprenorm_direct_cli_enables_monitor_without_processing(tmp_path, fake_backends, monkeypatch):
    from autobehaver.preinf.keyprenorm import cli

    captured = []

    def fake_run(*, cfg, args):
        captured.append(cfg)
        assert resources.current_monitor() is not None
        resources.record_work("normalized_files", completed=2, skipped=5)

    monkeypatch.setattr(cli, "run_keyprenorm", fake_run)
    cli.main(["--input_dir", str(tmp_path), "--num_workers", "1", "--edge_skip", "7",
              "--resource-monitor", "--resource-monitor-output", str(tmp_path / 'reports'),
              "--resource-monitor-interval", "3"])
    assert captured[0].edge_skip == 7
    reports = list((tmp_path / "reports").rglob("resource_summary.json"))
    assert len(reports) == 1
    summary = json.loads(reports[0].read_text())
    assert summary["sample_interval_seconds"] == 3
    assert summary["settings"]["num_workers"] == 1
    assert summary["workload"]["normalized_files"]["skipped"] == 5


@pytest.mark.parametrize("module_name, builder", [
    ("autobehaver.cli", "build_parser"),
    ("autobehaver.preinf.moviechecker.cli", "build_arg_parser"),
    ("autobehaver.preinf.autocropper.cli", "build_arg_parser"),
    ("autobehaver.preinf.dlc.cli", "build_arg_parser"),
    ("autobehaver.preinf.keypre.cli", "build_arg_parser"),
    ("autobehaver.preinf.keyprenorm.cli", "build_arg_parser"),
    ("autobehaver.preinf.posepred_inference.cli", "build_arg_parser"),
    ("autobehaver.postinf.mmgen.cli", "build_parser"),
    ("autobehaver.postinf.infmovie.cli", "build_parser"),
])
def test_actual_parsers_accept_common_monitor_flags(module_name, builder):
    import importlib

    if module_name == "autobehaver.preinf.moviechecker.cli":
        pytest.importorskip("cv2", reason="MovieChecker imports its existing OpenCV runtime")
    if module_name == "autobehaver.preinf.posepred_inference.cli":
        pytest.importorskip("torch", reason="PosePred imports its existing Torch runtime")
    module = importlib.import_module(module_name)
    parser = getattr(module, builder)()
    args, _ = parser.parse_known_args(["--resource-monitor", "--no-resource-monitor-gpu"])
    assert args.resource_monitor is True
    assert args.resource_monitor_gpu is False
    assert "--resource-monitor-output" in parser.format_help()


def test_featureimp_help_does_not_import_core_or_start_monitor(monkeypatch, tmp_path, capsys):
    from autobehaver.postinf.featureimp import cli

    monkeypatch.chdir(tmp_path)

    def forbidden():
        pytest.fail("Help must not import XGBoost or start GPU sampling")

    monkeypatch.setattr(cli, "_import_core", forbidden)
    monkeypatch.setattr(resources, "_NvidiaDevices", forbidden)
    with pytest.raises(SystemExit) as captured:
        cli.main(["--help", "--resource-monitor"])
    assert captured.value.code == 0
    assert "--resource-monitor" in capsys.readouterr().out
    assert not list(tmp_path.iterdir())
