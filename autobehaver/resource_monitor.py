"""Opt-in, process-tree resource accounting without importing ML frameworks."""

from __future__ import annotations

import argparse
import contextvars
import csv
import functools
import importlib.metadata
import json
import math
import os
import platform
import re
import sys
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path


SCHEMA = "autobehaver.resources.v1"
_ACTIVE = contextvars.ContextVar("autobehaver_resource_monitor", default=None)
_CATEGORICAL_SETTINGS = {
    "stage", "mode", "device", "scheduler_type", "ablation_preset",
    "frame_representation", "prediction_head", "positional_encoding_type",
    "norm_params_mode", "pretrain_loss", "tree_method", "importance_type",
    "cuda_device_name", "cuda_device_uuid",
}
_PACKAGES = (
    "autobehaver", "numpy", "pandas", "torch", "tensorflow", "deeplabcut",
    "xgboost", "scikit-learn", "psutil", "nvidia-ml-py",
)


def _utc_now():
    return datetime.now(timezone.utc).isoformat()


def _write_json(path, payload):
    temporary = path.with_suffix(".tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, allow_nan=False)
            handle.write("\n")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


class _ProcessTree:
    def __init__(self):
        import psutil

        self.psutil = psutil
        self.root = psutil.Process()
        self.started = time.time()
        self.previous = {}
        self.cpu_seconds = 0.0
        self.incomplete_samples = 0
        self.initial = True

    def sample(self):
        processes = [self.root]
        incomplete = False
        try:
            processes.extend(self.root.children(recursive=True))
        except self.psutil.Error:
            incomplete = True
        current = {}
        rss_bytes = 0
        observed = 0
        for process in processes:
            try:
                with process.oneshot():
                    created = process.create_time()
                    identity = (process.pid, created)
                    cpu_times = process.cpu_times()
                    cpu_seconds = cpu_times.user + cpu_times.system
                    memory = process.memory_info().rss
                if identity in current:
                    continue
                baseline = self.previous.get(identity)
                if baseline is None:
                    baseline = 0.0 if not self.initial and created >= self.started else cpu_seconds
                self.cpu_seconds += max(0.0, cpu_seconds - baseline)
                current[identity] = cpu_seconds
                rss_bytes += memory
                observed += 1
            except self.psutil.Error:
                incomplete = True
        self.initial = False
        if incomplete:
            self.previous.update(current)
        else:
            self.previous = current
        self.incomplete_samples += int(incomplete)
        return {
            "tree_rss_bytes": rss_bytes if observed else None,
            "tree_cpu_seconds_sampled": self.cpu_seconds,
            "observed_processes": observed,
        }

    def hardware(self):
        try:
            affinity = len(self.root.cpu_affinity())
        except (AttributeError, self.psutil.Error):
            affinity = None
        return {
            "physical_cpu_cores": self.psutil.cpu_count(logical=False),
            "ram_total_bytes": self.psutil.virtual_memory().total,
            "cpu_affinity_count": affinity,
        }


class _NvidiaDevices:
    def __init__(self):
        import pynvml

        self.nvml = pynvml
        self.devices = []
        pynvml.nvmlInit()
        try:
            for index in range(pynvml.nvmlDeviceGetCount()):
                handle = pynvml.nvmlDeviceGetHandleByIndex(index)
                name = pynvml.nvmlDeviceGetName(handle)
                identity = pynvml.nvmlDeviceGetUUID(handle)
                self.devices.append((handle, {
                    "uuid": identity.decode() if isinstance(identity, bytes) else str(identity),
                    "name": name.decode() if isinstance(name, bytes) else str(name),
                    "memory_total_bytes": pynvml.nvmlDeviceGetMemoryInfo(handle).total,
                }))
        except Exception:
            pynvml.nvmlShutdown()
            raise

    def sample(self):
        samples = []
        for handle, metadata in self.devices:
            sample = dict(metadata)
            for key, query in (
                ("device_memory_used_bytes", lambda: self.nvml.nvmlDeviceGetMemoryInfo(handle).used),
                ("device_utilization_percent", lambda: self.nvml.nvmlDeviceGetUtilizationRates(handle).gpu),
            ):
                try:
                    sample[key] = query()
                except self.nvml.NVMLError:
                    sample[key] = None
            samples.append(sample)
        return samples

    def close(self):
        self.nvml.nvmlShutdown()


class ResourceMonitor:
    """Measure a block; optional telemetry failures never replace its exception.

    RAM is sampled summed RSS, not unique physical memory. GPU observations are
    device-wide, including unrelated processes. Use ``record_work`` in the
    coordinating process only; workers are sampled, not independently logged.
    """

    def __init__(self, stage, *, enabled=False, output_dir="resource_reports",
                 sample_interval_seconds=2.0, gpu=True, samples=True):
        self.stage = str(stage)
        self.enabled = enabled
        self.output_dir = Path(output_dir)
        self.interval = float(sample_interval_seconds)
        if enabled and (not math.isfinite(self.interval) or self.interval < 0.1):
            raise ValueError("Resource sample interval must be finite and at least 0.1 seconds")
        self.gpu_enabled = gpu
        self.samples_enabled = samples
        self.run_dir = None
        self.summary = None
        self.settings = {}
        self.workload = {}
        self.validations = {}
        self.warnings = []
        self._stop = threading.Event()
        self._thread = None
        self._processes = None
        self._gpus = None
        self._csv_file = None
        self._csv_writer = None
        self._token = None
        self._owner_pid = os.getpid()
        self._peak_rss = None
        self._cpu_seconds = None
        self._sample_count = 0
        self._gpu_totals = {}

    def _warn(self, message):
        if message not in self.warnings:
            self.warnings.append(message)
            print(f"[resources] {message}", file=sys.stderr)

    def __enter__(self):
        if not self.enabled:
            return self
        self._started = time.perf_counter()
        self._parent_cpu_start = time.process_time()
        self._started_utc = _utc_now()
        try:
            safe_stage = re.sub(r"[^a-zA-Z0-9_.-]", "_", self.stage)[:80] or "stage"
            run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex
            self.run_dir = self.output_dir / f"{safe_stage}_{run_id}"
            self.run_dir.mkdir(parents=True, exist_ok=False)
        except Exception as exc:
            self._warn(f"Monitoring disabled: cannot create report directory ({type(exc).__name__}).")
            self.enabled = False
            return self
        self._hardware = {
            "platform": platform.platform(), "processor": platform.processor(),
            "logical_cpu_count": os.cpu_count(), "python": platform.python_version(),
        }
        self._versions = {}
        for package in _PACKAGES:
            try:
                self._versions[package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                self._versions[package] = None
            except Exception as exc:
                self._versions[package] = None
                self._warn(f"Version lookup unavailable for {package} ({type(exc).__name__}).")
        try:
            self._processes = _ProcessTree()
            self._hardware.update(self._processes.hardware())
        except Exception as exc:
            self._warn(f"Process-tree telemetry unavailable ({type(exc).__name__}); install the monitoring extra.")
        if self.gpu_enabled:
            try:
                self._gpus = _NvidiaDevices()
                if not self._gpus.devices:
                    self._warn("NVML reports no NVIDIA devices.")
            except Exception as exc:
                self._warn(f"NVIDIA telemetry unavailable ({type(exc).__name__}); CPU-only runs remain supported.")
        if self.samples_enabled:
            try:
                self._csv_file = (self.run_dir / "resource_samples.csv").open("x", newline="", encoding="utf-8")
                self._csv_writer = csv.DictWriter(self._csv_file, fieldnames=[
                    "elapsed_seconds", "tree_rss_bytes", "tree_cpu_seconds_sampled",
                    "observed_processes", "gpu_devices_json",
                ])
                self._csv_writer.writeheader()
            except Exception as exc:
                self._warn(f"Time-series output unavailable ({type(exc).__name__}).")
                if self._csv_file:
                    self._csv_file.close()
                self._csv_file = self._csv_writer = None
        self._sample()
        self._publish("running")
        self._token = _ACTIVE.set(self)
        self._thread = threading.Thread(target=self._sample_loop, name="autobehaver-resources", daemon=True)
        try:
            self._thread.start()
        except Exception as exc:
            self._thread = None
            self._warn(f"Periodic sampling unavailable ({type(exc).__name__}); only boundary samples recorded.")
        print(f"[resources] Reports: {self.run_dir}")
        return self

    def _sample_loop(self):
        while not self._stop.wait(self.interval):
            self._sample()

    def _sample(self):
        sample = {
            "elapsed_seconds": time.perf_counter() - self._started,
            "tree_rss_bytes": None, "tree_cpu_seconds_sampled": None,
            "observed_processes": None,
        }
        if self._processes is not None:
            try:
                sample.update(self._processes.sample())
                rss = sample["tree_rss_bytes"]
                if rss is not None:
                    self._peak_rss = max(self._peak_rss or 0, rss)
                self._cpu_seconds = sample["tree_cpu_seconds_sampled"]
            except Exception as exc:
                self._warn(f"Process-tree sample unavailable ({type(exc).__name__}).")
        gpu_samples = []
        if self._gpus is not None:
            try:
                gpu_samples = self._gpus.sample()
                for device in gpu_samples:
                    totals = self._gpu_totals.setdefault(device["uuid"], {
                        "uuid": device["uuid"], "name": device["name"],
                        "memory_total_bytes": device["memory_total_bytes"],
                        "sampled_peak_device_memory_bytes": None,
                        "utilization_sample_sum": 0.0, "utilization_sample_count": 0,
                    })
                    memory = device["device_memory_used_bytes"]
                    utilization = device["device_utilization_percent"]
                    if memory is not None:
                        totals["sampled_peak_device_memory_bytes"] = max(
                            totals["sampled_peak_device_memory_bytes"] or 0, memory)
                    if utilization is not None:
                        totals["utilization_sample_sum"] += utilization
                        totals["utilization_sample_count"] += 1
            except Exception as exc:
                self._warn(f"NVIDIA sample unavailable ({type(exc).__name__}).")
        self._sample_count += 1
        if self._csv_writer:
            try:
                self._csv_writer.writerow({**sample, "gpu_devices_json": json.dumps(gpu_samples)})
                self._csv_file.flush()
            except Exception as exc:
                self._warn(f"Time-series write failed ({type(exc).__name__}); summary still attempted.")
                self._csv_writer = None

    def record_work(self, unit, *, completed=0, skipped=0, failed=0):
        """Add observed work, never inferred from the size of an input folder."""
        if not self.enabled or os.getpid() != self._owner_pid:
            return
        values = {"completed": completed, "skipped": skipped, "failed": failed}
        if any(not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0
               for value in values.values()):
            raise ValueError("Work counts must be finite nonnegative numbers")
        counts = self.workload.setdefault(str(unit), dict.fromkeys(values, 0))
        for key, value in values.items():
            counts[key] += value

    def _publish(self, status, exception_type=None):
        elapsed = time.perf_counter() - self._started
        gpu_summary = []
        for totals in self._gpu_totals.values():
            count = totals["utilization_sample_count"]
            gpu_summary.append({
                **{key: value for key, value in totals.items() if key != "utilization_sample_sum"},
                "mean_device_utilization_percent": totals["utilization_sample_sum"] / count if count else None,
            })
        workload = {
            unit: {**counts, "completed_per_wall_second": counts["completed"] / elapsed if elapsed else None}
            for unit, counts in self.workload.items()
        }
        self.summary = {
            "schema": SCHEMA, "stage": self.stage, "status": status,
            "started_at_utc": self._started_utc,
            "finished_at_utc": None if status == "running" else _utc_now(),
            "wall_seconds": elapsed,
            "parent_cpu_seconds": max(0.0, time.process_time() - self._parent_cpu_start),
            "tree_cpu_seconds_sampled": self._cpu_seconds,
            "tree_mean_cpu_percent_one_core": 100 * self._cpu_seconds / elapsed
            if self._cpu_seconds is not None and elapsed else None,
            "sampled_peak_tree_rss_bytes": self._peak_rss,
            "sample_count": self._sample_count,
            "process_tree_incomplete_samples": self._processes.incomplete_samples if self._processes else None,
            "sample_interval_seconds": self.interval,
            "gpu_requested": self.gpu_enabled, "samples_requested": self.samples_enabled,
            "process_tree_backend": "psutil" if self._processes else None,
            "gpu_scope": "all NVML-visible devices; device-wide, NOT attributed to this run",
            "gpus": gpu_summary, "hardware": self._hardware, "versions": self._versions,
            "settings": dict(self.settings), "workload": workload,
            "validation": dict(self.validations), "exception_type": exception_type,
            "warnings": list(self.warnings),
        }
        try:
            _write_json(self.run_dir / "resource_summary.json", self.summary)
        except Exception as exc:
            self._warn(f"Summary write failed ({type(exc).__name__}).")

    def __exit__(self, exception_type, exception, traceback):
        if not self.enabled:
            return False
        try:
            self._stop.set()
            if self._thread is not None:
                self._thread.join()
            self._sample()
            if exception_type is None or (isinstance(exception, SystemExit) and exception.code in (None, 0)):
                status = "completed"
            elif isinstance(exception, KeyboardInterrupt):
                status = "interrupted"
            else:
                status = "failed"
            if status == "completed" and (any(counts["failed"] for counts in self.workload.values())
                                           or "failed" in (report["status"] for report in self.validations.values())):
                status = "partial_failure"
            self._publish(status, exception_type.__name__ if exception_type else None)
        except Exception as exc:
            self._warn(f"Monitoring finalization failed ({type(exc).__name__}).")
        finally:
            for close in (self._csv_file.close if self._csv_file else None,
                          self._gpus.close if self._gpus else None):
                if close:
                    try:
                        close()
                    except Exception as exc:
                        self._warn(f"Telemetry cleanup failed ({type(exc).__name__}).")
            if self._token is not None:
                _ACTIVE.reset(self._token)
        return False


def current_monitor():
    monitor = _ACTIVE.get()
    return monitor if monitor is not None and monitor._owner_pid == os.getpid() else None


def record_work(unit, *, completed=0, skipped=0, failed=0):
    """Record parent-side completion callbacks when a monitor is active."""
    monitor = current_monitor()
    if monitor is not None:
        monitor.record_work(unit, completed=completed, skipped=skipped, failed=failed)


def record_settings(settings):
    """Record numeric/bool settings and allowlisted categories, not paths/secrets."""
    monitor = current_monitor()
    if monitor is None:
        return
    values = settings if isinstance(settings, dict) else vars(settings)
    for key, value in values.items():
        if key.startswith("resource_monitor") or re.search(r"secret|token|password|credential|api_key", key, re.I):
            continue
        if isinstance(value, (bool, int)) or (isinstance(value, float) and math.isfinite(value)):
            monitor.settings[key] = value
        elif key in _CATEGORICAL_SETTINGS and (value is None or isinstance(value, str)):
            monitor.settings[key] = value


def record_validation(report):
    """Attach integrity counts separately from compute/throughput counters."""
    monitor = current_monitor()
    if monitor is not None:
        monitor.validations[report.stage] = {
            "status": report.status, "expected_count": len(report.expected_items),
            "outcome_counts": report.outcome_counts(),
        }


def add_monitor_arguments(parser):
    group = parser.add_argument_group("Resource monitoring (optional)")
    group.add_argument("--resource-monitor", action=argparse.BooleanOptionalAction, default=None,
                       help="Record wall time, CPU/RAM and optional NVIDIA telemetry (default: off).")
    group.add_argument("--resource-monitor-output", default=None, metavar="DIR",
                       help="Report root; creates a unique subdirectory per stage invocation.")
    group.add_argument("--resource-monitor-interval", type=float, default=None, metavar="SECONDS",
                       help="Sampling interval, at least 0.1 seconds (default: 2).")
    group.add_argument("--resource-monitor-gpu", action=argparse.BooleanOptionalAction, default=None,
                       help="Query device-wide NVIDIA NVML telemetry (default: on when monitoring).")
    group.add_argument("--resource-monitor-samples", action=argparse.BooleanOptionalAction, default=None,
                       help="Stream a samples CSV as well as the summary (default: on).")
    return parser


def _monitor_options(argv):
    from .argparse_utils import normalize_argv
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    add_monitor_arguments(parser)
    arguments, _ = parser.parse_known_args(normalize_argv(argv, parser=parser))
    if arguments.resource_monitor is False:
        return {"enabled": False}
    options = {
        "enabled": False, "output_dir": "resource_reports",
        "sample_interval_seconds": 2.0, "gpu": True, "samples": True,
    }
    for key, attribute in (
        ("enabled", "resource_monitor"), ("output_dir", "resource_monitor_output"),
        ("sample_interval_seconds", "resource_monitor_interval"),
        ("gpu", "resource_monitor_gpu"), ("samples", "resource_monitor_samples"),
    ):
        override = getattr(arguments, attribute)
        if override is not None:
            options[key] = override
    for key in ("enabled", "gpu", "samples"):
        if not isinstance(options[key], bool):
            parser.error(f"resource_monitor.{key} must be a boolean")
    if not options["enabled"]:
        return {"enabled": False}
    try:
        interval = float(options["sample_interval_seconds"])
        if not math.isfinite(interval) or interval < 0.1:
            raise ValueError
        options["sample_interval_seconds"] = interval
        output = Path(options["output_dir"]).expanduser()
        if not output.is_absolute():
            output = Path.cwd() / output
        options["output_dir"] = output
    except (TypeError, ValueError):
        parser.error("Monitoring requires a valid output directory and finite interval >= 0.1 seconds")
    return options


def monitored_cli(stage):
    """Wrap a stage CLI without changing argv, exit codes, or science defaults."""
    def decorate(main):
        @functools.wraps(main)
        def wrapped(argv=None):
            arguments = list(sys.argv[1:] if argv is None else argv)
            if "--help" in arguments or "-h" in arguments:
                return main(arguments)
            options = _monitor_options(arguments)
            if not options["enabled"]:
                return main(arguments)
            with ResourceMonitor(stage, **options):
                return main(arguments)
        return wrapped
    return decorate


def export_summaries(root, output):
    """Export one row per run, without summing overlapping jobs or overwriting."""
    fields = [
        "report", "stage", "status", "started_at_utc", "finished_at_utc", "wall_seconds",
        "parent_cpu_seconds", "tree_cpu_seconds_sampled", "tree_mean_cpu_percent_one_core",
        "sampled_peak_tree_rss_bytes", "sample_count", "sample_interval_seconds",
        "gpu_scope", "hardware", "versions", "gpus", "settings", "workload", "validation", "warnings",
    ]
    count = 0
    with Path(output).open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for path in sorted(Path(root).rglob("resource_summary.json")):
            with path.open(encoding="utf-8") as source:
                summary = json.load(source)
            if summary.get("schema") != SCHEMA:
                continue
            row = {key: summary.get(key) for key in fields}
            row["report"] = str(path)
            for key, value in row.items():
                if isinstance(value, (dict, list)):
                    row[key] = json.dumps(value, sort_keys=True)
            writer.writerow(row)
            count += 1
    return count


def main(argv=None):
    parser = argparse.ArgumentParser(description="Export Autobehaver resource summaries; does not run the pipeline.")
    parser.add_argument("root", help="Root containing per-run resource summaries")
    parser.add_argument("--output", required=True, help="New combined CSV path (must not already exist)")
    args = parser.parse_args(argv)
    if not Path(args.root).is_dir():
        parser.error("Report root must be an existing directory")
    print(f"Exported {export_summaries(args.root, args.output)} run(s) to {args.output}")


if __name__ == "__main__":
    main()
