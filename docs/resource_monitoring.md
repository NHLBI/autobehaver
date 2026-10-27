# Runtime and resource monitoring

Enable `--resource-monitor` to record elapsed time and available CPU, memory and
NVIDIA GPU measurements. Monitoring is off by default and adds a sampling thread
and report writes; measure its overhead before using it for benchmarks.

## Usage

Install the optional reporting dependencies in the stage's environment:

```bash
python -m pip install -e ".[monitoring]"
python environment/check_environment.py --require-monitoring
```

`psutil` supplies process-tree CPU/memory measurements; `nvidia-ml-py` supplies
GPU telemetry through the NVIDIA driver. Missing telemetry is reported as
unavailable. The monitor does not initialize a model or GPU workload.

The main processing-stage commands, including training and FeatureImp's staged
modes, accept:

```text
--resource-monitor
--resource-monitor-output <directory>
--resource-monitor-interval <seconds>
--no-resource-monitor-gpu
--no-resource-monitor-samples
```

For example, add monitoring to a KeyPre run:

```bash
python -m autobehaver.preinf.keypre --main_dir /path/to/project --resource-monitor
```

The minimum sampling interval is 0.1 seconds; the default is 2 seconds. Reports
default to `resource_reports` relative to the current directory. Each invocation
uses a separate stage/timestamp/UUID folder. `--no-resource-monitor` disables it.

## Reports and interpretation

`resource_summary.json` records status, timing, hardware/software metadata,
sampled measurements, workload counters and available integrity counts.
`resource_samples.csv` contains optional time-series samples.

| Measurement | Interpretation |
| --- | --- |
| `wall_seconds` | Stage time including monitoring setup, loading, computation, validation, publication and final sampling; excludes interpreter/module startup and final report writing/cleanup. |
| `parent_cpu_seconds` | CPU time for the coordinating process and its threads. |
| `tree_cpu_seconds_sampled` | Observed CPU-time increments for the process tree; short-lived workers may be missed. |
| `tree_mean_cpu_percent_one_core` | Sampled CPU use with one occupied core equal to 100%; parallel work may exceed 100%. |
| `sampled_peak_tree_rss_bytes` | Largest sampled sum of process memory; shared pages may be counted more than once and between-sample peaks may be missed. |
| GPU memory/utilization | Device-wide measurements, including other processes; unavailable values are null. |
| Workload throughput | Newly completed units divided by full stage wall time, including time spent on failures. |

Summaries start as `running` and are finalized on normal return or handled
exceptions. Failed workload/integrity counts produce `partial_failure`. A hard
termination can leave an unfinished report; check status before analysis. Report
write failures warn without stopping the scientific stage.

Workload units follow observed completion events:

| Stage | Recorded units |
| --- | --- |
| DeepLabCut | Validated videos with newly changed outputs and their output frames; unchanged/reused outputs are skipped. |
| KeyPre | Completed, failed and skipped input CSV files. |
| KeyPreNorm | Written normalization files, failed inputs and reused files; parameter-only runs have no file-normalization throughput. |
| PosePred inference | Published output files/rows and skipped files; rows do not count ensemble or augmentation passes. |
| Supervised training | Training sequences and train/validation epochs; validation sequences are not training throughput. |
| Pretraining | Completed epochs, including completion of a partially resumed epoch. |
| FeatureImp | Completed/failed subset tasks; a task may include several classifier fits. |
| MovieChecker, Autocropper, MMGen, InfMovie | Timing and settings, without an inferred workload count. |

Integrity counts describe artifact validation, separately from compute work.
The summary records selected settings; keep the full resolved stage settings,
manifests and model/input identities with benchmark results.

## Library calls and export

Wrap a custom stage in the coordinating process:

```python
from autobehaver.resource_monitor import ResourceMonitor

with ResourceMonitor("custom_stage", enabled=True, output_dir="resource_reports") as monitor:
    result = process_one_movie()
    monitor.record_work("movies", completed=1)
```

Use `record_settings(config)` inside a library-call context when needed. A monitor
observes its process tree; remote jobs or daemon-launched containers need a monitor
inside their own allocation/container. Asynchronous GPU work must be synchronized
by the caller before timing ends.

Export reports to a new CSV file:

```bash
python -m autobehaver.resource_monitor resource_reports --output benchmark_runs.csv
```

Each report becomes one row with structured fields retained as JSON cells. The
exporter does not combine overlapping stages into a whole-pipeline duration.

## Maintained HPC submissions

The standard HPC templates enable native resource monitoring for processing
jobs and capture each command with GNU `time`. Reports remain in the prepared
job bundle rather than the dataset's numbered output folders. CPU allocations
request `e7543`; DLC and PosePred request A100 GPUs. DLC's scientific API runs in
its separate existing environment while the coordinating main process records
stage timing and available process-tree measurements.

After processing jobs terminate, including failures, the collector exports
`resource_summaries_<invocation>.csv` and Slurm accounting in
`scheduler_report_<invocation>.psv`. The collector prints both exact paths;
retries preserve previous reports. Per-command
`*_time.txt` files remain under `logs/`. Missing optional telemetry remains
unavailable; it does not invalidate the scheduler or GNU `time` records. Confirm
stage outcomes, reused work and recorded hardware before deriving benchmarks.
Fast batching controls input counts and job caps; it does not guarantee a
particular job duration.

## Benchmarking

Fix the input inventory, model identities, stage settings and hardware/software
versions. Distinguish fresh runs from resumed/reused outputs and cold from warm
caches. Include stage loading and publication, report queue/setup/transfer time
separately, and repeat representative cases on an otherwise idle machine. Measure
monitoring overhead with the same workload and sampling interval. Report useful
throughput and variability alongside sampled memory; GPU values require an idle
device for attribution to the measured run.
