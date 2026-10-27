# HPC submission (Slurm)

The Python processing commands can run locally or on a compute cluster. The
prepared submission workflow uses Slurm (`sbatch`, `squeue`, and `sacct`),
Bash, and Linux `flock`; it does not provide PBS or LSF submission.

Before using the supplied templates on another cluster, adapt the partition
names, CPU constraints, GPU resource syntax, scratch allocation/path, and DLC
environment activation to that site. These settings are in `scripts/hpc/*.sbatch`;
`submit.sh` also contains the Slow DLC time/scratch override. The supplied
presets retain the development cluster's working allocations and are not
universal HPC defaults. Use your own hostname, storage paths, Python environment,
and model paths. Existing prepared bundles retain their frozen settings.

Prepare the pipeline on your desktop, then submit it with one Bash command from
your existing HPC connection. Every stage uses the same dataset root and its
normal numbered output folders.

Preparation runs on Windows or Linux. Submission runs on a Linux HPC cluster with Slurm and uses
the standard `flock` utility to prevent overlapping submission commands for the
same prepared bundle. The lock file stays in job storage; ownership ends when
the submitting process exits.

One submission schedules Autocropper, DLC, KeyPre plus normalization, PosePred,
MMGen processing and final outputs, then benchmark collection. MovieChecker review
precedes preparation by default; Fast preparation can instead explicitly use only
recording-time occupancy metadata. FeatureImp and annotated-movie rendering are
separate steps.

## Fast and Slow

`--speed fast` is the default. It balances inputs across useful allocations with
shared minimum inputs per job and maximum parallel processing jobs per stage:

| Setting | Default |
| --- | --- |
| Full movies per crop job | 2 |
| Chamber recordings per later-stage job | 20 |
| CPU jobs per stage | 50 |
| GPU jobs per stage (DLC and PosePred) | 50 |

Set all four using `--batching FULL_MOVIES CHAMBERS CPU_JOBS GPU_JOBS`.
For N inputs and minimum M, the builder uses at most max(1, floor(N/M)) jobs,
limited by the relevant CPU/GPU cap, then balances the inputs. A dataset smaller
than M uses one job. These settings do not change concurrent workers inside jobs.
MMGen keeps animal recordings together, so its target is approximate. Final merge
and accounting jobs are additional dependent jobs, not concurrent process shards.
Caps do not impose an account-wide running limit: stages and projects can overlap.
Defaults are preferences, not detected HPC account limits or runtime guarantees.

The older `--batch-sizes CROP DLC FEATURES POSEPRED MMGEN` interface still uses
explicit stage sizes for compatibility, without shared balancing/caps. It cannot
be combined with `--batching`. Prefer the shared settings for new schedules.

A batch advances when its own required upstream jobs succeed. MMGen waits for
all predictions, processes independent animal partitions in parallel, then runs
one final merge and the normal summary outputs. Multiple recordings of the same
animal stay together. Queue availability affects the elapsed time for a whole
stage. Input counts do not predict a fixed elapsed time.

`--speed slow` runs the entire dataset in one job per stage: cropping, DLC,
KeyPre plus normalization, PosePred and MMGen, followed by benchmark collection.

## Prepare and publish through a mounted drive

For transferred DLC coordinates, use `--start-at features --local-project PATH`.
Both Fast and Slow modes inventory `3_dlc` (or its legacy `DLC` subfolder),
validate coordinate headers and unique recording identities, and schedule only
KeyPre/normalization, PosePred, MMGen and accounting. Keep filename maps and saved
chamber selections with the CSVs. Original/cropped movies and a new MovieChecker
review are not required for this start. Prediction CSVs are not DLC coordinates.
Feature-stage templates request ten CPUs and 16 GiB. KeyPre and normalization
select their own concurrency and bounded chunk sizes within that allocation,
leaving 25% RAM headroom (at least 1 GiB). The scripts pass their RAM request
explicitly; detected Slurm/cgroup limits can reduce it. Batch size remains separate.
See [preprocessing memory](PREPROCESSING_MEMORY.md) for CLI controls and limits.

Complete MovieChecker review first for the default `reviewed` mode. Fast preparation scans `1_full_movies`, reads
its `exclusions.csv`, and uses recording metadata from `rec_data/trials` (or the
movie folder when that metadata directory is absent). Fully excluded cameras
receive no jobs. A chamber is included only when both the recording occupancy and
MovieChecker selection include it. The planner derives the chamber identities and
generates all batch lists; no manually supplied movie list is required.
It reads through `--local-project`, the desktop view of the dataset.
It does not decode movies, copy raw data, or run scientific processing.

For new recordings whose occupancy metadata is trusted, use
`--moviechecker-mode recording-only` to omit MovieChecker review. This mode
requires Fast preparation and recording metadata for every selected camera.
It uses recording-time occupancy alone and ignores even an existing
`exclusions.csv`; visually discovered recording faults therefore will not be
excluded. The choice is saved in `plan.json` and `review_mode.txt`, and the
crop job uses the same choice. Reviewed mode remains the default.

To test a few full movies in the ordinary dataset folders, add
`--only-original-movies MOVIE1.mp4 MOVIE2.mp4` to Fast preparation. These must be
exact basenames in `1_full_movies`. The planner checks their review/metadata
coverage, then schedules only their chambers through MMGen in the same numbered
dataset folders. It creates no trial or combined-input dataset tree. The
selection is recorded in `plan.json`; omit it for the full dataset.

For example, in PowerShell from the package directory:

```powershell
python scripts/hpc/prepare.py `
  --speed fast `
  --local-project 'Z:\autobehaver\example_project' `
  --project /data/example_lab/autobehaver/example_project `
  --output-dir 'C:\job-preparation\example_run' `
  --remote-job-dir /data/example_lab/autobehaver/_job_bundles/example_run `
  --publish-to 'Z:\autobehaver\_job_bundles\example_run' `
  --python /data/example_user/autobehaver/bin/python `
  --resources EXISTING_RESOURCES `
  --inference-model MODEL16 MODEL32 MODEL64
```

Choose new local and mounted job-folder names for each preparation. `--publish-to`
automatically copies the prepared schedule, templates and a small snapshot of the
current source package to the mounted folder. Jobs use that source snapshot.
Publication checks space and verifies source/copy hashes, excludes model resources
and refuses to overwrite an existing bundle. `--publish-to` and `--remote-job-dir`
must name the same folder through their Windows and Linux views. Models,
normalization parameters and Python environments retain their existing HPC paths.
Omit `--inference-model` to use the three existing checkpoints under
`RESOURCES/posepred`; supplying three absolute checkpoint paths avoids model copies.

Preparation prints each stage's job count and the final submission command.
`plan.json` records the selected preset, batch sizes, recordings and dependencies.
`schedule.tsv` and `batches/` contain the exact schedule and input lists. These
files live in job storage, outside the dataset's numbered processing folders.

## Submit

Paste the printed command in your existing HPC terminal:

```bash
bash /data/example_lab/autobehaver/_job_bundles/example_run/submit.sh
```

The login-node script reads the small prepared schedule, calls `sbatch` and records
accepted job IDs immediately in `jobs.tsv`. It does not scan movies, load models
or run preparation/processing. Accepted submissions are preserved if a later
submission fails. Use `--resume` to continue that submission after checking
its reported error.
Benchmark collection runs after processing jobs terminate, including failures.
The collector writes `scheduler_report_*.psv` and `resource_summaries_*.csv`,
with the exact invocation-specific paths printed in its log. Per-job
logs, GNU `time` reports and native resource reports stay in the prepared bundle.
The launcher spaces separate `sbatch` calls by at least one second, including
new jobs on `--resume`, to limit scheduler request rates. A large Fast
schedule can therefore take several minutes to submit; job IDs are recorded
incrementally in `jobs.tsv`.
Unavailable optional CPU/GPU telemetry is reported rather than treated as a
scientific-stage failure; see [resource monitoring](resource_monitoring.md).

If automatic publication is unwanted, omit `--publish-to` and provide
`--package EXISTING_HPC_SOURCE`. Copy the prepared job folder to its matching
remote path before submission, including the empty logs/resources directories.
That existing source must support this prepared schedule and the batch-list CLI.
Manual copying and `--publish-to` are alternative transfer routes; publication
does not connect over SSH or submit jobs.

Use `--start-at dlc` only when `2_cropped_movies` already contains the intended
inputs. Fast inventories those included crops; Slow processes them in one job.

The templates use `e7543` CPU allocations and A100 GPUs. DLC uses the existing
DeepLabCut/2.3.9 module through `python-DLC` while Autobehaver stays in the main
environment. Scratch is requested only for DLC. Slow DLC receives a larger time
and scratch allowance for its full-dataset workload. Native run records, filename
maps and scientific validation remain with stage outputs; scheduler accounting,
timing and resource reports remain in job storage.

Default pipeline allocations are:

| Stage | CPUs | RAM (GiB) | Time limit |
| --- | ---: | ---: | --- |
| Autocropper | 8 | 8 | 18 hours |
| DLC | 6 | 16 | 24 hours |
| KeyPre + normalization | 10 | 16 | 18 hours |
| PosePred | 12 | 24 | 24 hours |
| MMGen shards | 12 | 32 | 18 hours |
| MMGen finalization | 12 | 80 | 48 hours |
| Benchmark collection | 2 | 4 | 15 minutes |

Single-job MMGen uses the finalization allocation because it performs both steps.
Slow DLC receives a larger time/scratch allowance. These requests are starting
settings, not guaranteed minima; inspect resource reports for representative data
and adjust allocations as needed. The desktop exposes stage CPU/RAM settings and
separate MMGen time limits. Prepared bundles keep their saved requests and source
snapshot throughout a run or resume.

The [desktop launcher](DESKTOP.md) uses the same preparation/submission machinery
and provides scheduler monitoring, pipeline recovery, FeatureImp and PosePred
training. See the workflow-specific recovery limits in that guide.

## Failures and restart

Each processing command gets two attempts in the same allocation, with a
30-second wait after the first failure. A successful retry lets dependent jobs
proceed normally. Both attempts retain timing/resource reports and log messages.
Signals and cancellation are not retried. A persistent error still fails the
job and blocks dependent processing, preserving complete-input requirements.
Fast MMGen processing uses its native resume checks so a retry does not duplicate
already committed recordings. A failure before the first commit can also resume:
the native store identity is checked, interrupted rows remain pending, and the
standard shard lock is released when an ordinary exception ends an attempt.
Shard processing and shard-matrix finalization use OS file locks, which also
release when a process is killed. Lock files remain as coordination records;
merge probes actual ownership rather than file existence. Older directory locks
are converted automatically after scheduler accounting or a local process check
confirms their owner has finished. An older lock with no verifiable owner remains
protected and reports the precise reason.

After a persistent error is resolved, use the same prepared submission:

```bash
bash /data/example_lab/autobehaver/_job_bundles/example_run/submit.sh --resume
```

The submitter reads scheduler states, retains successful and active jobs, submits
failed/cancelled jobs and missing submissions, and rebuilds dependencies using
their new job IDs. It appends accepted IDs to `jobs.tsv`; the latest entry for
each batch is authoritative. Benchmark collection is refreshed for the restart.
Repeating `--resume` does not duplicate active jobs. Unavailable scheduler state
or an active dependent of a failed parent stops restart before submission; wait
for accounting or scheduler cancellation to settle and try again.

A process killed between scheduler acceptance and saving the returned job ID
can leave an accepted job absent from `jobs.tsv`. If submission is interrupted
at that point, check the scheduler before restarting. The submission lock
prevents overlapping commands; it cannot make those two operations indivisible.

Every collector invocation writes uniquely named `scheduler_report_*.psv` and
`resource_summaries_*.csv` files and prints their exact paths in its log. Earlier
reports remain available after a restart. Crop, DLC and PosePred also retain
individual batch validation reports beside their conventional latest report.

Filename maps are read by the requested recording's key. Native writers lock
only that name, check conflicts and publish a complete immutable JSON record by
atomic replacement. Identical concurrent requests are harmless; different
original names cannot overwrite the same mapping. Directory discovery accepts a
snapshot while unrelated maps arrive and refreshes its cache on later changes.
The small per-name `.lock` files remain with `_names`; their OS locks release
automatically when a writer exits. Keep them with the other filename records.

Existing mapped artifact names are reused after relocating a project. If a new
root makes an existing path too long, use a shorter root rather than renaming
only part of the linked output set. Always transfer the `_names` directory with
its artifacts.

Optional registry and crop-selection writes use the same process-held locking
mechanism. A lock directory left by an older package cannot establish whether
its writer is still active: stop or verify completion of older writers before
removing that legacy directory. Current lock files must remain in place.
