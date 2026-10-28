# FeatureImp from the desktop launcher

Choose the project or an alternate master matrix, the two training groups,
normalization reference, sex and included individuals. Numerical analysis
settings are available under Advanced. The GUI runs one chosen seed per analysis.

Use **Filter group**, **Filter sex**, and **Filter trial** to show matching animals.
The filters combine. **Include matches** or **Exclude matches** changes those
animals only; **Include only matches** also excludes everyone else. To restrict
an analysis to its two training groups, choose **Training groups A + B**, leave
the other filters at All, and click **Include only matches**. The individual
checkboxes remain available for specific exclusions. The separate analysis Sex
setting still applies when building. The displayed inclusion count is before
that analysis setting. A trial is read from the matrix's trial column or the
standard `_trial..._` individual-name tag; unavailable metadata is shown as Unknown.

The scroll wheel navigates the tab when hovering over settings controls; change
values with clicks or the keyboard instead.

With **Plan locally and run training shards** checked:

1. **Build job** copies the selected matrix locally, normalizes it, and freezes
   the feature selection, folds, subset assignments and random seeds. No models
   train during planning. Review the animal/feature/task counts before sending.
2. **Send to HPC** copies and verifies the frozen plan and package through the
   mounted drive. It does not start processing.
3. **Submit job** installs the plan in a new ordinary `9_feature_importance`
   analysis folder and submits training shards, finalization, and the selected
   PCA/FSEA stages with scheduler dependencies. An accounting job follows.

Defaults are 8 training jobs, each 12 CPUs/32 GiB; finalization and PCA each
8 CPUs/32 GiB; FSEA 8 CPUs/64 GiB. These are editable initial allocations, not
measured guarantees for every matrix. Training uses at most CPUs minus two
workers, further limited by available memory and a measured first task.
Finalization reduces independent shards in parallel and combines results in
their original order. The final small ordered reductions remain serial.

RAM admission uses the smallest supplied, Slurm, cgroup and available-system
limit, reserves 25% or at least 1 GiB, and accounts for the parent process.
Training includes conservative per-worker estimates and measured peak memory
with 50% headroom; later observations can reduce concurrency between task
groups. These controls are estimates, not a hard guarantee against native-library
memory spikes. FSEA checks an explicit permutation working-memory estimate
before its native calculation; it fails with an actionable allocation message
if insufficient. It never reduces the configured permutation count (100,000 by
default) to fit RAM.

PCA and FSEA run in separate allocations after training memory is released.
Their separate processes also isolate failures. Output logs report resource
admission and phase timings; each job retains resource-monitor reports, and
the collector records scheduler accounting.

Uncheck sharding to retain the single-job workflow. That job reserves the largest
RAM allocation required by the selected stages. For new configurations this is
64 GiB when FSEA is enabled, rather than reserving it for every training shard.

Each bundle has a durable accepted-job ledger and rejects accidental duplicate
submission. **Resume interrupted submission** adds only jobs never accepted;
it checks the accepted jobs first. It does not retry failed jobs. Automatic
failed-stage recovery is not yet provided for sharded FeatureImp; retain the
original analysis, plan and completed shards for a stage-specific retry.
Never rebuild a new plan over an existing analysis or modify a submitted source
bundle. Single-job FeatureImp and pipeline recovery are available.

Plans use relative portable artifact paths, Linux line endings for launcher
files, and content fingerprints. Shared reduction inputs are hashed once per
operation and verified again before results are accepted; no verification cache
survives the operation. Planning retains the original absolute paths as audit
metadata, but execution resolves the frozen artifacts in the relocated analysis.
