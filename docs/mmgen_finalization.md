# MMGen finalization workers

The desktop has separate shard and finalization CPU/RAM/hour settings.
Defaults: shards 12 CPUs / 32 GiB / 18 hours; finalization 12 CPUs / 80 GiB /
48 hours. Time requests remain subject to cluster policy. Single-job MMGen runs
also finalize, so they use the finalization template.

Fast preparation uses at least 80 recording units per MMGen shard (or the larger
shared chamber minimum), retaining the CPU-job cap and animal-identity grouping.
Explicit per-stage batch-size arguments override the shared batching defaults.

The full-report path builds one sequential recording-to-row lookup, supporting
multiple spans per recording. Workers independently read, transform, export and
summarize recordings. The parent merges in sorted recording order, preserving
numerical reductions and the density-sampling random stream.

Memory admission uses the smallest detected Slurm/cgroup/local-available limit,
CPU affinity and requested worker ceiling. It reserves at least 25% RAM and at
least 2 GiB. Each recording estimate is 1 GiB plus frame count times the larger
of 4096 bytes and eight times persisted row width. The largest recording also
caps pool size, accounting for idle interpreters. Measured worker peak RSS can
raise subsequent allowances but never lower the initial estimates. Estimates
are not OS-enforced guarantees. An oversized recording is rejected before load;
arena calibration is not split into chunks. Later spatial summaries use the
same memory admission with a conservative serialized-size estimate.

Completed CSV/reduction parts and checksums live in .occupancy_parts. Reuse is
bound to source-store path/size/mtime, available run identities, settings and
implementation hashes. Incomplete or damaged parts are rebuilt. Restart repeats
ordered reduction and CSV assembly, but reuses completed recording transforms.
Parts are removed only after the final output receipt, unless keep-temp-files
is requested. Allow space for parts plus the assembled CSV during finalization.

Reuse requires compatible calculation identities. Keep the frozen source bundle
with its submitted run; changing the installed source does not update that bundle.
Finalizing a store with different source requires the explicit compatibility
contract described below.

For a separately audited existing-store restart, the native finalization template
accepts action `finalize` (count1) and uses `--finalize_only`. A prepared bundle
containing `finalize_compatibility.json` selects that action for its finalization
job. The explicit contract pins the previous calculation hash, new source digest
and project directory. It permits no scientific-setting change and cannot be
used for per-recording processing. Original input and commit checks remain
enabled. The output receipt includes `MMGEN_FINALIZATION_EXECUTION.json` with
both original producer and new execution identities. This is not automatic
compatibility with arbitrary prior versions.
