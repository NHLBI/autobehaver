# KeyPre and normalization memory

Choose the job's CPUs and RAM. Both stages default to automatic worker selection
(`--num_workers 0`). A positive worker count is a ceiling, still constrained by
available CPUs and the memory budget. The stages run sequentially and plan their
own concurrency; normalization does not inherit a KeyPre-specific worker limit.

```bash
python -m autobehaver.preinf.keypre --main_dir /path/to/project --memory_gb 32
python -m autobehaver.preinf.keyprenorm --main_dir /path/to/project --memory_gb 32 \
  --norm_params /path/to/norm_params.csv
```

`--memory_gb` is a total GiB budget including safety reserve, not a per-worker
allowance. It cannot raise a detected Slurm or cgroup limit. If omitted, the
planner uses Slurm per-node memory, or per-CPU memory times CPUs assigned to this
task, together with accessible cgroup v1/v2 limits. Local available system RAM
can lower the budget further. A Slurm job with no discoverable allocation and no
explicit budget stops with an explanatory error; it never assumes ownership of
all node RAM. On other schedulers pass the allocation explicitly. CPU selection
respects Slurm CPUs per task, OS affinity where supported, and the worker ceiling.
Allocation fields follow the [Slurm sbatch reference](https://slurm.schedmd.com/sbatch.html);
hierarchical Linux limits follow the [kernel cgroup documentation](https://docs.kernel.org/admin-guide/cgroup-v2.html).

The planner reserves 25% of the effective budget, at least 1 GiB. Within the
remainder it chooses workers and chunks, allowing for pandas arrays, parsing,
CSV formatting, HDF5 buffers and process overhead. KeyPre targets 16,384 retained
rows per chunk; normalization retains its 4,096-row target. A constrained budget
first reduces chunk rows, then worker count if even 1,024-row blocks do not fit.
An inadequate budget fails before processing rather than forcing one oversized
worker. These allowances are conservative estimates, not an OS memory limiter;
the scheduler/cgroup enforces the actual job limit. Other software in the same
allocation, unusual inputs, and different library builds can affect peaks.

KeyPre reads bounded DLC blocks. Each feature block includes three derivative
frames plus half the widest rolling window on each side (seven context frames
for the shipped settings). Only the retained center is written. Recording edges,
column order, NaNs and original source frame checks retain their meanings.
Resetting floating-point
rolling accumulators at chunk boundaries can produce rounding-level differences;
chunked output is checked numerically rather than assumed byte-identical.

Each producing worker reads back its output in bounded blocks. The coordinator
reuses that numeric validation if source/output signatures and published content
receipts still agree. It does not repeat full-table parsing after all workers
finish. Existing-output reuse checks still verify source/content identities;
labels and original-frame alignment remain checked. A changed implementation
invalidates incompatible KeyPre receipts; frozen submitted bundles keep their
own code and receipts.

Preprocessing uses crash-aware workers with bounded queued tasks. A killed child
fails the stage and stops its executor siblings, rather than leaving an accepted
task waiting forever. Completed files remain available for native verified reuse.
This does not automatically retry failed jobs or cancel other HPC jobs.

The stage log describes actual workers, chunk rows, budget source and reserve.
`execution_plan` in each run manifest records the resolved values. Existing
opt-in `--resource-monitor` reports measure elapsed time and sampled process-tree
RAM/CPU. Sampled summed RSS can count shared pages twice or miss brief spikes;
it is not exact Slurm/cgroup peak accounting. Use representative HPC observations
to assess throughput and headroom for a new environment.
