# PosePred inference execution

The native inference command and desktop/HPC jobs automatically choose batch
size and preparation concurrency. Prepared bundles retain their frozen code and
settings; rebuild a bundle to use changed settings or source.

## What runs concurrently

Automatic mode uses up to two DataLoader processes for feature windows and up to
eight CPU threads for deterministic augmentation noise, within the task's CPU
allocation. NumPy's native normal generator runs outside Python's interpreter
lock. Threads avoid copying large noise arrays between processes. One process
owns the GPU and the resident ensemble.
Short recordings that fit one maximum-size host batch use synchronous loading
in automatic mode, avoiding process startup for very little loading work.

Only a bounded queue of upcoming augmentation passes is prepared. GPU runs reuse
one pinned noise transfer buffer sized to the host batch ceiling, waiting for its
previous transfer before overwriting it. Calibration therefore does not accumulate
a pinned noise buffer per tested batch size. CPU preparation overlaps GPU
work; transfers and forwards use the same CUDA stream. This does not claim that
copies overlap GPU kernels or that the GPU can never wait for input.

Augmentation noise uses PCG64 blocks tied to recording, model, frame and seed
identities. The default ensemble uses three final models with ten augmentation
passes per model. Batch-dependent floating-point differences remain possible on GPUs.

## RAM and GPU memory

RAM planning uses the smallest detected Slurm/cgroup/explicit limit and available
host RAM, with a 25% reserve (at least 1 GiB). It accounts for source loading,
full-timeline result buffers, worker overhead, shared/pinned loader buffers and
the noise queue together. Loader prefetch is one batch per worker. A long
recording can reduce concurrency/batch size; a recording whose estimated fixed
cost cannot fit is rejected before loading its feature table. Slurm with no
detectable allocation requires `--memory_gb`; it does not assume the whole node
belongs to this task. The optional monitoring extra supplies psutil for current process-memory
measurements; see [resource monitoring](resource_monitoring.md).

These host allowances are conservative estimates, not OS-enforced limits or a
guarantee against every external allocation or unusual input. Source and result
storage still scale with recording length. Queues are bounded, but full-recording
result buffers are retained in memory.

After loading the ensemble, automatic GPU calibration measures the actual
ensemble/augmentation path on a small initial batch. It tests geometrically
larger batches, with two timed trials per size and a maximum host batch of 8192.
It stops at the joint RAM limit, projected GPU memory headroom, or two sizes
without a greater-than-5% throughput gain. It selects the smallest measured batch
within 5% of the best observed speed. The default VRAM ceiling is 80%, reduced
further when other allocations occupy the selected GPU. It is a ceiling, not a
goal to fill memory. Probes are discarded and excluded from completed-window
counts. The chosen size is reused across recordings; an input-limited first
recording can trigger recalibration on a longer one.

A CUDA allocation failure releases failed temporary buffers and halves the
batch, retrying the same uncommitted windows. No results from a partial ensemble
are written, and successful earlier windows are retained. Non-memory errors
propagate normally. A single window that cannot fit stops with an error. CPU
memory admission estimates reduce exhaustion risk; processing cannot recover
within the same process after an OS kill. CPU-only automatic execution uses at most 256 windows per forward.

## Controls

Native stage: `python -m autobehaver.preinf.posepred_inference`.

| Option | Meaning |
| --- | --- |
| `--batch 0` | Automatic sizing (native-stage default). A positive value is a ceiling; RAM admission and CUDA backoff can reduce it. |
| `--num_workers -1` | Automatic loader process count (native-stage default). Zero means synchronous loading; positive values are a ceiling. |
| `--noise_workers 0` | Automatic noise preparation threads. A positive value is a ceiling. |
| `--memory_gb 32` | Optional total job RAM cap in GiB; cannot override a lower detected allocation. |
| `--vram_fraction 0.8` | Automatic calibration ceiling; must be greater than zero and at most 0.9. |

The desktop passes its PosePred RAM allocation to the command. Its CPU
allocation determines automatic concurrency.
The combined training/inference CLI retains its training defaults; select
`--batch 0 --num_workers -1` explicitly when using its inference mode.

## Measurement

The existing `--resource-monitor` option records allocator peaks and selected
CUDA device identity, loader startup/wait, CPU waiting for prepared noise, CUDA
forward time, CUDA transfer-plus-noise-application time, result processing and
finalization. CUDA timings use events, resolved once per batch rather than after
every forward. Timing categories can overlap and must not be summed as elapsed
runtime. Result processing includes device-to-host copies and CPU formatting.
Calibration time is separate. Allocator peaks after calibration describe
production execution (including retries); reserved memory can include cached
blocks from calibration. NVML readings remain device-wide sampled observations,
distinct from PyTorch allocator measurements. Unavailable CUDA metrics are absent.

Use completed valid windows and whole-job elapsed time for throughput. Completed
output rows also include unavailable edge frames. Startup, file loading, output
writes and validation matter for end-to-end runtime. Calibration rates measure
the forward/augmentation path only, not file decoding or publication.

Batch calibration and memory estimates do not guarantee a particular throughput
or peak allocation on every GPU. Use the per-job measurements to assess the
selected device and dataset.
