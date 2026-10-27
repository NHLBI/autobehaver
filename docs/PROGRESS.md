# Command-line progress

Autobehaver automatically uses an updating line when its progress stream is an
interactive terminal. Redirected output, including ordinary Slurm job logs, uses
plain text: start/end messages and throttled milestones. Unchanged file counts
are not repeatedly printed. Existing warnings, errors and result summaries remain.

No additional dependencies are required. To override automatic display:

```bash
export AUTOBEHAVER_PROGRESS=log   # plain text, even in an interactive terminal
export AUTOBEHAVER_PROGRESS=off   # suppress progress only, not warnings/errors
export AUTOBEHAVER_PROGRESS=auto  # default
```

`bar` forces a live line; avoid it for saved logs. A pipeline using `tee` normally
selects plain-text output when the progress stream is also redirected.

## What the counts mean

- Autocropper: completed movie tasks plus the latest worker's operation. Encoding
  reports FFmpeg's frame count for the current chamber group; validation is separate.
  With concurrent movies, the detail names the latest reporting operation, not
  an aggregate frame percentage across movies.
- KeyPre and normalization: completed file results. KeyPre reports each bounded
  group separately. A file may finish as succeeded, reused or failed; consult the
  stage's final validation summary for those outcomes.
- PosePred: processed inference windows, counted across variable-sized batches,
  followed by writing the recording outputs.
- Training: training and validation batches. Epoch metrics/checkpoint messages
  remain unchanged. Pretraining reports loaded chunks separately.
- Annotated movies: completed recording groups.
- MMGen: recording tasks, shard hash checks, matrix-part checks, shard merging,
  indexing chunks and report tasks are separately named operations.
- FeatureImp: completed pool tasks and named phases, including enrichment.
- DeepLabCut: named backend operations and output verification. The third-party
  backend does not expose a stable frame-progress callback to Autobehaver; no
  percentage is invented. Its native output is preserved. Redirected external
  backends receive `TQDM_DISABLE=1` (honored by compatible tqdm versions).

A percentage is local to the named operation. A total is omitted when unavailable;
ETA appears only after measurable work has completed. Finishing a progress counter
does not replace scientific validation or indicate that the entire stage succeeded.

## Closing summaries

Autocropper, DLC, KeyPre, normalization, PosePred, MMGen and FeatureImp print a
consistent closing summary after the stage returns, including its final checks
and cleanup. It names the operation, outcome, elapsed time and output location.
Known counts are included; counts unavailable to the stage are omitted.
`validated` means an output passed the existing checks, not necessarily that it
was newly created. `reused` is shown separately where the stage tracks it.
MMGen and FeatureImp name individual operations such as shard processing or
finalization; a shard summary does not certify the whole pipeline.

Failed validation or an exception produces a failed summary, while preserving
the original exception and exit code. Empty selections keep their existing exit
codes even when described as `nothing to process`. Completion summaries remain
visible when progress is disabled. They do not add new output checks or alter
processing and recovery rules.

## Implementation details

`autobehaver.progress` owns presentation and throttling. Progress is not written
through the scientific loggers, so live redraws do not accumulate in their file
handlers. Slurm's captured output receives the sparse plain-text form. Worker
processes do not draw bars. Autocropper supplies a bounded progress queue to its
workers; full queues drop display updates, never processing work. Other pools
report completed results from their existing parent-side iterators.

Reporting does not alter worker admission, ordering, batches, model settings or
calculations. A closed output stream cannot fail a calculation. FFmpeg progress
uses a separate stdout pipe and retains stderr diagnostics, timeout and nonzero
exit handling. Prepared jobs retain their frozen source package and reporting behavior.
