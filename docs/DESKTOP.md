# Desktop job launcher (Windows)

The desktop launcher prepares and submits Autocropper through MMGen, FeatureImp
from a completed master matrix, and PosePred training. It uses the maintained
commands and frozen job bundles.

From a source checkout, install the optional desktop dependency and launch:

```powershell
python -m pip install -e ".[desktop]"
python -m autobehaver.desktop
```

The installed command `autobehaver-desktop` is equivalent. Python with PySide6
runs on the desktop; the existing scientific environment runs on HPC.
The desktop extra includes Paramiko for its persistent SSH connection. The desktop
version currently requires a source checkout containing `scripts/hpc`, plus
mounted HPC storage.

An agent can prepare a named, validated draft and open it for review with
`python -m autobehaver.desktop --open PATH_TO_DRAFT.json`. This does not build,
send or submit jobs. See [agent-assisted setup](AGENT_SETUP.md) for the shared
settings reference and commands. Each Advanced section is collapsible, shows a
count of changed settings, and has its own reset button. Changed sections expand
when loading a setup; the built-run summary lists the overrides.

## First setup

In **Connection**, enter your SSH username and the paired desktop
and HPC paths for the data root and job-bundle root. Set the existing
HPC Python and resource paths. Supply all three PosePred checkpoints, or
leave all three blank to use the defaults under the resource directory.
Settings and local preparation files are kept in `~/.autobehaver/desktop` by
default; `--state-dir PATH` selects another local location.

**Connect to HPC** authenticates inside the app using password/MFA dialogs.
The app checks the server key against existing SSH known hosts or its own saved
host keys before asking for credentials. A first connection to an unknown host
asks the user to verify its fingerprint; a changed key blocks authentication.
Passwords and MFA responses are not saved. The authenticated connection is kept
alive and reused by Submit. **Disconnect** closes it. Closing the app also closes
the connection; a later session requires connecting again.

## Prepare, send, submit

1. Choose the shared **Project folder** above the tabs, then select **Pipeline**
   or **FeatureImp** and fill in its controls.
2. **Build job locally** inventories the inputs, freezes the settings and source,
   and shows a review summary. It does not publish or submit anything.
3. **Send to HPC** publishes the frozen bundle through the mounted drive,
   checks free space, and verifies each copied file. Existing bundles are not
   overwritten. Models and raw movies are not copied.
4. Connect if needed, then **Submit job** checks publication and runs the prepared
   Bash launcher on the existing connection. Progress appears in **Review / output**.

Accepted job records disable Submit and change its label to **Submitted** or
**Partially submitted**. These labels describe submission, not processing success.
Keep the app open until submission finishes; jobs may start while later jobs are
being submitted. If the connection fails, inspect accepted job IDs before using
the workflow-specific recovery actions below. Submission output is also saved in
`submission_output.log` inside the local run folder.

Build, Send and Submit remain above the scrollable tabs. The window fits the
available screen and can be resized for a small remote desktop.

## New, saved and built runs

- **New run** saves the current setup, then starts a fresh setup with connection
  and model locations retained. Set the project and job/analysis name for the new run.
- **Save run** stores an editable named draft under `saved_runs` in the local state
  directory. Saving the same draft updates it; changing its project, workflow or
  name creates a separate draft. It does not overwrite submitted job bundles.
- **Open saved run** accepts a saved draft JSON file or a built run's
  `desktop_run.json`. It restores the saved settings, including allocations and
  FeatureImp selections. A built run opens Monitor; its settings are also restored
  in the relevant tabs. Built runs live under the state directory's `runs` folder.
- **Build** creates a new frozen bundle. Editing controls afterward does not edit
  that bundle. The Review panel describes the saved bundle that Send/Submit uses;
  changing project/workflow/name disables those actions until the matching setup
  is restored or a new bundle is built.

Local preferences restore convenient defaults but do not replace a named saved
run. Accepted submission IDs prevent accidental duplicate submission, including
after reopening a run.

## Pipeline

The **Job label** names only the separate scripts/logs bundle; it does not rename
the dataset or its numbered output folders. Choose full movies, existing crops,
or **Existing DLC -> MMGen (start at KeyPre)**. For the DLC option select the
project root containing `3_dlc`, with the original coordinate CSVs and any `_names`
or `_crop_selection.json` metadata retained. Old prediction CSVs cannot substitute
for coordinates. This start does not require movies or a new MovieChecker review;
saved chamber exclusions are still honored. The full selected
dataset is included; there is no implicit small test selection. Fast/Slow and
MovieChecker choices follow [the native planner](HPC.md). Recording-only
mode requires Fast and ignores existing MovieChecker exclusions. Review the
movie/chamber and job counts before sending. Job settings control batch sizes
and CPU/RAM allocations in this new bundle; reducing CPUs caps explicit worker
counts in its templates. KeyPre + normalization defaults to **10 CPUs / 16 GiB**;
PosePred defaults to **12 CPUs / 24 GiB**. MMGen has separate shard and
finalization allocations; see [MMGen finalization](mmgen_finalization.md).
The **KeyPre / normalization worker ceiling** defaults to **8**. Actual workers
and chunk sizes are selected from each stage's CPU/RAM budget. Choose
**Automatic** to remove this additional worker ceiling. Each stage reserves 25% RAM (at least 1 GiB);
long recordings are processed in bounded feature chunks with temporal context.
Saved runs retain their recorded settings rather than adopting new-run defaults.
See [preprocessing memory](PREPROCESSING_MEMORY.md) for allocation detection,
checks, and measurement limits. Scientific settings otherwise use native defaults.
Outputs use the normal numbered dataset folders. Starts beyond KeyPre and output
archiving are not currently exposed in the desktop launcher.

## FeatureImp

FeatureImp follows the shared project automatically. It finds
`8_master_matrix/MASTER_FEATURE_MATRIX.csv`, displays the `9_feature_importance`
results location, and loads groups and individuals when the tab opens. If the
matrix is absent, it shows a waiting message and checks every 15 seconds while
the tab is open and the app is idle. **Refresh matrix and groups** checks now.
No analysis starts automatically. **Use a different project or matrix** exposes
manual input overrides when needed. Switching projects clears stale selections;
results from an older background read cannot populate the newly selected project.

Select two training groups and a normalization reference, then include/exclude
animals by group, sex or trial, or through the individual table. Included groups
outside the two training groups are scored as intermediate/test groups. Nine
predefined feature families are used. PCA and FSEA are optional and selected by
default in a new setup.

Sharded execution is selected by default. Build freezes the selected matrix and
plans the training tasks locally. Submit schedules training shards, finalization
and the selected PCA/FSEA analyses. Training, finalization/PCA and FSEA have
separate CPU/RAM controls. The source matrix is unchanged. Unchecking sharding
uses the single-job workflow. See [FeatureImp desktop workflow](featureimp-desktop.md)
for allocations, selection controls and recovery limits. This workflow does not
rerun MMGen.

## Monitoring

Open a saved run and select Monitor. Connect to HPC and click Refresh status.
Each stage shows job counts and proportional colored segments: waiting (gray),
running (orange), finished (green), failed or blocked (red), and unknown.
MMGen includes the final merge job. Click a stage count to see job IDs, scheduler
states, exit codes, dependency reasons and output/error logs. Logs may not exist
until a job starts. Finished means scheduler success, not scientific approval.

Optional refresh runs every 60 seconds while connected and idle. Each refresh
uses one sacct accounting query and one squeue query; it never submits, cancels
or retries jobs. Local saved snapshots show timestamps and are marked as saved
when reopened. Missing scheduler records remain unknown; failed refreshes mark
the previous display stale. A disconnected refresh shows submission records only.

The Connection tab keeps the configured hostname. HPC labels do not imply
portability of the supplied site-specific Slurm scheduling templates. Distributable
hostname, username and personal paths are blank; entries are remembered in local settings.

## Recover failed pipeline jobs

Open the latest run in a project's attempt history, connect, and use Monitor >
Build recovery. This reads live status and creates a new frozen bundle containing
replacement jobs and a new accounting job. Review, Send, then Submit as usual.
Successful and active processing jobs remain in place; their IDs are retained as
dependencies. Original batch membership is preserved even if batch defaults have
changed. CPU/RAM settings in the form apply to replacement jobs.

Only terminal failures/cancellations are replaced. If a dependent job is still
pending cancellation, wait and refresh. Missing states are not guessed. Submit
checks all referenced states again; a newly failed retained job requires a fresh
plan. A lock and accepted-replacement record prevent another bundle from retrying
the same failed IDs. These runtime records live beside the original bundle; its
source/scripts are never replaced. The monitor includes retained and replacement
jobs, and keeps each log's original location. Open the newest recovery run for
further failures.

For single-job FeatureImp, Build recovery uses the original frozen matrix and original group,
sex, seed and analysis settings; current FeatureImp CPUs/RAM apply to the retry.
A distinct results folder preserves the failed attempt. The live state and prior
replacement claims are checked again at submission. Active/completed runs cannot
be retried by this action.

For interrupted submission, use Review / output > Resume interrupted submission.
This is available for partially submitted pipeline and FeatureImp bundles built
with continuation support. Pipeline continuation rechecks scheduler states,
retains active/completed jobs, and submits remaining work, including terminal
failures when dependencies allow. Interrupted recovery submissions retain their
ownership claims.

For sharded FeatureImp, continuation submits only jobs that were never accepted;
it does not retry failed processing. Use a stage-specific retry with the original
plan for failed shards or analysis stages. Build recovery is not available for
this workflow.

Continuation does not cancel downstream jobs. If dependencies are still pending
cancellation or a retained ancestor failed, inspect/recover that state first.
Unknown states or missing acceptance records are not guessed. Older frozen
bundles are not upgraded by opening them in the launcher.

Recovery cannot repair corrupt inputs or arbitrary processing bugs. Existing
crop pixels are reused only when their producer is pinned by the recovery plan and inputs, settings, shape and output
hashes pass checks. Changing code does not label old pixels as newly processed.


## Shared Fast batching

Fast batching has two minimum-input settings: full movies per crop job and chamber
recordings per job afterward. CPU/GPU job caps apply per processing stage, not
across your entire account. Inputs are balanced; a dataset below the minimum uses
one job. MMGen keeps each animal together, and its final merge/accounting jobs are
additional dependent work. Stage CPUs/RAM and concurrent KeyPre files stay separate.
Defaults are 2 full movies, 20 chamber recordings, 50 CPU jobs and 50 GPU jobs.
Saved runs keep their original schedules and resource settings.

## Process new recordings

After the initial project has completed MMGen, add new recordings to the normal
numbered input folder and select Process new recordings and update the master
matrix in Fast mode. Full movies, existing crops and existing DLC starts are
supported; normal recording review/occupancy requirements still apply.

Build compares selected inputs with the existing MMGEN_SOURCE_LEDGER.csv and its
receipt. Only recordings absent from that committed ledger enter upstream jobs.
The selected lists are frozen: files arriving after Build require another run.
An empty delta or missing/changed ledger is reported before publication. Do not
rename historical recordings to present them as new. Mixed old/new chambers from
an already committed full movie require starting from crops/DLC instead.

One final MMGen append job validates the existing generation and scientific
settings, recomputes affected animals from all their recordings, and updates the
combined MASTER_FEATURE_MATRIX.csv plus ledger/receipts. Unaffected animals are
preserved. Changed historical source contents are rejected by native append checks.
The matrix generation is checked again before submission; this is not a way to
run two simultaneous updates against the same project.

Prior full report tables remain historical. Reports for new/affected animals are
written under 8_master_matrix/_mmgen_append_batches; the review summary states
this distinction. FeatureImp can then be rerun separately using the combined matrix.
Legacy projects without MMGen generation records need a normal run first.

## PosePred training

Use **PosePred training** for one configuration per Build. Choose a training
folder containing normalized feature HDF5 files and human-label files, and its
metadata CSV. Metadata requires `filename` and `labels`; pretraining only requires
`filename`. Paths must be relative to the training folder. A defined split also
requires `set` with both `train` and `val`, unless training without validation.
The shared Project folder above the tabs is for processing/FeatureImp; training
uses its own Training data folder.

Choose supervised training, self-supervised pretraining, or a learning-rate sweep.
The remaining controls use native CLI defaults, choices and help. Hover for option
descriptions. Modes use the applicable options exactly as the CLI does. CPU/RAM,
partition, one-GPU request and time limit are editable. The defaults (16 CPUs,
32 GiB, `gpu:a100:1`, 24 hours) are starting requests, not benchmark guarantees;
match them to your dataset, model and cluster. Loader workers cannot exceed CPUs.

An optional checkpoint enables native resume. Full resume restores saved training
settings and checks input identity; **Weights-only restart** resets optimizer and
scheduler state. It retains the native checkpoint split behavior. A new job never
replaces deployed inference models. Multiple configurations use separate Builds.

Build freezes source, metadata and an optional ordered feature list. Feature/label
files stay in place; Build checks paths and basic metadata, while native training
performs full HDF5, label alignment and content-identity checks on the compute node.
Checkpoint contents are hashed at Build and checked again before submission/run.
Do not edit feature/label inputs while a prepared or running job uses them.

The Review tab shows the command, mode, resources and exact output path. Blank
Output parent writes to the unique bundle's `results`; a custom parent receives
a unique bundle-named subfolder. `Checkpoints` and `Records` are created there,
leaving the input directory untouched. Existing output folders stop execution.
The CLI also supports `--output_directory` and `--metadata_path` directly;
omitting them uses the native default output location and suffix.

Send and Submit use the existing publication verification, persistent connection,
accepted-ID record and duplicate-submit guard. Monitor shows the training job
state and links its output/error logs. Resource samples, final resource summary
and command timing are recorded in the bundle; no extra accounting job is created.
For a failed training job, inspect its logs, then Build a new run with its checkpoint
if appropriate. Automatic Build recovery and partial schedule resume apply to
the other supported workflows, not training. GUI status is scheduler status,
not a live epoch plot or a scientific assessment of the resulting model.
