# Prepare an experiment with an agent

An agent can help choose settings, save them in a named draft, and open that draft
in the launcher. The user can then review and edit every supported setting before
using Build, Send and Submit. No agent service or GUI automation is required.

## What to establish

| Workflow | Decisions to discuss | Facts to inspect |
|---|---|---|
| Pipeline | Project, starting stage, saved recording review, Fast/Slow batching, new-only processing, models | Numbered input folders, recording review/occupancy files, model locations, existing outputs |
| FeatureImp | Matrix, training groups A/B, normalization reference, sex and included animals, seed, PCA/FSEA, sharded execution | Available groups/identities in the matrix, paths, resources available for each phase |
| PosePred training | Metadata/split, supervised/pretrain/LR sweep, heads/features, architecture, optimization, seed, optional checkpoint | Feature/label files and metadata, checkpoint availability, GPU environment |

Pipeline order is full movies → chamber crops → DLC coordinates → KeyPre features
and normalization → PosePred behavior/orientation predictions → MMGen master
matrix. The launcher starts at full movies, existing crops, or existing DLC.
FeatureImp consumes the master matrix and is a separate run. PosePred training
consumes labeled features and metadata; it is not part of routine inference.
See [pipeline usage](../README.md) and [desktop controls](DESKTOP.md).

Keep the cohort and scientific choices explicit. A matrix may contain extra
groups that should remain excluded. Groups beyond A/B are scored if included;
the normalization reference is a separate choice. Do not automatically make the
reference group A or silently include every animal. When the user wants all
animals, omitting `selected_identities` means all; otherwise supply exact
`[individual, group, sex]` rows. Sex filtering also applies.

## Supported commands

Run these from the source checkout with the launcher's Python environment:

```text
python -m autobehaver.desktop_agent settings
python -m autobehaver.desktop_agent defaults
python -m autobehaver.desktop_agent inspect "PATH_TO_PROJECT"
python -m autobehaver.desktop_agent validate --workflow pipeline --config setup.json --check-inputs
python -m autobehaver.desktop_agent draft --workflow pipeline --config setup.json --output drafts/example.json --check-inputs
python -m autobehaver.desktop --open drafts/example.json
```

`settings` prints the authoritative keys, types, defaults, choices, bounds and
help as JSON. `defaults` prints an editable starting configuration. A setup file
may contain only overrides; unspecified fields receive current defaults.
Use JSON booleans and numbers, not strings representing them. Training numeric
text boxes deliberately use strings, exactly as reported by `settings`.

`inspect` reads directory names and matrix identity columns without running
analysis. It returns exact group names and identities for explicit selection.
`validate` rejects unknown keys, invalid values and inconsistent settings.
`--check-inputs` also checks local starting folders, matrix membership and groups
after exclusions/sex filtering, or training metadata availability. It is not a
full scientific-input validation or a remote environment/model-load check.
Native preparation and processing retain their own validation.

`draft` creates a new file exclusively; it never overwrites an existing draft or
job record. It does not build, send, submit, connect to HPC, or run inference.
`--open` skips restoring the previous run and opens the exact draft. By default
that session uses a sibling `example.gui-state` directory, keeping the usual
launcher preferences separate. An explicit `--state-dir PATH` overrides this.
Avoid sharing one state directory between simultaneously running launchers.
An ordinary launch without `--open` retains its usual saved-run behavior.

The source checkout and desktop extra are required for the GUI. Settings
inspection and draft generation do not require Qt or scientific GPU libraries.
When launching on Windows from an agent, prefer a hidden console/background
process, with the GUI itself visible. Do not close a busy existing launcher.

## A minimal pipeline setup

```json
{
  "dataset": "D:/experiments/example",
  "pipeline_name": "example_from_dlc",
  "start_at": "features"
}
```

This is a draft to review, not yet a complete connection configuration. Before
Build/Send/Submit, fill the Connection tab: `data_local` and `data_remote` identify
the same data root; `jobs_local` and `jobs_remote` identify the same bundle root;
`python` and `resources` are absolute HPC paths. `host`/`username` identify the
connection. Passwords/MFA are entered at connection time and never saved.
Either leave all three model paths blank to use resource defaults, or supply all
three. Do not guess model identities from shorthand names.

## Advanced controls and defaults

Each advanced section shows how many settings differ from defaults. Changed
sections expand when a setup is loaded. Reset affects only that section.
Build records the effective configuration and lists advanced overrides in its
review summary. Editing a built setup requires a new Build to change its jobs.

| Section | Controls | Effect |
|---|---|---|
| PosePred compute | Batch ceiling, loader and augmentation workers, VRAM budget fraction | Resource use and throughput; automatic batch/worker budgeting remains available |
| PosePred predictions | Averaging passes, TTA, noise/seed, minimum behavior/orientation bouts | Can change predictions |
| FeatureImp analysis | Folds, hits per feature, features per model, trees, learning rate/depth/subsampling, completeness/variance filters, null permutations and weighting, PCA components, FSEA permutations/set sizes/FDR | Can change analysis and runtime; applied to both single-job and sharded preparation |
| Training groups | Native architecture, optimization, loading/reproducibility, feature selection, pretraining and SWA options | Native training configuration; defaults/help come from its parser |

Usual allocation defaults remain 10 CPUs/16 GiB with an 8-worker ceiling for
features, 12 CPUs/24 GiB for PosePred, 12 CPUs/32 GiB for MMGen shards and
12 CPUs/80 GiB for finalization. Fast batching caps apply per stage, not across
the whole account. Confirm cluster constraints rather than treating these as
guaranteed available resources. Memory budgeting reduces risk; it cannot
guarantee that arbitrary input/model combinations fit.

Advanced controls intentionally do not expose arbitrary shell arguments,
destructive overwrite switches or every diagnostic/compatibility flag. For an
unsupported native option, say it is unavailable in the GUI and discuss using
the documented CLI. Never hide extra scientific options in a draft that the
GUI cannot display or preserve.

## Review and handoff

1. Summarize the project, workflow, groups/cohort, seed/models and overrides.
2. Validate and save a new named draft; report any unverified paths/environment.
3. Open that exact draft and let the user review the settings.
4. Build locally, then review the frozen jobs and resource summary.
5. Send and submit only within the user's authorization. Monitor the saved run;
   accepted jobs are not proof of successful processing.
