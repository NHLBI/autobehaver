# Working with Autobehaver

For experiment setup, read [the agent setup guide](docs/AGENT_SETUP.md) and use
the shared settings contract. Begin with `python -m autobehaver.desktop_agent
settings`; do not infer GUI keys or command options from memory.

- Discuss the requested experiment, input stage, comparisons, exclusions and
  intended outputs. Inspect available files before asking questions whose answers
  are already in the project. Do not infer biological groups from their names.
- Prepare a **new named draft** with the maintained `desktop_agent` command.
  Open it with `python -m autobehaver.desktop --open PATH` for user review.
  Opening a draft is not authorization to build, transfer, submit or run analysis.
  Follow the user's existing authorization; do not ask again for authorized work.
- Keep the normal Build, Send, Submit workflow. Use the maintained preparation
  commands for command-line operation. Do not invent shell wrappers for ordinary
  jobs, alter frozen bundles, overwrite prior runs, or silently change cohorts,
  models, scientific settings or seeds to make a run faster.
- Read active advanced overrides back to the user. Compute limits and scientific
  choices are different: TTA, smoothing, normalization, model parameters and FSEA
  thresholds can change results. Defaults are suggestions, not account limits.
- Use read-only inspection by default. Preserve source recordings, models,
  annotations and previous outputs. Store credentials outside draft files.
- Verify the saved draft and exact opened path. Report local validation separately
  from scheduler acceptance and successful processing. Missing files, unknown
  cluster limits, and untested model/environment compatibility remain explicit.

When changing code, keep GUI controls, draft validation and native command
propagation consistent. Add focused tests for settings roundtrip and generated
jobs. Preserve scientific defaults unless a change is specifically authorized.
See [development checks](docs/VALIDATION.md) and [desktop usage](docs/DESKTOP.md).
