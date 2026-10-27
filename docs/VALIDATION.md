# Validation and integrity

Stages check declared inputs, output structure and provenance before publishing
or reusing results. An integrity failure stops the stage with a nonzero exit;
scientific quality-control warnings are recorded without altering measurements.

## Running checks

In the scientific environment, install the development dependencies and run:

```bash
python -m pip install -e ".[dev,monitoring]"
python -m pytest -q
```

The tests cover input contracts, frame alignment, scientific calculations,
parallel execution, interrupted processing, output reuse, and job preparation.
Tests using synthetic inputs or mocked schedulers do not establish GPU performance,
cluster authentication, or resource requirements for a particular dataset. Inspect
completed job states and resource reports when choosing allocations for new data.

## Completion

Every requested input is accounted for as newly succeeded, independently checked
and reused, deliberately excluded, or failed. Missing/duplicate outcomes and
failed requested inputs prevent successful stage completion. Keep stage validation
reports and completion receipts with the resulting files.

FeatureImp freezes matrix/fold/task identities in `PLAN_MANIFEST.json`, shard
plans and task records. A completed shard is reusable only when its declared
artifacts still match the recorded hashes.

## Data contracts

- Recording metadata and enabled registries require unambiguous identities,
  valid chamber selections and safe artifact paths.
- Preprocessing, normalized features and predictions retain their declared
  feature schema, frame counts and source-frame identities. Expected unavailable
  prediction edges remain blank; required model inputs must be finite.
- Present prediction probabilities are finite, in range, and consistent with
  the recorded label vocabulary and model ensemble.
- MMGen verifies positional/prediction linkage and analyzed-frame accounting.
  Master matrices use the complete catalog and preserve unavailable measurements
  as NaN. Observed zero and missing data remain distinct.
- FeatureImp requires unique animal identifiers and numeric scientific cells,
  filters its own analysis copy, and checks fold/task assignment consistency.
- Training validates every metadata-declared feature/label file and split.
  Full resumes verify feature/label content; intentional weights-only restarts
  use the explicit optimizer-restart option.

Supervised restarts retain the saved train/validation boundary even when the
optimizer is restarted. The learning-rate finder uses training loss only and can
use its explicitly selected current files. Augmented copies of the same recording
cannot occupy opposite splits, including recordings in nested directories. Full
training resumes reconstruct saved optimizer and scheduler settings before
restoring their state.

## Reruns and compatibility

Existence alone is not successful completion. Reused outputs are checked against
the requested inputs/settings; supported stages regenerate invalid outputs while
preserving prior artifacts as required. Legacy formats are accepted only where
their frame and feature meaning can be established.

Use recorded source-frame identities for existing predictions. For maintenance
smoothing, see [resmoothing](resmoothing.md). Do not use unsafe alignment
fallbacks for new analyses.

Keep the same frozen package version throughout staged/resumed MMGen processing.
MMGen binds shards to a shared inventory of the complete recording
set and verifies both each partition and their combined coverage. Older shards
without that inventory must be finished with their original frozen package or
regenerated in fresh outputs; the merger does not guess the missing coverage.
Manifests and validation history record what ran; they do not establish the
correctness of a biological interpretation.

## Source release inventory

After source edits, regenerate and independently check the release:

```bash
python scripts/release_inventory.py --write
python scripts/release_inventory.py
```

This verifies the version, selected source files and exact file hashes, and parses
every included Python file. SHA256SUMS also covers RELEASE_MANIFEST.json. The source
policy excludes local environments, caches, generated test artifacts, trained
models and normalization parameters. ASSET_SHA256SUMS is a separate, small index
for the supplied model and normalization files. Mounted job publication uses the
same source policy and verifies the copied source bytes.

## Chamber detection

Autocropper fits a translated, rotated and scaled six-chamber layout to circle
candidates. Edge-coverage scores measure detection support; they are not calibrated
probabilities. Missing centers may be inferred from supported detections spanning
both rows and multiple columns. Ambiguous or out-of-frame layouts are rejected.
Inspect saved detection images and explicitly marked inferred centers when
reviewing unfamiliar recording geometry.
