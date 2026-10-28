# Model and normalization resources

Trained checkpoints and recording data are distributed separately from ordinary
Git history, alongside the accompanying article. Use the supplied model files
and normalization parameters at the paths below, or pass their locations
explicitly to the relevant stage.

## PosePred

The default final ensemble requires:

```text
autobehaver/resources/posepred/nh16_e301.pth
autobehaver/resources/posepred/nh32_e301.pth
autobehaver/resources/posepred/nh64_e301.pth
autobehaver/resources/norm_params/norm_params.csv
```

The three models use the `no_pos` schema (820 normalized input features).
Expected SHA-256 values for the final models and normalization file are recorded
in [ASSET_SHA256SUMS](../ASSET_SHA256SUMS). Verify the files
against that index before processing. From the repository root on systems with
`sha256sum`, use:

```bash
sha256sum -c ASSET_SHA256SUMS
```

Pass different checkpoint locations with PosePred's `--inference_model` option
and matching normalization data with KeyPreNorm's `--norm_params` option.
Inference completion manifests retain the checkpoint and source-data hashes.
HPC preparation accepts the same three existing checkpoints with
`--inference-model MODEL16 MODEL32 MODEL64`. Its `--resources` path supplies the
matching normalization data and DLC project. Mounted source publication excludes
model resources and reuses these existing HPC paths.

## DeepLabCut

The extraction stage requires a complete trained DeepLabCut project, including
its model files. The standard restored location is
`autobehaver/resources/dlc/m1/config.yaml`; another project can be selected with
`--dlc_config`. Run this stage in a supported DLC-capable environment, or use
the main environment with `--dlc_python /path/to/dlc-environment/bin/python`
to dispatch extraction to a separate DeepLabCut environment. The backend must see the same model and
scratch paths; output validation and processing records remain in the main process.

## Resource releases

Keep each model with its matching normalization parameters, feature schema,
class dictionaries, licenses and checksums.
Use a fixed release identifier for scientific runs. A source-only installation
can display CLI help and run synthetic tests without the trained models;
model inference requires the real resources.
