# Autobehaver environments

Autobehaver uses one main environment plus an optional DeepLabCut environment.
This split is intentional: DeepLabCut/TensorFlow is the most version-sensitive
part of the workflow, while the rest of Autobehaver can run without it once DLC
keypoint CSVs already exist.

## Main environment

Use this for the standard Autobehaver pipeline after DLC keypoints are available:

```bash
conda env create -f environment/environment.yml
conda activate autobehaver
python -m pip install -e .
python environment/check_environment.py
python -m autobehaver
```

This environment is used for autocropper, keypre, keyprenorm, PosePred training
and inference, MMGen, FeatureImp, and InfMovie. It intentionally does not install
DeepLabCut or TensorFlow.

The provided main Conda environment is intended primarily for Linux systems with
CUDA-capable PyTorch. CPU-only, macOS, or different-CUDA users may need to
replace the PyTorch lines in `environment/environment.yml` with the install
recipe recommended by PyTorch for their platform.

## Optional DLC/TensorFlow environment

Use this only when you need to run the DLC extraction stage yourself:

```bash
conda env create -f environment/environment-dlc-tf.yml
conda activate autobehaver-dlc
python -m pip install -e .
python environment/check_environment.py --dlc-only --require-dlc
python -m autobehaver.preinf.dlc --help
```

On HPC systems, an existing DeepLabCut module can be used instead of creating the
Conda environment. For example:

```bash
module load DeepLabCut/2.3.9
cd /path/to/autobehaver
python-DLC -m autobehaver.preinf.dlc --help
```

If the package is not installed into the module's Python environment, run DLC
commands from the repository root or set `PYTHONPATH` to the repository root.

## Why separate environments?

DLC 2.x workflows depend on TensorFlow and GPU/system libraries that can conflict
with or complicate the main scientific environment. Separating DLC keeps the
main environment easier to create and makes it clear that downstream Autobehaver
stages can be tested from existing DLC CSV outputs.

## Pip requirement files

The text requirement files are lightweight references for users who cannot use
the Conda environment files directly:

```text
environment/requirements-core.txt
environment/requirements-dlc-tf.txt
```

The Conda YAML files are the recommended installation path.
