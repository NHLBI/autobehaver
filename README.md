<p align="center">
  <img src="assets/autobehaver_banner_v1.png" alt="Autobehaver" width="900">
</p>

# Autobehaver

Autobehaver is a Python package for processing and analyzing animal behavior videos. The current implementation was developed for high-throughput analysis of *Drosophila* recordings, but the package could be adapted for other projects.

Autobehaver is designed for experiments where the goal is to compare behavioral signatures between groups. The pipeline converts videos and keypoints into per-animal behavioral feature vectors, trains classifier ensembles to distinguish reference groups such as control and mutant animals, and thus creates a behavioral axis upon which animals with intermediate phenotypes can be placed. Autobehaver also uses feature-importance analysis to identify the behavioral features that separate groups, providing some interpretability. Our goal is to use Autobehaver for rescue-style experiments, where the key question is whether a treatment, genotype, or condition shifts an individual mutant's behavioral profile away from the mutant state and towards the control state.

The accompanying preprint is available on bioRxiv: [Autobehaver: An AI-Based Pipeline for Animal Behavior Analysis](https://www.biorxiv.org/content/10.64898/2026.05.12.724596v1).

Autobehaver is under active development. The repository is intended to make the current codebase usable, inspectable, and reproducible.



## Repository layout

```text
autobehaver/     Python package source
configs/         Example YAML configuration files
environment/     Conda environment recipes and environment checks
tests/           Package tests
assets/          README images and other small repository artwork
```



## Installation

Autobehaver is intended to be run from a cloned copy of this repository.



### Main Autobehaver environment

Use this environment for most of the pipeline:

```bash
git clone https://github.com/NHLBI/autobehaver.git
cd autobehaver

conda env create -f environment/environment.yml
conda activate autobehaver

python -m pip install -e .
python environment/check_environment.py
python -m autobehaver
```

`python -m pip install -e .` installs the local source tree in editable mode. After this, the environment can find Autobehaver reliably, and the top-level roadmap command is available:

```bash
python -m autobehaver
```



### DeepLabCut environment

The main environment does not install DeepLabCut or TensorFlow. Use a separate DLC-capable environment only when running the DLC stage itself:

```bash
conda env create -f environment/environment-dlc-tf.yml
conda activate autobehaver-dlc
python -m pip install -e .
python environment/check_environment.py --dlc-only --require-dlc
python -m autobehaver.preinf.dlc --help
```

On an HPC system that already provides a DeepLabCut module, you can use that module instead. For example:

```bash
module load DeepLabCut/2.3.9
cd /path/to/autobehaver
python-DLC -m autobehaver.preinf.dlc --help
```

A typical workflow is to run DLC once in the DLC-capable environment, then switch back to the main environment for keypoint preprocessing, PosePred inference, master-matrix generation, feature-importance analysis, and overlay movie rendering.

For more detail, see `environment/environment_README.md`.



## Configuration files

The `configs/` folder contains example YAML files:

```text
configs/smoke_config.yaml        Small example configuration showing stage wiring
configs/pipeline_template.yaml   Template for adapting the pipeline to a new project
```

Each command-line stage also accepts explicit arguments. Run the relevant `--help` command before adapting a stage to a new dataset.



## Pipeline overview

Autobehaver is organized into pre-inference, inference/training, and post-inference stages. For a command-line roadmap of the full workflow, run:

```bash
python -m autobehaver
```

Each stage has its own `--help` page. The stage summaries below are meant to orient new users; run the stage-specific help command before adapting a stage to a new dataset.

### 1. Movie checking

Module:

```text
autobehaver.preinf.moviechecker
```

Moviechecker records which source movies are available and can create exclusion metadata used by downstream stages.

```bash
python -m autobehaver.preinf.moviechecker --help
```

### 2. Autocropper

Module:

```text
autobehaver.preinf.autocropper
```

Autocropper detects chambers in full-frame movies and writes cropped per-chamber movies for individual-animal analysis.

```bash
python -m autobehaver.preinf.autocropper --help
```

### 3. DeepLabCut keypoint extraction

Module:

```text
autobehaver.preinf.dlc
```

The DLC wrapper runs DeepLabCut on cropped movies and writes keypoint CSV files. This stage should be run from a DLC-capable environment or from an HPC DLC module. Downstream stages do not require DLC once the keypoint CSVs exist.

```bash
python -m autobehaver.preinf.dlc --help
```

### 4. Keypoint preprocessing

Module:

```text
autobehaver.preinf.keypre
```

Keypre converts DLC keypoint CSVs into positional and kinematic feature files. These files provide the per-frame keypoint-derived measurements used by later inference and feature-matrix stages.

```bash
python -m autobehaver.preinf.keypre --help
```

### 5. Feature normalization

Module:

```text
autobehaver.preinf.keyprenorm
```

Keyprenorm normalizes keypoint-derived feature files so that they can be passed into PosePred inference using the same feature scale used during model training.

```bash
python -m autobehaver.preinf.keyprenorm --help
```

### 6. PosePred inference

Module:

```text
autobehaver.preinf.posepred_inference
```

PosePred inference applies a trained behavior/orientation model to normalized feature files and writes per-frame behavior and orientation predictions.

```bash
python -m autobehaver.preinf.posepred_inference --help
```

### 7. Master-matrix generation

Module:

```text
autobehaver.postinf.mmgen
```

MMGen combines pose/orientation inference outputs with preprocessed keypoint features to produce individual-level behavioral feature matrices. These matrices summarize time budgets, bout structure, transitions, kinematics, posture, spatial features, sleep-related features, and related measurements.

```bash
python -m autobehaver.postinf.mmgen --help
```

### 8. Feature importance, PCA, and FSEA

Module:

```text
autobehaver.postinf.featureimp
```

FeatureImp reads the master feature matrix, trains classifier ensembles to distinguish reference groups, computes feature-importance outputs, and can run reference-axis PCA and feature set enrichment analysis.

```bash
python -m autobehaver.postinf.featureimp --help
```

### 9. Inference overlay movie

Module:

```text
autobehaver.postinf.infmovie
```

InfMovie renders overlay movies showing cropped videos together with predicted behavior and orientation labels.

```bash
python -m autobehaver.postinf.infmovie --help
```

## PosePred training

PosePred training has a separate entry point:

```bash
python -m autobehaver.training.posepred_training --help
```

The public example workflow is focused on inference and post-analysis. Training a production PosePred model requires labeled training data and additional project-specific setup.



## Development checks

From the main environment:

```bash
python environment/check_environment.py
python -m compileall -q autobehaver tests
pytest -q
```

From a DLC-capable environment or HPC DLC module:

```bash
python environment/check_environment.py --dlc-only --require-dlc
python -m autobehaver.preinf.dlc --help
```



## Citation

If you use Autobehaver, please cite the preprint:

> O'Neill, R.S., Aviles, S., and Rusan, N.M. **Autobehaver: An AI-Based Pipeline for Animal Behavior Analysis.** bioRxiv, 2026. doi: [10.64898/2026.05.12.724596](https://doi.org/10.64898/2026.05.12.724596)



## License

Autobehaver is distributed as a U.S. Government Work. It was prepared by NIH/NHLBI employees as part of their official duties and is not subject to U.S. copyright protection under 17 U.S.C. 105. See `LICENSE.txt` for details.

Third-party software and resources remain under their own licenses; see `THIRD_PARTY_NOTICES.md`.
