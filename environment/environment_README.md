# Autobehaver environments

The main environment runs MovieChecker, Autocropper and all stages after keypoint
extraction. DeepLabCut uses a separate environment because its TensorFlow/system
dependencies differ from the core scientific stack.

## Main environment

From the repository root:

```bash
conda env create -f environment/environment.yml
conda activate autobehaver
python -m pip install -e ".[monitoring]"
python environment/check_environment.py --require-monitoring
python -m autobehaver
```

Local and container environments share `requirements-core.txt`, Python 3.12,
PyTorch 2.10.0 and OpenCV 4.10.0.84. Local OpenCV supports review windows; the CPU
container uses headless OpenCV. For GPU workstations, select an official CUDA
wheel for the same PyTorch version after creating the environment.

The monitoring extra supplies process CPU/RAM and optional NVIDIA telemetry for
the maintained benchmark-enabled job templates. It does not enable monitoring
for ordinary commands. Existing environments can install this extra in place;
the check reports unavailable backends before a long run.

## DeepLabCut environment

```bash
conda env create -f environment/environment-dlc-tf.yml
conda activate autobehaver-dlc
python -m pip install -e .
python environment/check_environment.py --dlc-only --require-dlc
python -m autobehaver.preinf.dlc --help
```

An existing cluster DeepLabCut module can be used instead:

```bash
module load DeepLabCut/2.3.9
cd /path/to/autobehaver
conda activate autobehaver
python -m autobehaver.preinf.dlc --main_dir /path/to/project --dlc_python python-DLC
```

Run the Autobehaver CLI in the supported main environment. `--dlc_python` uses
the module's existing interpreter only for TensorFlow and DeepLabCut API calls;
it supports older cluster containers without requiring Autobehaver to run there.
The main process retains output validation, processing records, and resource
monitoring of backend children. Both environments must see the project, model,
repository, and job scratch paths. Without this option, DeepLabCut runs in the
CLI's own environment, which must meet Autobehaver's Python requirement.

## Dependency files

```text
environment/requirements-core.txt          shared scientific dependencies
environment/requirements-local-core.txt    local PyTorch/OpenCV variants
environment/requirements-dlc-tf.txt        separate DeepLabCut stack
```

Keep scientific dependency versions pinned in the shared dependency file.
After changing dependencies, rebuild the relevant environments, run
`python -m pip check`, and verify the package tests. The supplied DeepLabCut
environment uses Python 3.10 separately from the Python 3.12 main environment.
