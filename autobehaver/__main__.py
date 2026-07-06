"""Top-level roadmap CLI for Autobehaver."""

from __future__ import annotations

import argparse
from textwrap import dedent

from . import __version__


PIPELINE_ROADMAP = dedent(
    """
    Autobehaver pipeline roadmap
    ============================

    This top-level command is a roadmap. It points to each pipeline stage and
    shows where to find the detailed command-line options. Run the stage-specific
    --help command before adapting a stage to a new dataset.

    Setup
    -----
    Main environment, used for most stages:
       conda env create -f environment/environment.yml
       conda activate autobehaver
       python -m pip install -e .
       python environment/check_environment.py

    Optional DLC/TensorFlow environment, used only for DeepLabCut extraction:
       conda env create -f environment/environment-dlc-tf.yml
       conda activate autobehaver-dlc
       python -m pip install -e .
       python environment/check_environment.py --dlc-only --require-dlc

    On Biowulf HPC systems, an existing DeepLabCut module can be used instead of the
    optional Conda DLC environment.

    Pipeline stages
    ---------------
    1. Movie inventory and exclusions
       Records available source movies and optional exclusion metadata used by
       downstream stages.
       python -m autobehaver.preinf.moviechecker --help

    2. Autocrop full-frame movies
       Detects chambers and writes cropped per-chamber movies.
       python -m autobehaver.preinf.autocropper --help

    3. DeepLabCut keypoint extraction
       Runs DLC on cropped movies and writes keypoint CSV files. Use the DLC
       environment or an HPC DLC module for this stage.
       python -m autobehaver.preinf.dlc --help

    4. Keypoint preprocessing
       Converts DLC keypoint CSVs into positional and kinematic feature files.
       python -m autobehaver.preinf.keypre --help

    5. Feature normalization
       Normalizes keypoint-derived feature files for PosePred.
       python -m autobehaver.preinf.keyprenorm --help

    6. PosePred inference
       Applies a trained PosePred model to normalized features and writes
       per-frame behavior/orientation predictions.
       python -m autobehaver.preinf.posepred_inference --help

    7. Master-matrix generation
       Combines inference outputs and keypoint-derived features into an
       individual-level master feature matrix.
       python -m autobehaver.postinf.mmgen --help

    8. Feature importance and post-analysis
       Runs feature-importance models plus reference PCA/FSEA post-analysis
       outputs from a master feature matrix.
       python -m autobehaver.postinf.featureimp --help

    9. Inference overlay movies
       Renders movies with behavior/orientation overlays for visual inspection.
       python -m autobehaver.postinf.infmovie --help

    Training
    --------
    PosePred model training has a separate entry point:
       python -m autobehaver.training.posepred_training --help

    Example configs
    ---------------
    Example YAML configs live in configs/:
       configs/smoke_config.yaml
       configs/pipeline_template.yaml

    These files show the expected project layout and stage settings. Edit paths
    before running them on a new dataset.

    Validation
    ----------
       python environment/check_environment.py
       python -m compileall -q autobehaver tests
       pytest -q
    """
).strip()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m autobehaver",
        description="Print an Autobehaver pipeline roadmap and stage help commands.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=PIPELINE_ROADMAP,
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"Autobehaver {__version__}",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    del args
    print(PIPELINE_ROADMAP)


if __name__ == "__main__":
    main()
