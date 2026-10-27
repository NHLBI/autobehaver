"""Display the pipeline stages and their command-line entrypoints."""

from __future__ import annotations

import argparse
from textwrap import dedent

from . import __version__


PIPELINE_ROADMAP = dedent(
    """
    Autobehaver pipeline

    Run each stage's --help command for its paths and analysis options.

    Setup
    -----
    Main environment:
       conda env create -f environment/environment.yml
       conda activate autobehaver
       python -m pip install -e .
       python environment/check_environment.py

    Separate DeepLabCut environment:
       conda env create -f environment/environment-dlc-tf.yml
       conda activate autobehaver-dlc
       python -m pip install -e .
       python environment/check_environment.py --dlc-only --require-dlc

    An existing cluster DeepLabCut environment can also run extraction.

    Pipeline stages
    ---------------
    1. MovieChecker: review recordings and save chamber exclusions.
       python -m autobehaver.preinf.moviechecker --help

    2. Autocropper: crop full-frame recordings into chamber movies.
       python -m autobehaver.preinf.autocropper --help

    3. DeepLabCut: extract keypoints from cropped movies.
       python -m autobehaver.preinf.dlc --help

    4. KeyPre: calculate positional and kinematic features.
       python -m autobehaver.preinf.keypre --help

    5. KeyPreNorm: normalize feature files for PosePred.
       python -m autobehaver.preinf.keyprenorm --help

    6. PosePred: predict behavior and orientation on the source-frame timeline.
       python -m autobehaver.preinf.posepred_inference --help

    7. MMGen: combine features and predictions into per-animal matrices.
       python -m autobehaver.postinf.mmgen --help

    8. FeatureImp: analyze group differences with feature importance, PCA and FSEA.
       python -m autobehaver.postinf.featureimp --help

    9. InfMovie: render behavior and orientation overlays.
       python -m autobehaver.postinf.infmovie --help

    Training
    --------
    Train PosePred with normalized features, labels and metadata:
       python -m autobehaver.training.posepred_training --help

    Paths and records
    -----------------
    Use --main_dir for numbered project folders or explicit input/output paths.
    See examples/pipeline_commands.txt for commands to adapt to your project.
    Keep run manifests, validation receipts and _names maps with results.

    Validation
    ----------
       python environment/check_environment.py
       python -m pytest -q
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
