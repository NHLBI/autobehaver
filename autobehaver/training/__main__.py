"""Route ``python -m autobehaver.training`` to the PosePred training CLI."""

from .posepred_training.cli import main

if __name__ == "__main__":
    main()
