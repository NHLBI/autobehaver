"""Entrypoint for ``python -m autobehaver.training``.

This routes to the pose prediction training CLI.
"""

from .posepred_training.cli import main

if __name__ == "__main__":
    main()
