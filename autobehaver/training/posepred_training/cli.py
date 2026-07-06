"""CLI wrapper for pose prediction training.

Preferred entrypoint:

    python -m autobehaver.training.posepred_training ...
"""

from ...cli import build_parser, main

__all__ = ["build_parser", "main"]
