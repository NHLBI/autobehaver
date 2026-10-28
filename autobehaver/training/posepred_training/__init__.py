"""Expose the PosePred CLI and lazily load core training helpers."""

from __future__ import annotations

from .cli import build_parser, main

_CORE_EXPORTS = [
    "prepare_data",
    "model_setup",
    "train_one_epoch",
    "validate_one_epoch",
    "main_worker",
]

__all__ = ["build_parser", "main", *_CORE_EXPORTS]


def __getattr__(name: str):
    if name in _CORE_EXPORTS:
        from . import core

        value = getattr(core, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
