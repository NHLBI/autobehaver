"""Expose PosePred training helpers lazily to keep CLI help lightweight."""

from __future__ import annotations

_CORE_EXPORTS = [
    "prepare_data",
    "model_setup",
    "train_one_epoch",
    "validate_one_epoch",
    "main_worker",
]

__all__ = list(_CORE_EXPORTS)


def __getattr__(name: str):
    if name in _CORE_EXPORTS:
        from .. import _training_core

        value = getattr(_training_core, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
