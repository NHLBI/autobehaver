"""MMGen master-matrix entry points.

Combine inference and positional features into an animal-level matrix; analysis imports
load lazily for command-line help.
"""

from __future__ import annotations

from typing import Any

__all__ = ["MMGenConfig", "run_mmgen"]


def __getattr__(name: str) -> Any:
    """Resolve MMGen entry points without eager analysis imports."""
    if name in {"MMGenConfig", "run_mmgen"}:
        from . import core
        return getattr(core, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(__all__)
