"""Master matrix generation (mmgen).

Converts Autobehaver inference outputs + positional_calcs into a master feature
matrix and supporting summary artifacts. Imports are lazy so ``--help`` does not
load the full feature-generation stack.
"""

from __future__ import annotations

from typing import Any

__all__ = ["MMGenConfig", "run_mmgen"]


def __getattr__(name: str) -> Any:
    if name in {"MMGenConfig", "run_mmgen"}:
        from . import core
        return getattr(core, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(__all__)
