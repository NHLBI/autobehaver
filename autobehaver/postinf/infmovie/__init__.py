"""Inference overlay movie (infmovie) utilities.

Imports are intentionally lazy so ``python -m autobehaver.postinf.infmovie --help``
works without importing OpenCV-heavy rendering modules.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "InfMovieConfig",
    "run_infmovie",
    "InfMoviePipelineConfig",
    "run_infmovie_pipeline",
    "render_overlay_movie",
]


def __getattr__(name: str) -> Any:
    if name in {"InfMovieConfig", "run_infmovie"}:
        from . import core
        return getattr(core, name)
    if name == "InfMoviePipelineConfig":
        from . import config
        return getattr(config, name)
    if name == "run_infmovie_pipeline":
        from . import pipeline
        return getattr(pipeline, name)
    if name == "render_overlay_movie":
        from . import movies
        return getattr(movies, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(__all__)
