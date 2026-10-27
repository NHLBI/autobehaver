"""InfMovie inference-overlay entry points.

Rendering imports load lazily so command-line help does not require OpenCV.
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
    """Resolve public InfMovie entry points without eager rendering imports."""
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
