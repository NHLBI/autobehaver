"""Record DeepLabCut CSV outputs in the project registry.

Update once in the parent process; KeyPre consumes the recorded CSV path.
"""

from __future__ import annotations

import logging
from autobehaver.portable_paths import original_name
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

from autobehaver.registry import read_registry, relpath_str, upsert_registry_rows


def _best_csv_for_video(*, video_path: Path, search_dir: Path, output_index=None) -> Path | None:
    """Return the unique indexed DeepLabCut CSV for a video, rejecting ambiguity."""

    stem = Path(original_name(video_path)).stem
    if not search_dir.exists():
        return None
    from .core import build_dlc_output_index
    if output_index is None:
        output_index = build_dlc_output_index(search_dir)
    candidates = [Path(p) for p in output_index.get(stem, {}).get('csv', [])]
    if len(candidates)>1:
        raise ValueError(f'Ambiguous DLC CSV artifacts for {video_path}')
    return candidates[0] if candidates else None


def update_registry_from_dlc_outputs(
    *,
    registry_csv: Path,
    project_root: Path,
    videos: Iterable[Path],
    output_dir: Path | None,
    lock_timeout_sec: int = 120,
    lock_stale_sec: int = 3600,
) -> None:
    """Upsert DeepLabCut CSV paths for the supplied videos."""

    registry_csv = Path(registry_csv).expanduser().resolve()
    project_root = Path(project_root).expanduser().resolve()

    if not registry_csv.exists():
        logging.warning(f"Registry update (DLC) skipped: registry CSV not found: {registry_csv}")
        return

    df = read_registry(registry_csv)
    if df.empty:
        logging.info("Registry update (DLC): registry is empty; nothing to update")
        return

    # Map cropped_movie_relpath -> recording_id
    rel_to_id: dict[str, str] = {}
    for _, r in df.iterrows():
        rel = r.get("cropped_movie_relpath")
        rid = r.get("recording_id")
        if isinstance(rel, str) and rel.strip() and isinstance(rid, str) and rid.strip():
            rel_to_id[rel.replace("\\", "/")] = rid

    now = datetime.now().isoformat(timespec="seconds")
    rows: list[dict[str, Any]] = []
    indexes = {}

    for v in videos:
        try:
            vp = Path(v).expanduser().resolve()
        except Exception:
            vp = Path(v)

        # Identify registry row
        rel = relpath_str(vp, project_root)
        rid = rel_to_id.get(rel)

        if not rid:
            # Fallback: match by basename if the relpath differs across mirrors.
            bn = vp.name
            matches = [k for k in rel_to_id.keys() if Path(k).name == bn]
            if len(matches) == 1:
                rid = rel_to_id[matches[0]]

        if not rid:
            continue

        search_dir = output_dir if output_dir is not None else vp.parent
        if search_dir not in indexes:
            from .core import build_dlc_output_index
            indexes[search_dir] = build_dlc_output_index(search_dir)
        csv_path = _best_csv_for_video(video_path=vp, search_dir=search_dir, output_index=indexes[search_dir])
        if csv_path is None:
            continue

        rows.append(
            {
                "recording_id": rid,
                "dlc_csv_relpath": relpath_str(csv_path, project_root),
                "updated_at": now,
            }
        )

    if not rows:
        logging.info("Registry update (DLC): no DLC CSV outputs found to upsert")
        return

    upsert_registry_rows(
        csv_path=registry_csv,
        rows=rows,
        lock_timeout_sec=lock_timeout_sec,
        lock_stale_sec=lock_stale_sec,
    )
    logging.info(f"Registry updated (DLC): {registry_csv} (+{len(rows)} rows upserted)")
