"""Project-registry update for the DLC stage.

The DLC stage consumes *cropped chamber movies* and produces DLC outputs.
In the Autobehaver pipeline, we record the primary output CSV path for each
animal in the **project registry** (a single CSV at project root).

Why only the CSV?
-----------------
Downstream stages (keypre) primarily consume the DLC CSV files. DeepLabCut also
creates an .h5 and *_meta.pickle per video; those filenames often retain DLC's
long scorer/model tag. We intentionally avoid renaming those by default.

This helper updates the registry **once per DLC run** in the parent process to
avoid multiprocessing contention.
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

from autobehaver.registry import read_registry, relpath_str, upsert_registry_rows


def _best_csv_for_video(*, video_path: Path, search_dir: Path) -> Path | None:
    """Return the best matching DLC CSV for a video.

    We look for files in ``search_dir`` whose filename starts with the video
    stem.

    Preference order:
      1) <stem>.csv (preferred after tag-stripping)
      2) shortest filename (heuristic: tag-stripped is shorter)
      3) newest mtime
    """

    stem = video_path.stem
    if not search_dir.exists():
        return None

    cands = [p for p in search_dir.glob(f"{stem}*.csv") if p.is_file() and p.name.startswith(stem)]
    if not cands:
        return None

    exact = search_dir / f"{stem}.csv"
    if exact.exists():
        return exact

    def sort_key(p: Path) -> tuple[int, float]:
        try:
            mtime = p.stat().st_mtime
        except Exception:
            mtime = 0.0
        return (len(p.name), -mtime)

    cands.sort(key=sort_key)
    return cands[0]


def update_registry_from_dlc_outputs(
    *,
    registry_csv: Path,
    project_root: Path,
    videos: Iterable[Path],
    output_dir: Path | None,
    lock_timeout_sec: int = 120,
    lock_stale_sec: int = 3600,
) -> None:
    """Upsert DLC CSV relpaths for the provided video list into the registry."""

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
        csv_path = _best_csv_for_video(video_path=vp, search_dir=search_dir)
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
