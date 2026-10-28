"""Discover InfMovie input pairs, select clips, and render overlays.

Support manual, uncertainty, behavior-diversity, and full-movie plans from folders or an
optional recording registry.
"""

from __future__ import annotations

from autobehaver.progress import progress

import csv
import json
import logging
import os
import random
import re
import time
from dataclasses import asdict, replace
from datetime import datetime
from multiprocessing import Pool
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

from autobehaver.registry import read_registry, relpath_str
from autobehaver.validation import StageValidationReport

from .config import InfMoviePipelineConfig
from .movies import render_overlay_movie
from .frame_map import load_source_frames, source_aligned, read_csv_summary
from .io import open_video_capture, worker_count, resolve_movie_fps

logger = logging.getLogger(__name__)


# Helpers

def _is_active(val: object) -> bool:
    """Interpret the registry 'active' column permissively."""

    if val is None:
        return True
    try:
        import pandas as pd  # type: ignore

        try:
            if pd.isna(val):  # type: ignore[attr-defined]
                return True
        except Exception:
            pass
    except Exception:
        pd = None  # type: ignore

    if isinstance(val, bool):
        return bool(val)
    s = str(val).strip().lower()
    if s in {"0", "false", "no", "n", "off"}:
        return False
    if s in {"1", "true", "yes", "y", "on"}:
        return True
    return True


def _sanitize_filename_component(s: str, *, max_len: int = 140) -> str:
    """Sanitize a string for safe filenames across platforms."""

    s = str(s)
    # Replace spaces with underscores and drop problematic characters.
    s = s.replace(" ", "_")
    s = re.sub(r"[^A-Za-z0-9._\-]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    if len(s) <= max_len:
        return s
    # Truncate and add a stable hash suffix.
    import hashlib

    h = hashlib.sha1(s.encode("utf-8")).hexdigest()[:10]
    keep = max(10, max_len - 11)
    return f"{s[:keep]}_{h}"


def _sec_tag(x: float, *, decimals: int = 1) -> str:
    """Format seconds for filename tags (12.5 -> 12p5)."""

    try:
        v = float(x)
    except Exception:
        v = 0.0
    fmt = f"{{:.{max(0, min(6, int(decimals)))}f}}".format(v)
    s = fmt
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    s = s.replace("-", "m")
    s = s.replace(".", "p")
    if s == "":
        s = "0"
    return s


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _get_video_info(movie_path: Path, *, default_fps: float = 40.0) -> Tuple[float, int]:
    """Return (fps, frame_count) with a safe fallback."""

    fps = resolve_movie_fps(None, default_fps)
    n_frames = 0

    try:
        import cv2

        cap = open_video_capture(str(movie_path))
        if cap.isOpened():
            fps2 = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
            n2 = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            cap.release()
            if np.isfinite(fps2) and fps2 > 0:
                fps = fps2
            if n2 > 0:
                n_frames = n2
        else:
            cap.release()
    except Exception:
        # cv2 missing or video unreadable.
        pass

    return fps, int(n_frames)


def _count_csv_rows(csv_path: Path) -> int:
    """Count data rows in a CSV (excluding header)."""

    try:
        with csv_path.open("r", newline="") as f:
            n = sum(1 for _ in f)
        return max(0, int(n) - 1)
    except Exception:
        return 0


def _pick_label_columns(headers: List[str], *, source: str) -> Tuple[Optional[str], Optional[str]]:
    """Return (pose_label_col, orient_label_col) based on desired source."""

    source = (source or "smoothed").strip().lower()

    pred_pose = "Predicted Pose Label" if "Predicted Pose Label" in headers else None
    pred_orient = "Predicted Orient Label" if "Predicted Orient Label" in headers else None

    sm_pose = "Smoothed Pose Label" if "Smoothed Pose Label" in headers else None
    sm_orient = "Smoothed Orient Label" if "Smoothed Orient Label" in headers else None

    if source == "predicted":
        return pred_pose, pred_orient

    # smoothed preferred; fall back to predicted.
    return sm_pose or pred_pose, sm_orient or pred_orient


def _read_csv_headers(csv_path: Path) -> List[str]:
    try:
        with csv_path.open("r", newline="") as f:
            r = csv.reader(f)
            return next(r)
    except Exception:
        return []


def _load_labels_for_selection(
    csv_path: Path,
    *,
    source: str,
    limit_frames: Optional[int] = None,
    summary=None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Load pose and orientation labels from a checked summary or selected CSV columns.

    Use blank labels for missing columns; unreadable inputs or inputs without either
    label column return empty arrays.
    """

    if summary is not None:
        pose_col, orient_col = _pick_label_columns(list(summary.headers), source=source)
        count = len(summary.frames) if limit_frames is None else min(len(summary.frames), int(limit_frames))
        missing = np.full(count, '', dtype=object)
        return (summary.columns.get(pose_col, missing)[:count], summary.columns.get(orient_col, missing)[:count])
    headers = _read_csv_headers(csv_path)
    pose_col, orient_col = _pick_label_columns(headers, source=source)

    if pose_col is None and orient_col is None:
        return np.array([], dtype=object), np.array([], dtype=object)

    usecols = [c for c in [pose_col, orient_col] if c is not None]

    try:
        import pandas as pd  # type: ignore

        df = pd.read_csv(csv_path, usecols=usecols)
    except Exception as e:
        logger.warning("Could not read labels from %s: %s", csv_path, e)
        return np.array([], dtype=object), np.array([], dtype=object)

    if limit_frames is not None and limit_frames > 0:
        df = df.iloc[: int(limit_frames)].copy()

    pose_arr: np.ndarray
    orient_arr: np.ndarray

    if pose_col is not None and pose_col in df.columns:
        s = df[pose_col].fillna("").astype(str)
        # pandas converts NaN -> "nan" when casting to str; normalize that away
        s = s.replace({"nan": "", "None": ""})
        pose_arr = s.to_numpy()
    else:
        pose_arr = np.array([""] * len(df), dtype=object)

    if orient_col is not None and orient_col in df.columns:
        s = df[orient_col].fillna("").astype(str)
        s = s.replace({"nan": "", "None": ""})
        orient_arr = s.to_numpy()
    else:
        orient_arr = np.array([""] * len(df), dtype=object)

    return pose_arr, orient_arr


def _load_confidence_for_qc(csv_path: Path, *, limit_frames: Optional[int] = None, summary=None) -> Optional[np.ndarray]:
    """Compute per-frame confidence from max pose/orient probabilities.

    Returns (N,) confidence in [0,1], NaN for missing predictions, or None when
    probability columns are absent.
    """

    if summary is not None:
        count = len(summary.frames) if limit_frames is None else min(len(summary.frames), int(limit_frames))
        maxima = []
        for prefix in ('Pose-', 'Orient-'):
            columns = [values[:count] for name, values in summary.columns.items()
                       if name.startswith(prefix) and name.endswith('-Prob')]
            if columns:
                matrix = np.column_stack(columns)
                maxima.append(np.where(np.isnan(matrix), -np.inf, matrix).max(axis=1))
        if not maxima:
            return None
        conf = np.minimum(*maxima) if len(maxima) == 2 else maxima[0]
        return np.clip(np.where(np.isfinite(conf), conf, np.nan), 0.0, 1.0)
    headers = _read_csv_headers(csv_path)
    pose_prob_cols = [h for h in headers if h.startswith("Pose-") and h.endswith("-Prob")]
    orient_prob_cols = [h for h in headers if h.startswith("Orient-") and h.endswith("-Prob")]

    if not pose_prob_cols and not orient_prob_cols:
        return None

    usecols = pose_prob_cols + orient_prob_cols

    try:
        import pandas as pd  # type: ignore

        df = pd.read_csv(csv_path, usecols=usecols)
    except Exception as e:
        logger.warning("Could not read probabilities from %s: %s", csv_path, e)
        return None

    if limit_frames is not None and limit_frames > 0:
        df = df.iloc[: int(limit_frames)].copy()

    # Convert to float; non-numeric -> NaN.
    for c in usecols:
        try:
            df[c] = pd.to_numeric(df[c], errors="coerce")  # type: ignore
        except Exception:
            pass

    n = len(df)
    if n == 0:
        return None

    if pose_prob_cols:
        pose_max = df[pose_prob_cols].max(axis=1, skipna=True)
    else:
        pose_max = None

    if orient_prob_cols:
        orient_max = df[orient_prob_cols].max(axis=1, skipna=True)
    else:
        orient_max = None

    # Combine: use min(pose, orient) when both exist.
    if pose_max is not None and orient_max is not None:
        conf = np.minimum(pose_max.to_numpy(), orient_max.to_numpy())
    elif pose_max is not None:
        conf = pose_max.to_numpy()
    else:
        conf = orient_max.to_numpy()  # type: ignore

    # Keep missing predictions distinct from finite model confidence.
    conf = np.asarray(conf, dtype=float)
    conf = np.where(np.isfinite(conf), conf, np.nan)
    conf = np.clip(conf, 0.0, 1.0)

    return conf


def _run_length_encode(labels: np.ndarray) -> List[Tuple[str, int, int]]:
    """Return list of (label, start_frame, end_frame_inclusive)."""

    if labels.size == 0:
        return []

    out: List[Tuple[str, int, int]] = []
    cur = str(labels[0])
    start = 0
    for i in range(1, int(labels.size)):
        lbl = str(labels[i])
        if lbl != cur:
            out.append((cur, start, i - 1))
            cur = lbl
            start = i
    out.append((cur, start, int(labels.size) - 1))
    return out


def _combine_pose_orient(pose: np.ndarray, orient: np.ndarray) -> np.ndarray:
    """Combine pose+orient into a single label string array."""

    if pose.size == 0 and orient.size == 0:
        return np.array([], dtype=object)

    n = int(max(pose.size, orient.size))
    if pose.size != n:
        pose = np.resize(pose, n)
    if orient.size != n:
        orient = np.resize(orient, n)

    combined = np.empty(n, dtype=object)
    for i in range(n):
        p = str(pose[i]) if i < pose.size else ""
        o = str(orient[i]) if i < orient.size else ""
        if o and o.lower() not in {"nan", "none"}:
            combined[i] = f"{p}-{o}" if p else o
        else:
            combined[i] = p
    return combined


def _fixed_len_clip_from_event(
    *,
    n_total: int,
    clip_len: int,
    event_start: int,
    event_end: int,
    padding: int,
) -> Tuple[int, int]:
    """Convert an event region into a fixed-length clip [start, end_excl)."""

    if n_total <= 0:
        return 0, 0

    event_start = max(0, min(int(event_start), n_total - 1))
    event_end = max(event_start, min(int(event_end), n_total - 1))

    s = max(0, event_start - int(padding))
    e = min(n_total, (event_end + 1) + int(padding))
    if e <= s:
        e = min(n_total, s + 1)

    if clip_len <= 0 or clip_len >= n_total:
        return 0, int(n_total)

    # center around the padded event region
    center = (s + e) / 2.0
    start = int(round(center - clip_len / 2.0))
    start = max(0, min(start, n_total - clip_len))
    end = start + clip_len

    return int(start), int(end)


# Pair discovery

def discover_pairs_from_registry(
    *,
    registry_csv: Path,
    project_root: Path,
    file_filter: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Return list of dicts: recording_id, movie_path, inf_csv_path, metadata."""

    df = read_registry(registry_csv)
    if df.empty:
        return []

    if "active" in df.columns:
        try:
            df = df[df["active"].apply(_is_active)]
        except Exception:
            pass

    required_cols = ["recording_id", "cropped_movie_relpath", "posepred_inference_csv_relpath"]
    for c in required_cols:
        if c not in df.columns:
            return []

    # Filter rows with required relpaths.
    try:
        df2 = df[df["cropped_movie_relpath"].notna() & df["posepred_inference_csv_relpath"].notna()].copy()
    except Exception:
        df2 = df.copy()

    pairs: List[Dict[str, Any]] = []
    missing_artifacts: List[str] = []

    for _, r in df2.iterrows():
        rid = r.get("recording_id")
        mrel = r.get("cropped_movie_relpath")
        irel = r.get("posepred_inference_csv_relpath")
        if not isinstance(rid, str) or not rid.strip():
            continue
        if not isinstance(mrel, str) or not mrel.strip():
            continue
        if not isinstance(irel, str) or not irel.strip():
            continue

        from autobehaver.registry import resolve_registry_path
        movie_path = resolve_registry_path(project_root, r, "cropped_movie_relpath")
        inf_path = resolve_registry_path(project_root, r, "posepred_inference_csv_relpath")
        if file_filter:
            from autobehaver.portable_paths import original_name
            needle = str(file_filter).lower()
            values = [r.get(key) for key in ('recording_id', 'animal_id', 'genotype', 'sex')]
            values.extend([original_name(movie_path), original_name(inf_path)])
            if not any(isinstance(value, str) and needle in value.lower() for value in values):
                continue

        if not movie_path.exists() or not inf_path.exists():
            missing_artifacts.append(
                f"{rid}: movie_exists={movie_path.exists()}, inference_exists={inf_path.exists()}"
            )
            continue

        pairs.append(
            {
                "recording_id": rid,
                "animal_id": r.get("animal_id"),
                "genotype": r.get("genotype"),
                "sex": r.get("sex"),
                "movie_path": str(movie_path),
                "inf_csv_path": str(inf_path),
                "cropped_movie_relpath": mrel.replace("\\", "/"),
                "posepred_inference_csv_relpath": irel.replace("\\", "/"),
            }
        )

    if missing_artifacts:
        raise FileNotFoundError(
            "Registry declares infmovie artifacts that are missing: "
            f"{missing_artifacts[:25]}"
        )
    return pairs


def discover_pairs_from_folders(
    *,
    inference_csv_dir: Path,
    movie_dir: Path,
    file_filter: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Discover matching inference/movie pairs by filename parsing."""

    from .naming import extract_inf_csv_filename_part
    from .io import find_movie_file

    pairs: List[Dict[str, Any]] = []
    unmatched_csvs: List[str] = []

    csv_files = sorted([p for p in inference_csv_dir.glob("*.csv")])
    from autobehaver.portable_paths import original_name
    from autobehaver.preinf.autocropper.selection import filter_included_paths
    csv_files = filter_included_paths(csv_files)

    if file_filter:
        needle = str(file_filter).lower()
        csv_files = [p for p in csv_files if needle in original_name(p).lower()]

    for csv_path in csv_files:
        # Keep named prediction files even if malformed: downstream validation
        # must report their errors rather than silently dropping a recording.
        if not original_name(csv_path).lower().endswith("_inference.csv"):
            with csv_path.open("r", encoding="utf-8-sig", newline="") as stream:
                fields = set(next(csv.reader(stream), []))
            if not ("Frame Number" in fields and fields.intersection({
                "Predicted Pose Label", "Predicted Orient Label",
                "Smoothed Pose Label", "Smoothed Orient Label",
            })):
                logger.info("Skipping non-prediction CSV: %s", csv_path.name)
                continue
        filename_part = extract_inf_csv_filename_part(str(csv_path))
        movie_file = find_movie_file(str(movie_dir), filename_part)
        if not movie_file:
            unmatched_csvs.append(csv_path.name)
            continue
        from autobehaver.portable_paths import original_name
        rid = Path(original_name(movie_file)).stem
        pairs.append(
            {
                "recording_id": rid,
                "animal_id": None,
                "genotype": None,
                "sex": None,
                "movie_path": str(Path(movie_file).resolve()),
                "inf_csv_path": str(csv_path.resolve()),
                "cropped_movie_relpath": None,
                "posepred_inference_csv_relpath": None,
            }
        )

    if unmatched_csvs:
        raise FileNotFoundError(
            "Inference CSV files have no unambiguous matching movie: "
            f"{unmatched_csvs[:25]}"
        )
    return pairs


# Plan generation

def build_plan_manual(cfg: InfMoviePipelineConfig, *, pairs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Select the requested recording and convert its clip times to original-frame
    bounds.
    """
    if not cfg.recording_id:
        raise ValueError("manual mode requires recording_id")

    rid = str(cfg.recording_id).strip()
    matches = [p for p in pairs if str(p.get("recording_id", "")).casefold() == rid.casefold()]
    if not matches:
        matches = [p for p in pairs if rid.casefold() in str(p.get("recording_id", "")).casefold()]
    if not matches:
        raise FileNotFoundError(f"recording_id not found in discovered pairs: {rid}")
    if len(matches) != 1:
        raise ValueError(f"Ambiguous recording_id {rid!r}; matches: "
                         f"{[(p.get('recording_id'), p.get('movie_path')) for p in matches]}")
    match = matches[0]

    movie_path = Path(str(match["movie_path"]))
    csv_path = Path(str(match["inf_csv_path"]))

    fps, n_vid = _get_video_info(movie_path, default_fps=float(cfg.default_movie_fps))
    summary = read_csv_summary(csv_path, source_frame_offset=cfg.source_frame_offset, video_frames=n_vid)
    source_frames = summary.frames
    n_total = int(source_frames[-1]) + 1

    clip_len = int(round(float(cfg.duration_sec) * float(fps)))
    start_frame = int(round(float(cfg.start_sec) * float(fps)))
    start_frame = max(int(source_frames[0]), start_frame)
    if n_total > 0:
        start_frame = min(start_frame, max(0, n_total - 1))

    end_frame = start_frame + max(1, clip_len)
    if n_total > 0:
        end_frame = min(end_frame, n_total)

    start_sec = float(start_frame) / float(fps)
    end_sec = float(end_frame) / float(fps)

    mode = "manual"

    return [
        {
            "mode": mode,
            "recording_id": match.get("recording_id"),
            "animal_id": match.get("animal_id"),
            "genotype": match.get("genotype"),
            "sex": match.get("sex"),
            "movie_path": str(movie_path),
            "inf_csv_path": str(csv_path),
            "fps_in": float(fps),
            "start_frame": int(start_frame),
            "end_frame_exclusive": int(end_frame),
            "start_sec": float(start_sec),
            "end_sec": float(end_sec),
            "duration_sec": float(end_sec - start_sec),
            "reason": "manual",
            "score": None,
        }
    ]


def _summarize_labels(labels: np.ndarray) -> Dict[str, Any]:
    """Count observed labels and return their dominant label and fraction."""
    if labels.size == 0:
        return {"unique_labels": 0, "n_changes": 0, "dominant_label": None}

    labs = np.asarray(labels, dtype=object)
    n = int(labs.size)
    if n <= 1:
        return {"unique_labels": int(len(set(labs.tolist()))), "n_changes": 0, "dominant_label": str(labs[0])}

    changes = int(np.sum(labs[1:] != labs[:-1]))
    # dominant
    try:
        import collections

        c = collections.Counter([str(x) for x in labs.tolist()])
        dom = c.most_common(1)[0][0] if c else None
    except Exception:
        dom = None

    return {"unique_labels": int(len(set([str(x) for x in labs.tolist()]))), "n_changes": changes, "dominant_label": dom}


def build_candidates_interesting(
    *,
    combined_labels: np.ndarray,
    n_total: int,
    clip_len: int,
    min_unique: int,
    min_changes: int,
) -> List[Dict[str, Any]]:
    """Generate candidate windows for interesting mode."""

    if n_total <= 0 or clip_len <= 0 or combined_labels.size == 0:
        return []

    labels = combined_labels[:n_total]

    bouts = _run_length_encode(labels)
    if len(bouts) < 2:
        return []

    cands: List[Dict[str, Any]] = []

    # Evaluate windows anchored at bout starts.
    for bi, (_lbl, b_start, _b_end) in enumerate(bouts):
        start = int(b_start)
        end = start + int(clip_len)
        if end > n_total:
            break

        # Determine bouts that overlap [start, end)
        # Since we're iterating bout starts, bi is a lower bound.
        labels_in_window: List[str] = []
        last_label = None
        n_changes = 0
        j = bi
        while j < len(bouts):
            lbl_j, s_j, e_j = bouts[j]
            if s_j >= end:
                break
            if e_j < start:
                j += 1
                continue
            lbl_str = str(lbl_j)
            if last_label is not None and lbl_str != last_label:
                n_changes += 1
            last_label = lbl_str
            labels_in_window.append(lbl_str)
            j += 1

        uniq = len(set(labels_in_window))
        if uniq < int(min_unique):
            continue
        if n_changes < int(min_changes):
            continue

        # Score: prioritize many unique labels and some switching.
        score = float(uniq) + 0.05 * float(n_changes)

        # Penalize windows dominated by one label.
        # Compute dominant fraction roughly from bouts.
        try:
            import collections

            c = collections.Counter(labels_in_window)
            dom, dom_n = c.most_common(1)[0]
            dom_frac = float(dom_n) / float(max(1, len(labels_in_window)))
            score = score - 0.25 * dom_frac
        except Exception:
            pass

        cands.append(
            {
                "start_frame": int(start),
                "end_frame_exclusive": int(end),
                "score": float(score),
                "reason": f"interesting: uniq={uniq}, changes={n_changes}",
                "unique_labels": int(uniq),
                "n_changes": int(n_changes),
            }
        )

    return cands


def build_candidates_qc(
    *,
    combined_labels: np.ndarray,
    confidence: Optional[np.ndarray],
    n_total: int,
    fps: float,
    clip_len: int,
    padding: int,
    uncertainty_threshold: float,
    min_uncertain_frames: int,
    switch_window_sec: float,
    switch_min_changes: int,
    use_uncertainty: bool,
    use_switching: bool,
) -> List[Dict[str, Any]]:
    """Generate candidate clips for QC mode."""

    if n_total <= 0 or clip_len <= 0:
        return []

    labels = combined_labels[:n_total] if combined_labels.size else np.array([""] * n_total, dtype=object)

    cands: List[Dict[str, Any]] = []

    # Uncertainty candidates
    if use_uncertainty and confidence is not None and confidence.size:
        conf = np.asarray(confidence[:n_total], dtype=float)
        thr = float(uncertainty_threshold)
        missing = ~np.isfinite(conf)
        # Preserve the existing review priority for absent predictions while
        # identifying them explicitly, rather than calling them model uncertainty.
        ranking_conf = np.where(missing, 0.0, conf)
        uncertain = ranking_conf < thr

        # Find contiguous uncertain segments.
        i = 0
        while i < n_total:
            if not uncertain[i]:
                i += 1
                continue
            j = i
            while j < n_total and uncertain[j]:
                j += 1
            seg_len = j - i
            if seg_len >= int(min_uncertain_frames):
                seg_conf = ranking_conf[i:j]
                missing_count = int(missing[i:j].sum())
                category = ('qc_missing_predictions' if missing_count == seg_len else
                            'qc_uncertainty_with_missing_predictions' if missing_count else 'qc_uncertainty')
                # Score: lower confidence -> higher score.
                score = float(1.0 - float(np.nanmin(seg_conf)))
                start, end = _fixed_len_clip_from_event(
                    n_total=n_total,
                    clip_len=clip_len,
                    event_start=i,
                    event_end=j - 1,
                    padding=padding,
                )
                cands.append(
                    {
                        "start_frame": int(start),
                        "end_frame_exclusive": int(end),
                        "score": float(score),
                        "reason": f"{category}: seg={seg_len} frames",
                        "missing_prediction_frames": missing_count,
                    }
                )
            i = j

    # Rapid switching candidates
    if use_switching and labels.size:
        # Transitions array length n_total-1.
        trans = (labels[1:] != labels[:-1]).astype(np.int32)
        # Window size in frames.
        win = int(round(float(switch_window_sec) * float(fps)))
        win = max(2, win)

        # Running count via cumulative sum.
        csum = np.zeros(n_total, dtype=np.int64)
        if trans.size:
            csum[1:] = np.cumsum(trans)

        positions = np.arange(n_total)
        counts = csum - csum[np.maximum(0, positions - win)]

        # Pick top indices.
        order = np.argsort(-counts)
        picked = 0
        for idx in order[: min(len(order), 2000)]:
            if int(counts[idx]) < int(switch_min_changes):
                break
            start, end = _fixed_len_clip_from_event(
                n_total=n_total,
                clip_len=clip_len,
                event_start=int(idx),
                event_end=int(idx),
                padding=padding,
            )
            score = float(counts[idx])
            missing_count = int(np.count_nonzero(labels[max(0, int(idx) - win):int(idx) + 1] == ''))
            category = 'qc_switching_with_missing_predictions' if missing_count else 'qc_switching'
            cands.append(
                {
                    "start_frame": int(start),
                    "end_frame_exclusive": int(end),
                    "score": float(score),
                    "reason": f"{category}: {int(counts[idx])} changes in ~{switch_window_sec:g}s",
                    "missing_prediction_frames": missing_count,
                }
            )
            picked += 1
            if picked >= 50:
                break

    return cands


def _dedupe_and_select(
    candidates: List[Dict[str, Any]],
    *,
    n_select: int,
    min_gap_sec: float,
    max_per_recording: int,
    rng: random.Random,
) -> List[Dict[str, Any]]:
    """Greedy select top candidates with spacing constraints."""

    # Sort by score desc (None -> 0)
    def _score(c: Dict[str, Any]) -> float:
        s = c.get("score")
        try:
            return float(s) if s is not None else 0.0
        except Exception:
            return 0.0

    cands = list(candidates)

    # Shuffle for tie-breaking, then sort.
    rng.shuffle(cands)
    cands.sort(key=_score, reverse=True)

    selected: List[Dict[str, Any]] = []
    per_rec: Dict[str, int] = {}
    starts_by_rec: Dict[str, List[int]] = {}

    for c in cands:
        rid = str(c.get("recording_id") or "")
        if not rid:
            continue

        if per_rec.get(rid, 0) >= int(max_per_recording):
            continue

        s = int(c.get("start_frame", 0))
        min_gap_frames = max(1.0, float(min_gap_sec) * float(c['fps_in']))

        ok = True
        for prev in starts_by_rec.get(rid, []):
            if abs(int(prev) - int(s)) < min_gap_frames:
                ok = False
                break

        if not ok:
            continue

        selected.append(c)
        per_rec[rid] = per_rec.get(rid, 0) + 1
        starts_by_rec.setdefault(rid, []).append(int(s))

        if len(selected) >= int(n_select):
            break

    return selected


def build_plan_interesting(cfg: InfMoviePipelineConfig, *, pairs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Rank behavior-diversity candidates and select spaced clips per recording."""
    rng = random.Random(int(cfg.random_seed) if cfg.random_seed is not None else None)

    all_candidates: List[Dict[str, Any]] = []

    for p in pairs:
        rid = str(p.get("recording_id") or "")
        movie_path = Path(str(p["movie_path"]))
        csv_path = Path(str(p["inf_csv_path"]))

        fps, n_vid = _get_video_info(movie_path, default_fps=float(cfg.default_movie_fps))
        summary = read_csv_summary(csv_path, source_frame_offset=cfg.source_frame_offset, video_frames=n_vid, include_selection=True)
        source_frames = summary.frames
        n_total = int(source_frames[-1]) + 1
        if n_total <= 0:
            continue

        clip_len = int(round(float(cfg.clip_duration_sec) * float(fps)))
        clip_len = max(1, clip_len)
        if clip_len >= n_total:
            continue

        pose, orient = _load_labels_for_selection(csv_path, source=str(cfg.selection_label_source), limit_frames=len(source_frames), summary=summary)
        combined = _combine_pose_orient(pose, orient)

        cands = build_candidates_interesting(
            combined_labels=combined,
            n_total=len(source_frames),
            clip_len=clip_len,
            min_unique=int(cfg.interesting_min_unique_labels),
            min_changes=int(cfg.interesting_min_changes),
        )

        # Keep top per recording.
        cands.sort(key=lambda d: float(d.get("score") or 0.0), reverse=True)
        cands = cands[:50]

        for c in cands:
            c2 = dict(c)
            c2.update(_summarize_labels(combined[int(c["start_frame"]):int(c["end_frame_exclusive"])]))
            c2["start_frame"] = int(c2["start_frame"]) + int(source_frames[0])
            c2["end_frame_exclusive"] = int(c2["end_frame_exclusive"]) + int(source_frames[0])
            c2["source_frame_offset"] = cfg.source_frame_offset
            c2.update(
                {
                    "mode": "interesting",
                    "recording_id": rid,
                    "animal_id": p.get("animal_id"),
                    "genotype": p.get("genotype"),
                    "sex": p.get("sex"),
                    "movie_path": str(movie_path),
                    "inf_csv_path": str(csv_path),
                    "fps_in": float(fps),
                }
            )
            all_candidates.append(c2)

    selected = _dedupe_and_select(
        all_candidates,
        n_select=int(cfg.num_clips),
        min_gap_sec=float(cfg.min_gap_sec),
        max_per_recording=int(cfg.max_clips_per_recording),
        rng=rng,
    )

    # Fill derived timing fields.
    for c in selected:
        fps = float(c.get("fps_in") or float(cfg.default_movie_fps) or 40.0)
        s = int(c.get("start_frame", 0))
        e = int(c.get("end_frame_exclusive", s + 1))
        c["start_sec"] = float(s) / fps
        c["end_sec"] = float(e) / fps
        c["duration_sec"] = float(e - s) / fps


    return selected


def build_plan_qc(cfg: InfMoviePipelineConfig, *, pairs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Rank uncertainty and rapid-switching candidates and select spaced clips per
    recording.
    """
    rng = random.Random(int(cfg.random_seed) if cfg.random_seed is not None else None)

    all_candidates: List[Dict[str, Any]] = []

    for p in pairs:
        rid = str(p.get("recording_id") or "")
        movie_path = Path(str(p["movie_path"]))
        csv_path = Path(str(p["inf_csv_path"]))

        fps, n_vid = _get_video_info(movie_path, default_fps=float(cfg.default_movie_fps))
        summary = read_csv_summary(csv_path, source_frame_offset=cfg.source_frame_offset, video_frames=n_vid, include_selection=True)
        source_frames = summary.frames
        n_total = int(source_frames[-1]) + 1
        if n_total <= 0:
            continue

        clip_len = int(round(float(cfg.clip_duration_sec) * float(fps)))
        clip_len = max(1, clip_len)
        if clip_len >= n_total:
            continue

        padding = int(round(float(cfg.clip_padding_sec) * float(fps)))

        pose, orient = _load_labels_for_selection(csv_path, source=str(cfg.selection_label_source), limit_frames=len(source_frames), summary=summary)
        combined = _combine_pose_orient(pose, orient)

        conf = _load_confidence_for_qc(csv_path, limit_frames=len(source_frames), summary=summary)

        cands = build_candidates_qc(
            combined_labels=combined,
            confidence=conf,
            n_total=len(source_frames),
            fps=float(fps),
            clip_len=clip_len,
            padding=padding,
            uncertainty_threshold=float(cfg.qc_uncertainty_threshold),
            min_uncertain_frames=int(cfg.qc_min_uncertainty_frames),
            switch_window_sec=float(cfg.qc_switch_window_sec),
            switch_min_changes=int(cfg.qc_switch_min_changes),
            use_uncertainty=bool(cfg.qc_use_uncertainty),
            use_switching=bool(cfg.qc_use_switching),
        )

        # Sort and keep top per recording.
        cands.sort(key=lambda d: float(d.get("score") or 0.0), reverse=True)
        cands = cands[:100]

        for c in cands:
            c2 = dict(c)
            c2.update(_summarize_labels(combined[int(c["start_frame"]):int(c["end_frame_exclusive"])]))
            c2["start_frame"] = int(c2["start_frame"]) + int(source_frames[0])
            c2["end_frame_exclusive"] = int(c2["end_frame_exclusive"]) + int(source_frames[0])
            c2["source_frame_offset"] = cfg.source_frame_offset
            c2.update(
                {
                    "mode": "qc",
                    "recording_id": rid,
                    "animal_id": p.get("animal_id"),
                    "genotype": p.get("genotype"),
                    "sex": p.get("sex"),
                    "movie_path": str(movie_path),
                    "inf_csv_path": str(csv_path),
                    "fps_in": float(fps),
                }
            )
            all_candidates.append(c2)

    selected = _dedupe_and_select(
        all_candidates,
        n_select=int(cfg.num_clips),
        min_gap_sec=float(cfg.min_gap_sec),
        max_per_recording=int(cfg.max_clips_per_recording),
        rng=rng,
    )

    for c in selected:
        fps = float(c.get("fps_in") or float(cfg.default_movie_fps) or 40.0)
        s = int(c.get("start_frame", 0))
        e = int(c.get("end_frame_exclusive", s + 1))
        c["start_sec"] = float(s) / fps
        c["end_sec"] = float(e) / fps
        c["duration_sec"] = float(e - s) / fps


    return selected


def build_plan_full(cfg: InfMoviePipelineConfig, *, pairs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Plan the valid original-frame range of each discovered recording."""
    plan: List[Dict[str, Any]] = []
    for p in pairs:
        movie_path = Path(str(p["movie_path"]))
        csv_path = Path(str(p["inf_csv_path"]))
        fps, n_vid = _get_video_info(movie_path, default_fps=float(cfg.default_movie_fps))
        summary = read_csv_summary(csv_path, source_frame_offset=cfg.source_frame_offset, video_frames=n_vid)
        source_frames = summary.frames
        n_total = int(source_frames[-1]) + 1
        if n_total <= 0:
            continue
        plan.append(
            {
                "mode": "full",
                "recording_id": p.get("recording_id"),
                "animal_id": p.get("animal_id"),
                "genotype": p.get("genotype"),
                "sex": p.get("sex"),
                "movie_path": str(movie_path),
                "inf_csv_path": str(csv_path),
                "fps_in": float(fps),
                "start_frame": int(source_frames[0]),
                "end_frame_exclusive": int(n_total),
                "start_sec": float(source_frames[0]) / float(fps),
                "end_sec": float(n_total) / float(fps) if fps > 0 else None,
                "duration_sec": float(n_total - source_frames[0]) / float(fps) if fps > 0 else None,
                "reason": "full",
                "score": None,
            }
        )
    return plan


def build_plan(cfg: InfMoviePipelineConfig, *, pairs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Dispatch clip planning for the requested selection mode."""
    mode = (cfg.mode or "manual").strip().lower()
    if mode == "manual":
        return build_plan_manual(cfg, pairs=pairs)
    if mode == "interesting":
        return build_plan_interesting(cfg, pairs=pairs)
    if mode == "qc":
        return build_plan_qc(cfg, pairs=pairs)
    if mode == "full":
        return build_plan_full(cfg, pairs=pairs)
    raise ValueError(f"Unknown infmovie mode: {cfg.mode}")


# Rendering

def _render_identity(payload):
    """Identify the sources, original frame range and display used for an overlay."""
    from autobehaver.completion import request_identity
    inputs = [payload['inf_csv_path'], payload['movie_path']]
    provenance = Path(str(payload['inf_csv_path']) + '.provenance.json')
    if provenance.is_file():
        inputs.append(provenance)
    defaults = {
        'start_frame': 0, 'end_frame_exclusive': None, 'source_frame_offset': None,
        'skip_first_n_frames': 5, 'timeline': 'none', 'timeline_compare': False,
        'timeline_labels': 'predicted', 'timeline_height': 18, 'white_mode': False,
        'speed_factor': 1.0, 'max_output_fps': 40.0, 'show_timestamp': False,
        'timestamp_position': 'top_right', 'timestamp_decimals': 1,
        'timestamp_reference': 'movie',
        'fps_in': None, 'default_movie_fps': 40.0,
    }
    settings = {key: payload.get(key, value) for key, value in defaults.items()}
    folder = Path(__file__).resolve().parent
    return request_identity('infmovie', inputs, settings,
                            code_paths=[folder/'movies.py', folder/'frame_map.py'])


def _render_worker(payload: Dict[str, Any]) -> Tuple[str, bool, str]:
    """Worker entrypoint returning ``(output_path, ok, message)``."""

    try:
        out_path = str(payload.get("output_movie_path") or "")
        requested_identity = payload.get('_render_identity') or _render_identity(payload)
        if _render_identity(payload) != requested_identity:
            raise RuntimeError('Overlay inputs changed after this render was planned')

        produced = render_overlay_movie(
            str(payload["inf_csv_path"]),
            str(payload["movie_path"]),
            str(payload["output_movie_path"]),
            start_frame=int(payload.get("start_frame", 0)),
            end_frame_exclusive=int(payload.get("end_frame_exclusive")) if payload.get("end_frame_exclusive") is not None else None,
            source_frame_offset=payload.get("source_frame_offset"),
            skip_first_n_frames=int(payload.get("skip_first_n_frames", 5)),
            timeline=str(payload.get("timeline", "none")),
            timeline_compare=bool(payload.get("timeline_compare", False)),
            timeline_labels=str(payload.get("timeline_labels", "predicted")),
            timeline_height=int(payload.get("timeline_height", 18)),
            white_mode=bool(payload.get("white_mode", False)),
            speed_factor=float(payload.get("speed_factor", 1.0)),
            max_output_fps=float(payload.get("max_output_fps", 40.0)),
            show_timestamp=bool(payload.get("show_timestamp", False)),
            timestamp_position=str(payload.get("timestamp_position", "top_right")),
            timestamp_decimals=int(payload.get("timestamp_decimals", 1)),
            timestamp_reference=str(payload.get("timestamp_reference", "movie")),
            atomic_write=True,
            csv_summary=payload.get('_csv_summary'),
            resolved_fps=payload.get('fps_in'),
            default_movie_fps=float(payload.get('default_movie_fps', 40.0)),
        )
        if not isinstance(produced, dict) or produced.get('output_movie_path') != out_path or produced.get('frames', 0) < 1:
            raise RuntimeError('Overlay renderer did not produce the requested artifact')
        ok, reason, _details = _inspect_rendered_movie(Path(out_path))
        if not ok:
            raise RuntimeError(reason)
        if _details['frames'] != produced['frames']:
            raise RuntimeError('Rendered movie differs from its completed frame schedule')
        if _render_identity(payload) != requested_identity:
            raise RuntimeError('Overlay inputs changed during rendering')
        from autobehaver.completion import write_receipt
        write_receipt(out_path, requested_identity, status='complete')
        return out_path, True, out_path
    except Exception as e:
        return str(payload.get("output_movie_path") or ""), False, str(e)


from autobehaver.completion import verified_run


@verified_run
def _render_recording_group(payloads):
    """Render a recording's selected clips using one checked CSV row index."""
    from autobehaver.preinf.autocropper.thread_limits import configure_native_threading
    configure_native_threading(1)
    try:
        first = payloads[0]
        summary = read_csv_summary(first['inf_csv_path'], source_frame_offset=first.get('source_frame_offset'))
        return [_render_worker({**payload, '_csv_summary': summary}) for payload in payloads]
    except Exception as exc:
        return [(str(payload.get('output_movie_path') or ''), False, str(exc)) for payload in payloads]


def _default_plan_paths(output_dir: Path) -> Tuple[Path, Path]:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    plans_dir = output_dir / "plans"
    _ensure_dir(plans_dir)
    plan_csv = plans_dir / f"infmovie_plan_{ts}.csv"
    run_log = plans_dir / f"infmovie_run_{ts}.json"
    return plan_csv, run_log


def _write_plan_csv(plan_rows: List[Dict[str, Any]], path: Path) -> None:
    if not plan_rows:
        return
    fieldnames: List[str] = []
    for r in plan_rows:
        for k in r.keys():
            if k not in fieldnames:
                fieldnames.append(k)

    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in plan_rows:
            w.writerow(r)


def _write_run_log(
    cfg: InfMoviePipelineConfig,
    *,
    plan_csv: Path,
    path: Path,
    n_pairs: int,
    n_plan: int,
    validation: StageValidationReport | None = None,
) -> None:
    """Write resolved overlay settings, selected-plan identity, and recorded validation
    outcomes.
    """
    payload = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "infmovie_config": asdict(cfg),
        "plan_csv": str(plan_csv),
        "n_pairs": int(n_pairs),
        "n_plan": int(n_plan),
        "status": validation.status if validation is not None else "planned",
        "validation_report": str(path.parent.parent / "INFMOVIE_VALIDATION.json")
        if validation is not None
        else None,
    }
    try:
        with path.open("w") as f:
            json.dump(payload, f, indent=2)
    except Exception as e:
        logger.warning("Could not write infmovie run log: %s (%s)", path, e)


def _assign_output_paths(cfg: InfMoviePipelineConfig, *, output_dir: Path, plan_rows: List[Dict[str, Any]]) -> None:
    """Assign portable clip names and output paths to the selected plan."""
    from autobehaver.portable_paths import generated_path, original_name
    """Set output_movie_path for each plan row."""

    mode = (cfg.mode or "manual").strip().lower()
    mode_dir = output_dir / mode
    _ensure_dir(mode_dir)

    for idx, r in enumerate(plan_rows):
        rid = _sanitize_filename_component(str(r.get("recording_id") or f"rec{idx}"), max_len=4096)
        s = _sec_tag(float(r.get("start_sec") or 0.0), decimals=1)
        e = _sec_tag(float(r.get("end_sec") or 0.0), decimals=1)
        speed_tag = ""
        try:
            if abs(float(cfg.speed_factor) - 1.0) > 1e-9:
                speed_tag = f"__{float(cfg.speed_factor):g}x"
        except Exception:
            speed_tag = ""

        import hashlib
        source_key = [str(r.get('recording_id') or ''),
                      *[original_name(str(r.get(key) or '')) for key in ('movie_path', 'inf_csv_path')],
                      int(r.get('start_frame', 0)), r.get('end_frame_exclusive')]
        token = hashlib.sha256(json.dumps(source_key, separators=(',', ':')).encode()).hexdigest()[:16]
        name = f"{rid}__t{s}-{e}s{speed_tag}__{mode}_{token}.mp4"
        # Bound generated clip names for portable paths.
        name = _sanitize_filename_component(name, max_len=len(name)+1)
        if not name.lower().endswith(".mp4"):
            name = name + ".mp4"

        out_path = generated_path(mode_dir, name).resolve()

        r["output_movie_path"] = str(out_path)
        # Keep a relpath for portability.
        try:
            r["output_movie_relpath"] = relpath_str(out_path, output_dir.parent)
        except Exception:
            r["output_movie_relpath"] = str(out_path)
    _check_unique_outputs(plan_rows)


def _check_unique_outputs(plan_rows):
    """Reject duplicate destinations before any rendering or receipt publication."""
    seen = set()
    for row in plan_rows:
        output = str(row.get('output_movie_path') or '')
        if not output:
            raise ValueError('InfMovie plan lacks an output_movie_path')
        key = os.path.normcase(str(Path(output).expanduser().resolve())).casefold()
        if key in seen:
            raise ValueError(f'Duplicate InfMovie output destination: {output}')
        seen.add(key)


def _inspect_rendered_movie(path: Path) -> tuple[bool, str, dict[str, Any]]:
    """Require a readable movie with valid timing and frame metadata."""
    if not path.is_file():
        return False, "rendered movie is missing", {}
    try:
        size = int(path.stat().st_size)
    except OSError as exc:
        return False, f"rendered movie cannot be inspected: {exc}", {}
    if size <= 1024:
        return False, f"rendered movie is implausibly small ({size} bytes)", {"size_bytes": size}
    try:
        import cv2

        capture = open_video_capture(str(path))
        opened = capture.isOpened()
        frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        readable, _frame = capture.read() if opened else (False, None)
        capture.release()
    except Exception as exc:
        return False, f"rendered movie cannot be decoded: {exc}", {"size_bytes": size}
    if not opened or not readable or frames < 1 or not np.isfinite(fps) or fps <= 0:
        return False, "rendered movie has no readable frames or valid frame rate", {
            "size_bytes": size,
            "frames": frames,
            "fps": fps,
        }
    return True, "ok", {"size_bytes": size, "frames": frames, "fps": fps}


@verified_run
def render_plan(
    cfg: InfMoviePipelineConfig,
    *,
    plan_rows: List[Dict[str, Any]],
) -> StageValidationReport:
    """Render selected clips, verify existing or new outputs, and report each clip's
    outcome.
    """
    cfg = replace(cfg, output_dir=str(Path(str(cfg.output_dir)).expanduser().resolve()))
    _check_unique_outputs(plan_rows)
    validation = StageValidationReport.for_items(
        "infmovie",
        [str(row.get("output_movie_path") or "") for row in plan_rows],
        metadata={"mode": str(cfg.mode), "output_directory": str(cfg.output_dir)},
    )
    if not plan_rows:
        validation.add_issue(
            "empty_render_plan",
            "error",
            "The selected inputs and mode produced no movies to render.",
        )
        validation.write(Path(str(cfg.output_dir)) / "INFMOVIE_VALIDATION.json")
        validation.raise_for_errors()

    # Build render payloads.
    payloads: List[Dict[str, Any]] = []
    skipped_paths: set[str] = set()
    expected_identities = {}
    planning_errors = {}
    for r in plan_rows:
        out_path = Path(str(r.get("output_movie_path") or ""))
        payload = {
                **r,
                "source_frame_offset": cfg.source_frame_offset,
                "skip_first_n_frames": int(cfg.skip_first_n_frames),
                "timeline": str(cfg.timeline),
                "timeline_compare": bool(cfg.timeline_compare),
                "timeline_labels": str(cfg.timeline_labels),
                "timeline_height": int(cfg.timeline_height),
                "white_mode": bool(cfg.white_mode),
                "speed_factor": float(cfg.speed_factor),
                "max_output_fps": float(cfg.max_output_fps),
                "show_timestamp": bool(cfg.show_timestamp),
                "timestamp_position": str(cfg.timestamp_position),
                "timestamp_decimals": int(cfg.timestamp_decimals),
                "timestamp_reference": str(cfg.timestamp_reference),
                "default_movie_fps": float(cfg.default_movie_fps),
            }
        try:
            expected = _render_identity(payload)
            expected_identities[str(out_path)] = expected
            payload['_render_identity'] = expected
        except Exception as exc:
            planning_errors[str(out_path)] = (False, f'Cannot verify overlay inputs: {exc}')
            continue
        if (not cfg.clobber) and out_path.exists():
            from autobehaver.completion import receipt_matches
            ok, reason, _details = _inspect_rendered_movie(out_path)
            if ok and receipt_matches(out_path, expected):
                skipped_paths.add(str(out_path))
                continue
            print(f"[infmovie] existing movie does not match the current sources/settings or completed output; re-rendering: {out_path}")
        payloads.append(payload)

    if skipped_paths:
        print(f"[infmovie] skipped {len(skipped_paths)} validated, already-rendered outputs")

    results_by_output: dict[str, tuple[bool, str]] = dict(planning_errors)

    groups = {}
    for payload in payloads:
        key = (payload['inf_csv_path'], payload['movie_path'], payload.get('source_frame_offset'))
        groups.setdefault(key, []).append(payload)
    grouped_payloads = list(groups.values())
    n_workers = worker_count(cfg.num_workers, len(grouped_payloads))

    if n_workers == 1:
        for group in progress(grouped_payloads, desc='Rendering annotated recordings'):
            for output, ok, msg in _render_recording_group(group):
                results_by_output[output] = (ok, msg)
                if not ok:
                    print(f"[infmovie] ERROR rendering {output}: {msg}")
    elif payloads:
        with Pool(processes=n_workers) as pool:
            for results in progress(pool.imap_unordered(_render_recording_group, grouped_payloads), total=len(grouped_payloads), desc='Rendering annotated recordings'):
                for output, ok, msg in results:
                    results_by_output[output] = (ok, msg)
                    if not ok:
                        print(f"[infmovie] ERROR rendering {output}: {msg}")

    for row in plan_rows:
        output = str(row.get("output_movie_path") or "")
        result = results_by_output.get(output)
        if result is not None and not result[0]:
            validation.add_outcome(output, "failed", reason=result[1])
            continue
        ok, reason, details = _inspect_rendered_movie(Path(output))
        from autobehaver.completion import receipt_matches
        if ok and not receipt_matches(output, expected_identities[output]):
            ok, reason = False, 'rendered movie does not match its current sources/settings and completed output record'
        validation.add_outcome(
            output,
            "failed" if not ok else ("reused" if output in skipped_paths else "succeeded"),
            reason=reason if not ok else ("existing movie and its sources/settings matched" if output in skipped_paths else None),
            details=details,
        )

    validation.write(Path(str(cfg.output_dir)) / "INFMOVIE_VALIDATION.json")
    validation.raise_for_errors()
    return validation


# Public orchestrator

def run_infmovie_pipeline(cfg: InfMoviePipelineConfig) -> Path:
    """Run inference overlays with folder or registry discovery.

    Returns:
        Path to the plan CSV written for this run.
    """

    # Resolve output directory.
    if not cfg.output_dir:
        raise ValueError("infmovie.output_dir must be set (e.g. 7_infmovies)")

    out_dir = Path(cfg.output_dir).expanduser().resolve()
    cfg = replace(cfg, output_dir=str(out_dir))
    _ensure_dir(out_dir)

    # Discover pairs.
    pairs: List[Dict[str, Any]] = []

    if cfg.registry_enabled and cfg.registry_csv:
        reg_path = Path(cfg.registry_csv).expanduser().resolve()
        if not reg_path.exists():
            raise FileNotFoundError(f"Registry enabled but CSV not found: {reg_path}")
        project_root = reg_path.parent
        pairs = discover_pairs_from_registry(
            registry_csv=reg_path,
            project_root=project_root,
            file_filter=cfg.file_filter,
        )
        if not pairs:
            raise FileNotFoundError(
                "No (cropped_movie, posepred_inference_csv) pairs found in registry. "
                "Run the pipeline through posepred inference first, or disable registry."
            )
        logger.info("infmovie: discovered %d pairs from registry", len(pairs))
    else:
        if not cfg.inference_csv_dir or not cfg.movie_dir:
            raise ValueError(
                "Without registry, infmovie requires inference_csv_dir and movie_dir"
            )
        pairs = discover_pairs_from_folders(
            inference_csv_dir=Path(cfg.inference_csv_dir).expanduser().resolve(),
            movie_dir=Path(cfg.movie_dir).expanduser().resolve(),
            file_filter=cfg.file_filter,
        )
        logger.info("infmovie: discovered %d pairs from folders", len(pairs))

    # Build plan.
    plan_rows = build_plan(cfg, pairs=pairs)

    # Assign output paths.
    _assign_output_paths(cfg, output_dir=out_dir, plan_rows=plan_rows)

    # Write plan CSV.
    plan_csv, run_log = _default_plan_paths(out_dir)
    if cfg.write_plan_csv:
        _write_plan_csv(plan_rows, plan_csv)
        print(f"[infmovie] Wrote plan: {plan_csv}")

    # Render.
    validation = render_plan(cfg, plan_rows=plan_rows)

    if cfg.write_run_log:
        _write_run_log(
            cfg,
            plan_csv=plan_csv,
            path=run_log,
            n_pairs=len(pairs),
            n_plan=len(plan_rows),
            validation=validation,
        )

    return plan_csv
