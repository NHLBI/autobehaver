"""Encode chamber crops with bounded FFmpeg threads and verified publication.

Requested chambers share a decoder within each split/crop graph.
"""

from __future__ import annotations

from autobehaver.ffmpeg_progress import run_ffmpeg
from autobehaver.progress import progress

import logging
import os
import subprocess
import shutil
import uuid
from autobehaver.completion import preserve_previous, write_receipt, file_signature
from .contract import crop_request, valid_crop, video_shape
from pathlib import Path
from typing import Any

import cv2
try:
    from imageio_ffmpeg import get_ffmpeg_exe
except ImportError:
    get_ffmpeg_exe = None

from .thread_limits import open_video_capture
from .io_utils import crop_partial_token
from autobehaver.portable_paths import original_name
from autobehaver.file_locks import FileLock


def _get_ffmpeg_executable() -> str:
    """Return the ffmpeg executable path."""
    if get_ffmpeg_exe is not None:
        try:
            return str(get_ffmpeg_exe())
        except Exception:
            pass
    executable = shutil.which('ffmpeg')
    if executable:
        return executable
    raise RuntimeError('Cropping requires FFmpeg: install imageio-ffmpeg or place ffmpeg on PATH.')


def _get_video_dimensions(video_path: str | Path) -> tuple[int, int]:
    """Return the source width and height in pixels."""
    cap = open_video_capture(video_path)
    if not cap.isOpened():
        cap.release()
        raise RuntimeError(f"Could not open video for cropping: {video_path}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    if width <= 0 or height <= 0:
        raise RuntimeError(f"Could not determine video dimensions for: {video_path}")
    return width, height


def _resolve_crop_box(
    video_path: str | Path,
    x: int,
    y: int,
    crop_dim: int,
    *,
    source_size: tuple[int, int] | None = None,
) -> tuple[int, int, int]:
    """Return a square crop box shifted inward to stay within the source frame."""
    if source_size is None:
        width, height = _get_video_dimensions(video_path)
    else:
        width, height = int(source_size[0]), int(source_size[1])
    if crop_dim > width or crop_dim > height:
        raise RuntimeError(
            f"Crop dimension {crop_dim} exceeds source frame size {width}x{height} for {video_path}"
        )

    half = crop_dim // 2
    x1 = max(0, int(x) - half)
    y1 = max(0, int(y) - half)
    x1 = min(x1, width - crop_dim)
    y1 = min(y1, height - crop_dim)
    return int(x1), int(y1), int(crop_dim)


def _unique_partial_path(output_path: Path, source_movie=None) -> Path:
    """Return a unique temporary crop path beside the destination."""

    return output_path.with_name(
        f".crop-{crop_partial_token(Path(original_name(source_movie or output_path)).stem)}-{uuid.uuid4().hex[:12]}.partial"
    )


def crop_single_chamber(
    video_path: str,
    output_path: Path,
    x: int,
    y: int,
    crop_dim: int,
    *,
    timeout: int,
    crf: int,
    ffmpeg_threads: int = 1,
    source_size: tuple[int, int] | None = None,
    identity: dict | None = None,
) -> str | None:
    """Encode, validate, and atomically publish one chamber crop.

    Use a unique temporary file so downstream readers see only complete movies.
    Return the output path on success or None on failure.
    """

    output_path = Path(output_path)
    chamber_id = output_path.stem.split("_")[-1]
    if identity is not None and valid_crop(video_path, output_path, identity, crop_dim):
        logging.info(f"Skipping validated output: {output_path.name}")
        return str(output_path)

    tmp_path = _unique_partial_path(output_path, video_path)
    partial_lock = FileLock(tmp_path.with_suffix('.lock'), timeout_sec=0)
    partial_lock.acquire()
    if tmp_path.exists():
        try:
            tmp_path.unlink()
        except Exception:
            logging.warning(f"Could not remove stale temp file {tmp_path}, will overwrite.")

    try:
        x1, y1, crop_side = _resolve_crop_box(
            video_path,
            x,
            y,
            crop_dim,
            source_size=source_size,
        )
        ffmpeg_exe = _get_ffmpeg_executable()
        threads = max(1, int(ffmpeg_threads))
        cmd = [
            ffmpeg_exe,
            "-y",
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            # Cap decoder threads separately from filtering and output encoding.
            "-threads",
            str(threads),
            "-i",
            str(video_path),
            "-map",
            "0:v:0",
            "-filter_threads",
            str(threads),
            "-vf",
            f"crop={crop_side}:{crop_side}:{x1}:{y1}",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-crf",
            str(int(crf)),
            "-threads",
            str(threads),
            "-x264-params",
            f"threads={threads}",
            # Camera timestamps are not perfectly uniform. FFmpeg's default
            # MP4 sync rounds them to CFR and silently drops/duplicates frames.
            # Preserve one output per input frame and the input time base.
            # These options also work with older FFmpeg (before -fps_mode).
            "-vsync",
            "0",
            "-enc_time_base",
            "-1",
            # Crops start at source frame zero. Avoid an MP4 edit list that can
            # hide the final frame when preserving fine-grained timestamps.
            "-use_editlist",
            "0",
            "-f",
            "mp4",
            str(tmp_path),
        ]
        completed = run_ffmpeg(cmd, total_frames=None,
                               desc=f'Cropping {Path(video_path).name} {chamber_id}',
                               timeout=max(int(timeout), 1))
        if completed.stderr:
            logging.debug(completed.stderr.strip())

        if not tmp_path.exists():
            raise RuntimeError(f"Expected temp output was not created: {tmp_path}")

        if not valid_crop(video_path, tmp_path, crop_dim=crop_dim):
            raise ValueError('Encoded crop has incorrect dimensions, frame count or unreadable frames')
        preserve_previous(output_path)
        os.replace(str(tmp_path), str(output_path))
        if identity is not None:
            write_receipt(output_path, identity, crop_box=[x1,y1,crop_side])
        logging.info(
            f"Cropped {chamber_id} from {video_path} and saved to {output_path} (CRF={int(crf)})"
        )
        return str(output_path)
    except subprocess.TimeoutExpired:
        logging.error(
            f"Cropping timed out for {chamber_id} from {video_path} after {max(int(timeout), 1)}s"
        )
    except subprocess.CalledProcessError as exc:
        stderr = (exc.stderr or "").strip()
        logging.error(
            f"ffmpeg failed while cropping {chamber_id} from {video_path}: {stderr or exc}"
        )
    except Exception:
        logging.exception(f"Error cropping {chamber_id} from {video_path}")
    finally:
        partial_lock.release()

    if tmp_path.exists():
        try:
            tmp_path.unlink()
        except Exception:
            pass
    return None


def crop_chambers_in_parallel(
    movie_path: Path,
    chamber_coords: dict[str, tuple[int, int]],
    output_paths_by_chamber: dict[str, Path],
    crop_dim: int,
    args: Any,
    *,
    source_size: tuple[int, int] | None = None,
    identities: dict[str, dict] | None = None,
) -> dict[str, str]:
    """Encode requested chambers from a shared source decode and validate each crop.

    Each encoder retains its thread limit; max_workers bounds the number of
    chambers per FFmpeg graph. Keep the historical function name for callers.
    """
    cropped_files: dict[str, str] = {}
    timeout = int(getattr(args, "timeout", 600))
    crf = int(getattr(args, "crop_crf", 12))
    ffmpeg_threads = max(1, int(getattr(args, "ffmpeg_threads", 1)))
    max_workers = max(1, int(getattr(args, 'max_workers', 6)))

    # Prefer the dimensions already obtained during chamber detection. Falling
    # back to one OpenCV open per movie is still much cheaper than one open per
    # chamber and reduces h264 decoder exhaustion under heavy parallelism.
    if source_size is None:
        try:
            source_size = _get_video_dimensions(movie_path)
        except Exception as exc:
            logging.exception(f"Could not read source dimensions for {movie_path}")
            for chamber_id in chamber_coords:
                if chamber_id in output_paths_by_chamber:
                    cropped_files[chamber_id] = f"failed: {exc}"
            return cropped_files

    chambers = [cid for cid in chamber_coords if cid in output_paths_by_chamber]
    if not chambers:
        return cropped_files
    base_request = None
    if identities is None:
        base_request = crop_request(movie_path, chambers[0], args)
        identities = {cid: crop_request(movie_path, cid, args, base_request=base_request) for cid in chambers}
    # Preserve the explicit chamber concurrency cap. The default six chambers
    # share one decode; a lower user cap processes a few bounded groups.
    if len(chambers) > max_workers:
        for start in range(0, len(chambers), max_workers):
            group = chambers[start:start + max_workers]
            cropped_files.update(crop_chambers_in_parallel(
                movie_path, {cid: chamber_coords[cid] for cid in group},
                {cid: output_paths_by_chamber[cid] for cid in group}, crop_dim, args,
                source_size=source_size, identities=identities))
        return cropped_files
    source_signature = file_signature(movie_path)
    source_shape = video_shape(movie_path)
    pending = []
    for cid in chambers:
        output = Path(output_paths_by_chamber[cid])
        if output.exists() and valid_crop(movie_path, output, identities[cid], crop_dim, source_shape=source_shape):
            cropped_files[cid] = output.name
            continue
        x, y = chamber_coords[cid]
        box = _resolve_crop_box(movie_path, x, y, crop_dim, source_size=source_size)
        output.parent.mkdir(parents=True, exist_ok=True)
        pending.append((cid, output, _unique_partial_path(output, movie_path), box))
    if not pending:
        return cropped_files

    threads = str(ffmpeg_threads)
    labels = ''.join(f'[split{i}]' for i in range(len(pending)))
    graph = [f'[0:v:0]split={len(pending)}{labels}']
    for i, (_cid, _output, _tmp, (x, y, side)) in enumerate(pending):
        graph.append(f'[split{i}]crop={side}:{side}:{x}:{y}[crop{i}]')
    command = [_get_ffmpeg_executable(), '-y', '-nostdin', '-hide_banner', '-loglevel', 'error',
               '-threads', threads, '-i', str(movie_path), '-filter_complex_threads', threads,
               '-filter_complex', ';'.join(graph)]
    for i, (_cid, _output, temporary, _box) in enumerate(pending):
        command.extend(['-map', f'[crop{i}]', '-an', '-c:v', 'libx264', '-preset', 'ultrafast',
                        '-crf', str(crf), '-threads', threads, '-x264-params', f'threads={threads}',
                        '-vsync', '0', '-enc_time_base', '-1', '-use_editlist', '0',
                        '-f', 'mp4', str(temporary)])
    partial_locks = []
    try:
        for _cid, _output, temporary, _box in pending:
            lock = FileLock(temporary.with_suffix('.lock'), timeout_sec=0)
            lock.acquire()
            partial_locks.append(lock)
        run_ffmpeg(command, total_frames=source_shape[0],
                   desc='Encoding chambers ' + ','.join(cid.replace('chamber', '') for cid in chambers) + ' frames',
                   timeout=max(timeout, 1))
        if file_signature(movie_path) != source_signature:
            raise ValueError('Source movie changed during shared crop encoding')
        # A failed member never makes an unchecked movie visible downstream.
        for cid, output, temporary, box in progress(pending, desc='Validating cropped chambers'):
            if not valid_crop(movie_path, temporary, crop_dim=crop_dim, source_shape=source_shape):
                cropped_files[cid] = 'failed: crop frame count, dimensions or decoding failed'
                continue
            preserve_previous(output)
            os.replace(temporary, output)
            write_receipt(output, identities[cid], crop_box=list(box))
            cropped_files[cid] = output.name
    except (subprocess.SubprocessError, OSError, ValueError) as exc:
        detail = (getattr(exc, 'stderr', '') or str(exc)).strip()
        logging.error('Shared-source crop failed for %s: %s', movie_path, detail)
        for cid, _output, _temporary, _box in pending:
            cropped_files.setdefault(cid, f'failed: {detail}')
    finally:
        try:
            for _cid, _output, temporary, _box in pending:
                if temporary.exists():
                    temporary.unlink()
        finally:
            for lock in partial_locks:
                lock.release()

    return cropped_files
