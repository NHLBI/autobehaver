"""Movie cropping utilities.

This module contains the per-chamber cropping logic.

Notes:
  - Cropping is parallelized with a ThreadPoolExecutor.
  - Encoding is performed directly via ffmpeg subprocesses so CRF and timeout
    handling work reliably across platforms.
  - ffmpeg/OpenCV thread counts are intentionally capped by default because this
    stage is often run inside Slurm/swarm arrays.
"""

from __future__ import annotations

import concurrent.futures
import logging
import os
import subprocess
import threading
from pathlib import Path
from typing import Any

import cv2
from imageio_ffmpeg import get_ffmpeg_exe


def _get_ffmpeg_executable() -> str:
    """Return the ffmpeg executable path."""
    try:
        return str(get_ffmpeg_exe())
    except Exception:
        return "ffmpeg"


def _get_video_dimensions(video_path: str | Path) -> tuple[int, int]:
    cap = cv2.VideoCapture(str(video_path))
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


def _unique_partial_path(output_path: Path) -> Path:
    """Return a process/thread-unique temp output path next to ``output_path``."""

    return output_path.with_name(
        f"{output_path.name}.pid{os.getpid()}.tid{threading.get_ident()}.partial"
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
) -> str | None:
    """Crop and encode a single chamber movie.

    We write to a unique temporary file first and then atomically rename into
    place. This prevents downstream pipeline stages from seeing partially-written
    MP4s and avoids two swarm subjobs clobbering the same ``.partial`` file.
    """

    output_path = Path(output_path)
    chamber_id = output_path.stem.split("_")[-1]
    if output_path.exists():
        logging.info(f"Skipping existing output: {output_path.name}")
        return str(output_path)

    tmp_path = _unique_partial_path(output_path)
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
            # Limit decoder/native ffmpeg threads. Without this, every chamber
            # crop can spawn additional threads, which explodes inside swarm.
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
            "-f",
            "mp4",
            str(tmp_path),
        ]
        completed = subprocess.run(
            cmd,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=max(int(timeout), 1),
        )
        if completed.stderr:
            logging.debug(completed.stderr.strip())

        if not tmp_path.exists():
            raise RuntimeError(f"Expected temp output was not created: {tmp_path}")

        if output_path.exists():
            try:
                tmp_path.unlink()
            except Exception:
                pass
            logging.info(
                f"Cropped {chamber_id} from {video_path} but final output already existed; keeping {output_path}"
            )
            return str(output_path)

        os.replace(str(tmp_path), str(output_path))
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

    if tmp_path.exists():
        try:
            tmp_path.unlink()
        except Exception:
            pass
    return None


def crop_chambers_in_parallel(
    movie_path: Path,
    occupied_coords: dict[str, tuple[int, int]],
    output_paths_by_chamber: dict[str, Path],
    crop_dim: int,
    args: Any,
    *,
    source_size: tuple[int, int] | None = None,
) -> dict[str, str]:
    cropped_files: dict[str, str] = {}
    timeout = int(getattr(args, "timeout", 600))
    crf = int(getattr(args, "crop_crf", 12))
    ffmpeg_threads = max(1, int(getattr(args, "ffmpeg_threads", 1)))
    max_workers = max(1, int(getattr(args, "max_workers", 6)))

    # Prefer the dimensions already obtained during chamber detection. Falling
    # back to one OpenCV open per movie is still much cheaper than one open per
    # chamber and reduces h264 decoder exhaustion under heavy parallelism.
    if source_size is None:
        try:
            source_size = _get_video_dimensions(movie_path)
        except Exception as exc:
            logging.exception(f"Could not read source dimensions for {movie_path}")
            for chamber_id in occupied_coords:
                if chamber_id in output_paths_by_chamber:
                    cropped_files[chamber_id] = f"failed: {exc}"
            return cropped_files

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_chamber = {
            executor.submit(
                crop_single_chamber,
                str(movie_path),
                Path(output_paths_by_chamber[chamber_id]),
                x,
                y,
                crop_dim,
                timeout=timeout,
                crf=crf,
                ffmpeg_threads=ffmpeg_threads,
                source_size=source_size,
            ): chamber_id
            for chamber_id, (x, y) in occupied_coords.items()
            if chamber_id in output_paths_by_chamber
        }

        for future in concurrent.futures.as_completed(future_to_chamber):
            chamber_id = future_to_chamber[future]
            try:
                result_path = future.result()
                if result_path is not None:
                    cropped_files[chamber_id] = Path(result_path).name
                else:
                    cropped_files[chamber_id] = "failed: cropping returned None"
            except Exception as e:
                logging.error(f"Failed to crop {chamber_id}: {e}")
                cropped_files[chamber_id] = f"failed: {str(e)}"

    return cropped_files
