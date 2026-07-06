"""Filesystem + logging helpers for autocropper."""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np


def setup_logging(log_directory: Path, log_level: str = "INFO") -> None:
    """Configure logging to file and console."""
    log_file_path = Path(log_directory) / "autocropper.log"

    logger = logging.getLogger()
    level_name = (log_level or "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    logger.setLevel(level)

    if logger.hasHandlers():
        logger.handlers.clear()

    file_handler = logging.FileHandler(log_file_path)
    file_handler.setLevel(level)
    file_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_formatter = logging.Formatter("%(message)s")
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)


def find_processing_log(search_dir: Path) -> Path | None:
    """Find the main pipeline processing log file in the input directory."""
    for file_path in Path(search_dir).glob("*_processing_log.txt"):
        return file_path
    return None


def ensure_user_permissions(path: Path) -> None:
    """Ensure standard user rwx permissions on Windows."""
    if os.name == "nt":
        import stat

        os.chmod(path, stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR)


def get_movie_files(search_dir: Path) -> list[Path]:
    """Return a sorted list of all MP4 files in the input directory."""
    return sorted(list(Path(search_dir).glob("*.mp4")))


def is_video_file_valid(video_path: Path) -> bool:
    """Quick check to see if a video file is valid and readable by OpenCV."""
    try:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            logging.warning(f"Validation failed: Could not open video file {video_path}.")
            cap.release()
            return False

        ret, _frame = cap.read()
        cap.release()

        if not ret:
            logging.warning(f"Validation failed: Could not read a frame from {video_path}.")
            return False

        return True
    except Exception as e:
        logging.error(
            f"An unexpected error occurred during video file validation for {video_path}: {e}"
        )
        return False


def save_chamber_coordinates_image(run_dir: Path, movie: Path, image: np.ndarray) -> None:
    chamber_images_dir = Path(run_dir) / "chamber_images"
    chamber_images_dir.mkdir(parents=True, exist_ok=True)
    ensure_user_permissions(chamber_images_dir)
    output_path = chamber_images_dir / (Path(movie).stem + "_circles.jpg")
    cv2.imwrite(str(output_path), image)
    ensure_user_permissions(output_path)


def log_movie_results(movie_path: Path, run_dir: Path, movie_data: dict[str, Any]) -> None:
    movie_name = Path(movie_path).stem
    json_path = Path(run_dir) / "movie_logs" / f"{movie_name}_log.json"

    serializable_data = json.loads(
        json.dumps(
            movie_data,
            default=lambda x: float(x)
            if isinstance(x, (np.float32, np.float64))
            else int(x)
            if isinstance(x, np.integer)
            else x,
        )
    )

    with json_path.open("w") as f:
        json.dump(serializable_data, f, indent=4)
    logging.info(f"Saved processing log to {json_path}")


def _file_matches_any_movie_stem(file_path: Path, movie_stems: set[str] | None) -> bool:
    if not movie_stems:
        return True
    name = file_path.name
    return any(stem and stem in name for stem in movie_stems)


def delete_incomplete_files(
    output_dir: Path,
    *,
    movie_stems: Iterable[str] | None = None,
    min_partial_age_seconds: int = 0,
) -> None:
    """Delete stale/broken cropped-video outputs.

    Parameters
    ----------
    movie_stems:
        Optional scope. When several swarm subjobs share an output directory,
        cleanup must not delete another subjob's active temp files. Passing the
        current run's movie stems keeps cleanup local to the selected movies.
    min_partial_age_seconds:
        Only delete ``*.partial`` files older than this age. This protects active
        ffmpeg subprocesses from another concurrently running Python process.
    """

    scoped_stems = {str(s) for s in movie_stems or [] if str(s)} or None
    min_partial_age_seconds = max(0, int(min_partial_age_seconds))
    now = time.time()

    for file_path in Path(output_dir).rglob("*"):
        if not file_path.is_file():
            continue
        if not _file_matches_any_movie_stem(file_path, scoped_stems):
            continue

        if file_path.name.endswith(".partial"):
            try:
                age_seconds = now - file_path.stat().st_mtime
            except Exception:
                age_seconds = min_partial_age_seconds + 1
            if age_seconds < min_partial_age_seconds:
                logging.debug(
                    f"Leaving active/recent temp video file in place: {file_path} "
                    f"(age={age_seconds:.0f}s < {min_partial_age_seconds}s)"
                )
                continue
            logging.info(f"Deleting stale temp video file: {file_path}")
            try:
                file_path.unlink()
            except Exception:
                pass
            continue

        if file_path.suffix.lower() != ".mp4":
            continue
        if file_path.stat().st_size < 1024:
            logging.info(f"Deleting incomplete video file: {file_path}")
            file_path.unlink()


def _load_movie_logs(
    movie_logs_dir: Path,
    *,
    movie_names: Iterable[str] | None = None,
) -> list[dict[str, Any]]:
    """Load per-movie JSON logs, optionally filtered to the current run subset."""
    selected_names = {str(name).strip() for name in movie_names or [] if str(name).strip()}
    logs: list[dict[str, Any]] = []

    if not movie_logs_dir.exists():
        return logs

    for json_file in sorted(movie_logs_dir.glob("*_log.json")):
        try:
            with json_file.open() as f:
                data = json.load(f) or {}
        except Exception:
            logging.warning(f"Skipping unreadable autocropper movie log during summary: {json_file}")
            continue

        movie_name = str(data.get("movie_name") or "").strip()
        if not movie_name:
            continue
        if selected_names and movie_name not in selected_names:
            continue
        logs.append(data)

    logs.sort(key=lambda x: str(x.get("movie_name") or ""))
    return logs


def summarize_results(run_dir: Path, *, movie_names: Iterable[str] | None = None) -> None:
    """Append a robust per-movie summary to the pipeline processing log."""
    log_file = find_processing_log(run_dir)
    movie_logs_dir = Path(run_dir) / "movie_logs"

    if log_file is None:
        return

    movie_logs = _load_movie_logs(movie_logs_dir, movie_names=movie_names)

    with log_file.open("a") as f:
        f.write("\n\nProcessing Results Summary\n")
        f.write("=" * 50 + "\n\n")

        status_counts: dict[str, int] = {}

        for movie_data in movie_logs:
            movie_name = str(movie_data.get("movie_name") or "<unknown>")
            status = str(movie_data.get("status") or ("failed" if "error" in movie_data else "unknown"))
            status_counts[status] = status_counts.get(status, 0) + 1

            f.write(f"Movie: {movie_name}\n")
            f.write("-" * 30 + "\n")
            f.write(f"Status: {status}\n")

            skip_reason = movie_data.get("skip_reason")
            if skip_reason:
                f.write(f"Skip reason: {skip_reason}\n")

            if "error" in movie_data:
                f.write(f"ERROR: {movie_data['error']}\n")

            excluded_chambers = movie_data.get("excluded_chambers") or []
            if excluded_chambers:
                f.write(f"Excluded chambers: {', '.join([str(x) for x in excluded_chambers])}\n")

            chamber_coordinates = movie_data.get("chamber_coordinates") or {}
            if isinstance(chamber_coordinates, dict) and chamber_coordinates:
                f.write("Chamber Coordinates:\n")
                for chamber_id, coords in sorted(chamber_coordinates.items()):
                    if not isinstance(coords, dict):
                        continue
                    f.write(
                        f"  {chamber_id}: x={coords.get('x', 'NA')}, y={coords.get('y', 'NA')}\n"
                    )
            else:
                f.write("Chamber Coordinates: not available\n")
            f.write("\n")

            validation_results = movie_data.get("validation_results") or {}
            if isinstance(validation_results, dict) and validation_results:
                f.write("Chamber Validation Results:\n")
                for chamber_id, result in sorted(validation_results.items()):
                    if not isinstance(result, dict):
                        continue
                    is_occupied = bool(result.get("is_occupied", False))
                    status_text = "OCCUPIED" if is_occupied else "EMPTY"
                    confidence = result.get("confidence")
                    if isinstance(confidence, (int, float)):
                        f.write(
                            f"  {chamber_id}: {status_text} (confidence: {float(confidence):.3f})\n"
                        )
                    else:
                        f.write(f"  {chamber_id}: {status_text}\n")
                f.write("\n")

            cropped_files = movie_data.get("cropped_files") or {}
            f.write("Cropped Chamber Files:\n")
            if isinstance(cropped_files, dict) and cropped_files:
                for chamber_id, file_info in sorted(cropped_files.items()):
                    f.write(f"  {chamber_id}: {file_info}\n")
            else:
                f.write("  none\n")

            crop_failures = movie_data.get("crop_failures") or {}
            if isinstance(crop_failures, dict) and crop_failures:
                f.write("Crop Failures This Attempt:\n")
                for chamber_id, reason in sorted(crop_failures.items()):
                    f.write(f"  {chamber_id}: {reason}\n")

            missing_required = movie_data.get("missing_required_chambers") or []
            if missing_required:
                f.write(
                    "Missing Required Chambers After Attempt: "
                    + ", ".join([str(x) for x in missing_required])
                    + "\n"
                )

            f.write("\n")

        if status_counts:
            f.write("Summary Counts:\n")
            for status, count in sorted(status_counts.items()):
                f.write(f"  {status}: {count}\n")
            f.write("\n")
