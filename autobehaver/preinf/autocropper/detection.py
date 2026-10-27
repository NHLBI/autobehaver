"""Detect circles and assign chamber identities for the standard camera layout."""

from __future__ import annotations

import logging
import itertools
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .thread_limits import open_video_capture
from .grid_fit import circle_support, fit_grid


def get_frame_count(video_path: Path) -> int:
    """Return the frame count reported by the video decoder."""
    cap = open_video_capture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return frame_count


def average_frames(video_path: str | Path, frame_avg: int) -> np.ndarray:
    """Average evenly spaced decoded frames into a uint8 BGR image."""
    cap = open_video_capture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []

    for i in range(frame_avg):
        frame_id = int(frame_count / frame_avg * i)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
        else:
            logging.warning(f"Failed to read frame {frame_id} from {video_path}")

    cap.release()

    if not frames:
        raise ValueError(f"Error: No frames read from {video_path}")

    avg_frame = np.mean(frames, axis=0).astype(np.uint8)
    return avg_frame


def downsample_image(image: np.ndarray, down_factor: int) -> np.ndarray:
    """Downsample an image by an integer factor using area interpolation."""
    if down_factor == 1:
        return image
    height, width = image.shape[:2]
    new_dimensions = (width // down_factor, height // down_factor)
    return cv2.resize(image, new_dimensions, interpolation=cv2.INTER_AREA)


def adjust_contrast(image: np.ndarray) -> np.ndarray:
    """Stretch the grayscale intensity range to 0..255."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    min_val = np.min(gray)
    max_val = np.max(gray)
    stretched = ((gray - min_val) / (max_val - min_val) * 255).astype(np.uint8)
    return stretched


def detect_circles(image: np.ndarray, min_radius: int, max_radius: int) -> tuple[Any, np.ndarray]:
    """Return Hough circle detections and the contrast-adjusted grayscale image."""
    gray = adjust_contrast(image)
    gray = cv2.medianBlur(gray, 5)
    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=100,
        param1=50,
        param2=30,
        minRadius=min_radius,
        maxRadius=max_radius,
    )
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        return circles, gray
    return [], gray


def calculate_theoretical_centers(
    image_width: int, image_height: int, chamber_columns: int, chamber_rows: int,
    calibration: dict | None = None,
) -> dict[str, tuple[int, int]]:
    # Empirical centers apply only to the supported six-chamber camera layout.
    """Return empirical or calibrated pixel centers for the supported six-chamber layout."""
    empirical_centers = {
        "chamber1": (373, 449),
        "chamber2": (996, 443),
        "chamber3": (1622, 440),
        "chamber4": (378, 1073),
        "chamber5": (1001, 1070),
        "chamber6": (1626, 1064),
    }
    if (image_width, image_height) not in ((2016, 1520), (2028, 1520)):
        raise ValueError(
            'Unsupported movie size: Autocropper supports standard recording sizes '
            '2016x1520 and 2028x1520 only.'
        )
    if (chamber_columns, chamber_rows) != (3, 2):
        raise ValueError('The empirical camera layout supports exactly3 columns and2 rows.')
    if calibration is not None:
        if (int(calibration['image_width']), int(calibration['image_height'])) != (image_width, image_height):
            raise ValueError('Camera calibration resolution does not match the movie; no automatic FOV scaling.')
        centers = calibration['centers']
        if set(centers) != set(empirical_centers):
            raise ValueError('Camera calibration must declare exactly chamber1 through chamber6.')
        return {k: tuple(map(float, centers[k])) for k in empirical_centers}
    return empirical_centers


def match_circles_to_chambers(
    detected_circles: Any, theoretical_centers: dict[str, tuple[int, int]], down_factor: int,
    *, max_center_distance_px: float | None = None,
) -> dict[str, tuple[int, int]]:
    """Assign detected circles to chamber IDs only when bounded matching is unique.

    Return centers in full-resolution pixels.
    """
    circles = np.asarray(detected_circles, dtype=float)
    ids = list(theoretical_centers)
    if circles.ndim != 2 or circles.shape[1] != 3 or not np.isfinite(circles).all() or np.any(circles[:,2] <= 0):
        raise ValueError('Chamber detections must contain finite centers and positive radii.')
    if int(down_factor) != down_factor or down_factor <= 0:
        raise ValueError('down_factor must be a positive integer.')
    full = circles[:,:2] * down_factor
    expected = np.asarray([theoretical_centers[k] for k in ids], dtype=float)
    distances = np.linalg.norm(expected[:,None,:] - full[None,:,:], axis=2)
    if max_center_distance_px is None or not np.isfinite(max_center_distance_px) or max_center_distance_px <= 0:
        raise ValueError('A calibrated positive max_center_distance_px is required; unbounded matching is prohibited.')
    # Hough can report circles between chambers. Ignore only detections outside
    # every bounded chamber region; duplicates within a region remain an error.
    within_layout = np.any(distances <= max_center_distance_px, axis=0)
    full = full[within_layout]
    distances = distances[:, within_layout]
    if len(full) != len(ids):
        raise ValueError(
            f'Exactly {len(ids)} chamber-region detections are required; found {len(full)} '
            f'from {len(circles)} total circles.'
        )
    assignments = []
    for order in itertools.permutations(range(len(ids))):
        chosen = distances[np.arange(len(ids)), order]
        if np.all(chosen <= max_center_distance_px):
            assignments.append(order)
    # Unique bounded matching is stricter than selecting an arbitrary closest
    # global assignment: two admissible assignments mean ambiguous identity.
    if len(assignments) != 1:
        raise ValueError(f'Chamber geometry is missing/outlying/ambiguous: {len(assignments)} admissible assignments.')
    return {chamber: tuple(map(int, full[index])) for chamber,index in zip(ids,assignments[0])}


def draw_circles(
    image: np.ndarray,
    circles: Any,
    down_factor: int,
    matched_centers: dict[str, tuple[int, int]],
    *,
    approx_radius: int,
) -> None:
    """Draw chamber centers, circle boundaries, and labels on the image in place."""
    font_scale = 1.0
    font_thickness = 2
    for chamber_id, (x, y) in matched_centers.items():
        for (cx, cy, r) in circles:
            if (x // down_factor == cx) and (y // down_factor == cy):
                r_full = r * down_factor
                break
        else:
            r_full = approx_radius

        cv2.circle(image, (x, y), r_full, (0, 255, 0), 2)
        cv2.drawMarker(
            image,
            (x, y),
            (0, 0, 255),
            markerType=cv2.MARKER_CROSS,
            markerSize=10,
            thickness=2,
        )

        text_size, _ = cv2.getTextSize(
            chamber_id, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness
        )
        text_x = x - text_size[0] // 2
        text_y = y + int(image.shape[0] * 0.05) + text_size[1] // 2

        cv2.putText(
            image,
            chamber_id,
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 0, 0),
            font_thickness,
        )


def detect_and_match_chambers(movie_path: Path, args: Any):
    """Detect and uniquely match circles to the standard chamber IDs.

    Return matched full-resolution centers, the averaged BGR frame, downsampled
    circle detections, and the crop side length in pixels.
    """
    avg_frame = average_frames(str(movie_path), args.frame_avg)
    downsampled_frame = downsample_image(avg_frame, args.down_factor)

    image_height, image_width = avg_frame.shape[:2]

    calibration = None
    calibration_path = getattr(args, 'chamber_calibration_json', None)
    if calibration_path:
        with open(calibration_path, encoding='utf-8') as handle:
            calibration = json.load(handle)
    # Unknown camera layouts fail before assigning chamber identities.
    empirical_centers = calculate_theoretical_centers(
        image_width, image_height, args.chamber_columns, args.chamber_rows, calibration)
    crop_dim = args.crop_dim_2k
    min_radius = int((args.approx_radius * 0.9) // args.down_factor)
    max_radius = int((args.approx_radius * 1.1) // args.down_factor)

    circles, gray = detect_circles(downsampled_frame, min_radius, max_radius)
    scores, edges = circle_support(gray, circles)
    diagnostic = getattr(args, '_detection_diagnostic_dir', None)
    report = {'candidate_scores': scores.tolist(), 'opencv': cv2.__version__}
    candidate_frame = avg_frame.copy()
    for index, (x, y, radius) in enumerate(circles):
        center = (int(x * args.down_factor), int(y * args.down_factor))
        cv2.circle(candidate_frame, center, int(radius * args.down_factor), (0, 180, 255), 2)
        cv2.putText(candidate_frame, f'{index + 1}: {scores[index]:.2f}', center,
                    cv2.FONT_HERSHEY_SIMPLEX, .7, (0, 0, 180), 2)
    if diagnostic:
        directory = Path(diagnostic); directory.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(directory / (movie_path.stem + '_candidates.png')), candidate_frame)
        cv2.imwrite(str(directory / (movie_path.stem + '_edges.png')), edges)
    try:
        matched_centers, fit = fit_grid(circles, empirical_centers, args.down_factor,
                                       avg_frame.shape, scores)
        report.update(fit)
    except Exception as error:
        report['error'] = str(error)
        raise
    finally:
        args._last_grid_fit = report
        if diagnostic:
            (Path(diagnostic) / (movie_path.stem + '_grid.json')).write_text(
                json.dumps(report, indent=2), encoding='utf-8')
    for chamber in report['inferred_chambers']:
        logging.warning('Inferred %s center from fitted grid; review the diagnostic image.', chamber)
        cv2.putText(avg_frame, chamber + ' INFERRED', matched_centers[chamber],
                    cv2.FONT_HERSHEY_SIMPLEX, .7, (0, 100, 255), 2)

    expected_chambers = args.chamber_columns * args.chamber_rows
    if len(matched_centers) != expected_chambers:
        raise RuntimeError(
            f"Failed to match all {expected_chambers} chambers. Only matched {len(matched_centers)}."
        )

    logging.info(f"Detected and matched {len(matched_centers)} chambers")

    return matched_centers, avg_frame, circles, crop_dim
