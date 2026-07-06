"""Chamber detection and matching.

This module contains the circle detection + chamber ID assignment logic for the
autocropper stage.

Implementation notes:
  - The current chamber grid coordinates are empirical and tailored to the lab rig.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import cv2
import numpy as np


def get_frame_count(video_path: Path) -> int:
    cap = cv2.VideoCapture(str(video_path))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return frame_count


def average_frames(video_path: str | Path, frame_avg: int) -> np.ndarray:
    cap = cv2.VideoCapture(str(video_path))
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
    if down_factor == 1:
        return image
    height, width = image.shape[:2]
    new_dimensions = (width // down_factor, height // down_factor)
    return cv2.resize(image, new_dimensions, interpolation=cv2.INTER_AREA)


def adjust_contrast(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    min_val = np.min(gray)
    max_val = np.max(gray)
    stretched = ((gray - min_val) / (max_val - min_val) * 255).astype(np.uint8)
    return stretched


def detect_circles(image: np.ndarray, min_radius: int, max_radius: int) -> tuple[Any, np.ndarray]:
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
    image_width: int, image_height: int, chamber_columns: int, chamber_rows: int
) -> dict[str, tuple[int, int]]:
    # Hardcoded empirical centers for the lab's rig.
    # NOTE: These values are currently assumed to be stable for the pipeline.
    empirical_centers = {
        "chamber1": (373, 449),
        "chamber2": (996, 443),
        "chamber3": (1622, 440),
        "chamber4": (378, 1073),
        "chamber5": (1001, 1070),
        "chamber6": (1626, 1064),
    }
    return empirical_centers


def match_circles_to_chambers(
    detected_circles: Any, theoretical_centers: dict[str, tuple[int, int]], down_factor: int
) -> dict[str, tuple[int, int]]:
    matched_centers: dict[str, tuple[int, int]] = {}
    used_circles: set[tuple[int, int]] = set()

    for chamber_id, (theo_x, theo_y) in theoretical_centers.items():
        min_distance = float("inf")
        best_match = None
        for (x, y, r) in detected_circles:
            x_full = x * down_factor
            y_full = y * down_factor
            if (x_full, y_full) in used_circles:
                continue
            distance = np.sqrt((x_full - theo_x) ** 2 + (y_full - theo_y) ** 2)
            if distance < min_distance:
                min_distance = distance
                best_match = (x_full, y_full)
        if best_match:
            matched_centers[chamber_id] = best_match
            used_circles.add(best_match)

    return matched_centers


def draw_circles(
    image: np.ndarray,
    circles: Any,
    down_factor: int,
    matched_centers: dict[str, tuple[int, int]],
    *,
    approx_radius: int,
) -> None:
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
    """Detect circles and match them to chamber IDs.

    Returns:
        matched_centers: dict[chamber_id, (x,y)]
        avg_frame: np.ndarray (used for visualization)
        circles: raw circles output from Hough transform
        crop_dim: crop size to use for this movie (depends on resolution)
    """
    avg_frame = average_frames(str(movie_path), args.frame_avg)
    downsampled_frame = downsample_image(avg_frame, args.down_factor)

    image_height, image_width = avg_frame.shape[:2]

    # Derive the 4K crop dimension from the source video dimensions.
    crop_dim = args.crop_dim_4k if image_width > 3000 else args.crop_dim_2k
    approx_radius_dynamic = args.approx_radius * 2 if image_width > 3000 else args.approx_radius

    min_radius = int((approx_radius_dynamic * 0.9) // args.down_factor)
    max_radius = int((approx_radius_dynamic * 1.1) // args.down_factor)

    circles, _ = detect_circles(downsampled_frame, min_radius, max_radius)

    if circles is None:
        raise RuntimeError(f"Failed to detect any circles in {movie_path}.")

    empirical_centers = calculate_theoretical_centers(
        image_width, image_height, args.chamber_columns, args.chamber_rows
    )
    matched_centers = match_circles_to_chambers(circles, empirical_centers, args.down_factor)

    expected_chambers = args.chamber_columns * args.chamber_rows
    if len(matched_centers) != expected_chambers:
        raise RuntimeError(
            f"Failed to match all {expected_chambers} chambers. Only matched {len(matched_centers)}."
        )

    logging.info(f"Detected and matched {len(matched_centers)} chambers")

    return matched_centers, avg_frame, circles, crop_dim
