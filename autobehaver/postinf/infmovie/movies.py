"""Inference overlay movie generation.

This code path is intentionally **optional** and depends on OpenCV (`cv2`).
It is split into its own subpackage (`infmovie`) so that mmgen (master matrix
generation) does not require OpenCV.

This module provides the overlay renderer used by both direct folder runs and
the registry-driven infmovie pipeline.
"""

from __future__ import annotations

import csv
import gc
import os
from multiprocessing import Pool, current_process

import numpy as np


def _format_speed_tag(speed_factor: float) -> str:
    """Format a speed factor for use in filenames.

    Examples:
      2.0  -> x2
      1.5  -> x1p5
      0.75 -> x0p75
    """

    try:
        sf = float(speed_factor)
    except Exception:
        sf = 1.0

    # If close to an integer, prefer an integer tag.
    if abs(sf - round(sf)) < 1e-9:
        return f"x{int(round(sf))}"

    s = f"{sf:.2f}".rstrip("0").rstrip(".")
    s = s.replace(".", "p")
    return f"x{s}"


def _compute_output_fps_and_interval(
    fps_in: float,
    speed_factor: float,
    max_output_fps: float,
) -> tuple[float, float, bool]:
    """Compute output FPS and a source-frame write interval.

    We want two things at once:
      1) The output file FPS should not exceed `max_output_fps` (presentation-friendly cap).
      2) The *playback speed* should match `speed_factor`.

    We achieve this by:
      - requesting fps_out = min(fps_in * speed_factor, max_output_fps)
      - if fps_out is capped, dropping frames so that one output frame represents
        `interval = (fps_in * speed_factor) / fps_out` source frames.

    Returns:
      fps_out: FPS to set on the output video writer
      interval: approximate number of source frames per output frame (>= 1)
      capped: True if fps_out was reduced due to max_output_fps
    """

    try:
        fps_in_f = float(fps_in)
    except Exception:
        fps_in_f = 0.0

    try:
        sf = float(speed_factor)
    except Exception:
        sf = 1.0

    try:
        cap = float(max_output_fps)
    except Exception:
        cap = 40.0

    if (not np.isfinite(fps_in_f)) or fps_in_f <= 0:
        fps_in_f = 30.0
    if (not np.isfinite(sf)) or sf <= 0:
        sf = 1.0

    # If cap is invalid, treat it as "no cap".
    if (not np.isfinite(cap)) or cap <= 0:
        cap = float("inf")

    # Hard clamp to avoid pathological FPS values that encoders/players dislike.
    cap = min(cap, 240.0)

    desired_fps = fps_in_f * sf
    fps_out = min(desired_fps, cap)

    if (not np.isfinite(fps_out)) or fps_out <= 0:
        fps_out = 30.0
    if fps_out < 1.0:
        fps_out = 1.0

    interval = desired_fps / fps_out if fps_out > 0 else 1.0
    if (not np.isfinite(interval)) or interval <= 0:
        interval = 1.0
    if interval < 1.0:
        interval = 1.0

    capped = fps_out < (desired_fps - 1e-9)
    return float(fps_out), float(interval), bool(capped)


def generate_movies(
    inference_csv_dir: str,
    movie_dir: str,
    output_dir: str,
    *,
    num_workers: int = 16,
    skip_first_n_frames: int = 5,
    timeline: str = "none",
    timeline_compare: bool = False,
    timeline_labels: str = "predicted",
    timeline_height: int = 18,
    white_mode: bool = False,
    speed_factor: float = 1.0,
    max_output_fps: float = 40.0,
    show_timestamp: bool = False,
    timestamp_position: str = "top_right",
    timestamp_decimals: int = 1,
) -> None:
    """Generate inference overlay movies for all inference CSVs in a folder."""

    # Lazy import so this tool can be installed without OpenCV unless used.
    try:
        import cv2  # noqa: F401
    except Exception as e:  # pragma: no cover
        raise ImportError(
            "OpenCV (cv2) is required for inference overlay movie generation. "
            "Install opencv-python (or equivalent) in this environment."
        ) from e

    os.makedirs(output_dir, exist_ok=True)

    csv_files = [
        f
        for f in os.listdir(inference_csv_dir)
        if f.endswith(".csv") and not f.endswith("_bouts.csv")
    ]

    args = [
        (
            os.path.join(inference_csv_dir, csv_file),
            movie_dir,
            output_dir,
            int(skip_first_n_frames),
            str(timeline),
            bool(timeline_compare),
            str(timeline_labels),
            int(timeline_height),
            bool(white_mode),
            float(speed_factor),
            float(max_output_fps),
            bool(show_timestamp),
            str(timestamp_position),
            int(timestamp_decimals),
        )
        for csv_file in csv_files
    ]

    if not args:
        print(f"No inference CSV files found in: {inference_csv_dir}")
        return

    with Pool(processes=int(num_workers)) as pool:
        pool.starmap(process_csv_and_movie, args)

    gc.collect()


def process_csv_and_movie(
    inf_csv_file_path: str,
    movie_dir: str,
    output_dir: str,
    skip_first_n_frames: int = 5,
    timeline: str = "none",
    timeline_compare: bool = False,
    timeline_labels: str = "predicted",
    timeline_height: int = 18,
    white_mode: bool = False,
    speed_factor: float = 1.0,
    max_output_fps: float = 40.0,
    show_timestamp: bool = False,
    timestamp_position: str = "top_right",
    timestamp_decimals: int = 1,
) -> None:
    """Render a single overlay movie for one inference CSV."""

    try:
        import cv2
    except Exception as e:  # pragma: no cover
        raise ImportError(
            "OpenCV (cv2) is required for inference overlay movie generation. "
            "Install opencv-python (or equivalent) in this environment."
        ) from e

    from .naming import extract_inf_csv_filename_part
    from .io import find_movie_file

    print(f"Process {current_process().name} is processing CSV: {inf_csv_file_path}")

    filename_part = extract_inf_csv_filename_part(inf_csv_file_path)

    movie_file_path = find_movie_file(movie_dir, filename_part)
    if not movie_file_path:
        print(f"No matching movie file found for: {os.path.basename(inf_csv_file_path)}")
        return

    print(f"Process {current_process().name} is processing Movie: {movie_file_path}")

    # For timeline mapping we want an estimate of total renderable frames.
    # We use (CSV rows, video frame count) and subtract skip_first_n_frames.
    try:
        with open(inf_csv_file_path, "r", newline="") as _f:
            # subtract header
            n_csv_rows = max(0, sum(1 for _ in _f) - 1)
    except Exception:
        n_csv_rows = 0

    cap_probe = cv2.VideoCapture(movie_file_path)
    n_vid = int(cap_probe.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap_probe.release()
    if n_vid > 0 and n_csv_rows > 0:
        n_total = min(n_csv_rows, n_vid)
    else:
        n_total = max(n_csv_rows, n_vid)

    n_render = max(0, int(n_total) - int(skip_first_n_frames))

    # Normalize timeline options.
    timeline = (timeline or "none").strip().lower()
    if timeline not in {"none", "pose", "orient", "both"}:
        timeline = "none"

    timeline_compare = bool(timeline_compare)
    timeline_mode = "compare" if timeline_compare else "single"

    timeline_labels = (timeline_labels or "predicted").strip().lower()
    if timeline_labels not in {"smoothed", "predicted"}:
        timeline_labels = "predicted"

    # Requested timelines by type.
    draw_pose_timeline_req = timeline in {"pose", "both"}
    draw_orient_timeline_req = timeline in {"orient", "both"}

    timeline_height = max(2, int(timeline_height))

    # Timeline label text scaling (kept proportional to bar height for aesthetics).
    # Baseline: 18px bar -> 0.5 font scale.
    #
    # We intentionally use a slightly gentler slope than a direct proportional mapping
    # so long labels (e.g., "Orientation timeline") don't clip at common panel widths
    # when timeline_height is large (e.g., 36px).
    #
    # Example mapping:
    #   timeline_height=18 -> 0.50
    #   timeline_height=36 -> 0.875
    timeline_label_scale = 0.125 + (float(timeline_height) / 48.0)
    timeline_label_scale = max(0.4, min(1.3, timeline_label_scale))
    timeline_label_thickness = 1 if timeline_label_scale <= 0.7 else 2

    # Playback speed.
    try:
        speed_factor = float(speed_factor)
    except Exception:
        speed_factor = 1.0
    if (not np.isfinite(speed_factor)) or speed_factor <= 0:
        print(
            f"WARNING: invalid speed_factor={speed_factor!r} for {os.path.basename(inf_csv_file_path)}; "
            "falling back to 1.0"
        )
        speed_factor = 1.0

    # Output FPS cap (presentation-friendly safeguard).
    try:
        max_output_fps = float(max_output_fps)
    except Exception:
        max_output_fps = 40.0
    if (not np.isfinite(max_output_fps)) or max_output_fps <= 0:
        print(
            f"WARNING: invalid max_output_fps={max_output_fps!r} for {os.path.basename(inf_csv_file_path)}; "
            "treating as no cap"
        )
        max_output_fps = float("inf")
    if max_output_fps > 240.0:
        print(
            f"WARNING: max_output_fps={max_output_fps:.2f} is very high; clamping to 240.0"
        )
        max_output_fps = 240.0

    show_timestamp = bool(show_timestamp)
    timestamp_position = (timestamp_position or "top_right").strip().lower()
    if timestamp_position not in {"top_left", "top_right", "bottom_left", "bottom_right"}:
        timestamp_position = "top_right"
    try:
        timestamp_decimals = int(timestamp_decimals)
    except Exception:
        timestamp_decimals = 1
    timestamp_decimals = max(0, min(6, timestamp_decimals))

    white_mode = bool(white_mode)
    # Colors are in OpenCV BGR order.
    panel_bg_color = (255, 255, 255) if white_mode else (0, 0, 0)
    panel_text_color = (0, 0, 0) if white_mode else (255, 255, 255)

    with open(inf_csv_file_path, newline="") as csvfile:
        csv_reader = csv.reader(csvfile)
        headers = next(csv_reader)

        pose_column_indices = []
        pose_labels = []
        orient_column_indices = []
        orient_labels = []
        predicted_pose_label_index = None
        predicted_orient_label_index = None
        smoothed_pose_label_index = None
        smoothed_orient_label_index = None

        for i, header in enumerate(headers):
            if header.startswith("Pose-") and header.endswith("-Prob"):
                label = header.replace("Pose-", "").replace("-Prob", "")
                pose_labels.append(label)
                pose_column_indices.append(i)
            elif header.startswith("Orient-") and header.endswith("-Prob"):
                label = header.replace("Orient-", "").replace("-Prob", "")
                orient_labels.append(label)
                orient_column_indices.append(i)
            elif header == "Predicted Pose Label":
                predicted_pose_label_index = i
            elif header == "Predicted Orient Label":
                predicted_orient_label_index = i
            elif header == "Smoothed Pose Label":
                smoothed_pose_label_index = i
            elif header == "Smoothed Orient Label":
                smoothed_orient_label_index = i

        # Build deterministic categorical color palettes for behavior (pose) and orientation.
        # We intentionally generate palettes *separately* so that orientation colors span
        # the full hue spectrum (instead of being a small slice after the pose palette).
        def _hue_order(n: int) -> list[int]:
            """Return an index order that maximizes adjacent hue contrast."""

            if n <= 0:
                return []
            half = (n + 1) // 2
            order: list[int] = []
            for k in range(half):
                order.append(k)
                j = k + half
                if j < n:
                    order.append(j)
            return order[:n]

        def _generate_palette(
            n: int,
            *,
            alternate_value: bool,
            sat_bright: float = 0.85,
            val_bright: float = 0.95,
            sat_dim: float = 0.80,
            val_dim: float = 0.70,
        ) -> list[tuple[int, int, int]]:
            import colorsys

            if n <= 0:
                return []
            order = _hue_order(n)
            out_colors: list[tuple[int, int, int]] = []
            for i, idx in enumerate(order):
                h = (idx / float(n)) % 1.0
                if alternate_value and (i % 2 == 1):
                    sat = sat_dim
                    val = val_dim
                else:
                    sat = sat_bright
                    val = val_bright
                r, g, b = colorsys.hsv_to_rgb(h, sat, val)
                out_colors.append((int(b * 255), int(g * 255), int(r * 255)))
            return out_colors

        pose_palette = _generate_palette(len(pose_labels), alternate_value=True)
        orient_palette = _generate_palette(len(orient_labels), alternate_value=False)

        pose_color_map = {lbl: pose_palette[i] for i, lbl in enumerate(pose_labels)}
        orient_color_map = {lbl: orient_palette[i] for i, lbl in enumerate(orient_labels)}

        # Decide which timeline bars will be drawn for this CSV.
        # `bars` is an ordered list; each bar has its own progressive state.
        bars: list[dict] = []

        def _maybe_add_bar(kind: str, source: str, label_index: int | None) -> None:
            if label_index is None:
                return
            if kind == "behavior":
                color_map = pose_color_map
                base = "Behavior"
            else:
                color_map = orient_color_map
                base = "Orientation"

            if timeline_mode == "compare":
                display_name = f"{base} {source}"
            else:
                # single mode: show source only when it is not the default/implicit case.
                if source == "smoothed":
                    display_name = f"{base} timeline (smoothed)"
                else:
                    display_name = f"{base} timeline"

            bars.append(
                {
                    "kind": kind,
                    "source": source,
                    "label_index": int(label_index),
                    "color_map": color_map,
                    "buffer": None,
                    "last_x": -1,
                    "display_name": display_name,
                }
            )

        # Determine which label columns drive the timeline bars.
        # Bottom text overlays use predicted labels.
        if timeline != "none":
            if timeline_mode == "single":
                if draw_pose_timeline_req:
                    pose_idx = predicted_pose_label_index
                    pose_src = "predicted"
                    if timeline_labels == "smoothed" and smoothed_pose_label_index is not None:
                        pose_idx = smoothed_pose_label_index
                        pose_src = "smoothed"
                    _maybe_add_bar("behavior", pose_src, pose_idx)

                if draw_orient_timeline_req:
                    orient_idx = predicted_orient_label_index
                    orient_src = "predicted"
                    if timeline_labels == "smoothed" and smoothed_orient_label_index is not None:
                        orient_idx = smoothed_orient_label_index
                        orient_src = "smoothed"
                    _maybe_add_bar("orientation", orient_src, orient_idx)
            else:
                # compare mode: always attempt to draw predicted + smoothed for each requested kind
                if draw_pose_timeline_req:
                    _maybe_add_bar("behavior", "predicted", predicted_pose_label_index)
                    if smoothed_pose_label_index is not None:
                        _maybe_add_bar("behavior", "smoothed", smoothed_pose_label_index)
                    else:
                        print(
                            f"WARNING: compare mode requested, but no Smoothed Pose Label column found in "
                            f"{os.path.basename(inf_csv_file_path)}. Drawing predicted only."
                        )

                if draw_orient_timeline_req:
                    _maybe_add_bar("orientation", "predicted", predicted_orient_label_index)
                    if smoothed_orient_label_index is not None:
                        _maybe_add_bar("orientation", "smoothed", smoothed_orient_label_index)
                    else:
                        print(
                            f"WARNING: compare mode requested, but no Smoothed Orient Label column found in "
                            f"{os.path.basename(inf_csv_file_path)}. Drawing predicted only."
                        )

        # Compute timeline geometry.
        bar_sep = 2  # pixels between consecutive bars
        group_sep = 6  # extra separation between Behavior and Orientation groups
        timeline_total_height = 0
        if bars:
            timeline_total_height += len(bars) * timeline_height
            # separators between bars
            for i in range(len(bars) - 1):
                if bars[i]["kind"] != bars[i + 1]["kind"]:
                    timeline_total_height += group_sep
                else:
                    timeline_total_height += bar_sep

        cap = cv2.VideoCapture(movie_file_path)
        fps_in = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        if (not np.isfinite(fps_in)) or fps_in <= 0:
            print(
                f"WARNING: could not read FPS for movie {os.path.basename(movie_file_path)}; "
                "defaulting to 30.0 FPS for timestamp/speed calculations."
            )
            fps_in = 30.0

        fps_out, write_interval, was_capped = _compute_output_fps_and_interval(
            fps_in=float(fps_in),
            speed_factor=float(speed_factor),
            max_output_fps=float(max_output_fps),
        )

        desired_fps = float(fps_in) * float(speed_factor)
        if was_capped or (write_interval > 1.0001):
            # write_interval > 1 implies we will drop frames to preserve speed.
            if np.isfinite(max_output_fps) and max_output_fps > 0 and max_output_fps < float("inf"):
                print(
                    f"INFO: requested {speed_factor:g}x at {fps_in:.2f} FPS -> desired FPS {desired_fps:.2f}. "
                    f"Capping output FPS at {fps_out:.2f} (max_output_fps={max_output_fps:.2f}) and "
                    f"dropping frames (~{write_interval:.3g} source frames/output frame) to preserve speed."
                )

        # Source frame index (0-based) and output-write schedule.
        source_frame_idx = 0
        next_write_offset = 0.0  # next source offset (after skip) at which to write
        written_frames = 0

        csv_base_name = os.path.splitext(os.path.basename(inf_csv_file_path))[0]
        # Add a speed tag to the output filename when speed != 1 to avoid accidental overwrites
        # and to make presentation exports self-describing.
        if abs(float(speed_factor) - 1.0) > 1e-9:
            speed_tag = _format_speed_tag(speed_factor)
            output_movie_filename = f"{csv_base_name}_movie_{speed_tag}.mp4"
        else:
            output_movie_filename = f"{csv_base_name}_movie.mp4"
        output_movie_path = os.path.join(output_dir, output_movie_filename)

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = None

        # Timestamp draw params (computed once after frame size is known).
        timestamp_font_scale = 0.8
        timestamp_thickness = 1

        # Timeline state (allocated once frame size is known).
        # Each bar has its own progressive buffer and last_x.

        def interpolate_color(prob: float):
            """Interpolate between medium gray and green based on probability."""

            # Colors are in OpenCV BGR order.
            p = float(prob)
            if p < 0.0:
                p = 0.0
            if p > 1.0:
                p = 1.0

            if white_mode:
                # On a white panel background, keep text darker for contrast:
                # dark gray (low confidence) -> darker green (high confidence).
                gray = 40
                green_max = 170
                blue = int(gray * (1 - p) + 0 * p)
                green = int(gray * (1 - p) + green_max * p)
                red = int(gray * (1 - p) + 0 * p)
                return (blue, green, red)

            # Default: medium gray to bright green on a black panel background.
            gray_value = 127
            red_intensity = int(gray_value * (1 - p))
            green_intensity = int(gray_value + (255 - gray_value) * p)
            blue_intensity = int(gray_value * (1 - p))
            return (blue_intensity, green_intensity, red_intensity)

        for row in csv_reader:
            ret, frame = cap.read()
            if not ret:
                print("No more frames to read from the movie.")
                break

            if all(not cell.strip() for cell in row):
                print("No more data in CSV. Stopping frame processing.")
                break

            # Keep the CSV row stream and the video frame stream in lock-step.
            # `source_frame_idx` is the 0-based index in the source movie (and in the CSV).
            if source_frame_idx < int(skip_first_n_frames):
                source_frame_idx += 1
                continue

            # Source offset relative to the first rendered frame (after skip).
            src_offset = int(source_frame_idx) - int(skip_first_n_frames)

            # Decide whether to write this frame, based on FPS capping.
            # When the requested output FPS exceeds max_output_fps, we cap FPS and drop frames
            # to preserve the requested speed_factor.
            force_last_frame = bool(n_render > 0 and src_offset >= (n_render - 1))
            should_write = (float(src_offset) + 1e-9) >= float(next_write_offset) or force_last_frame

            if out is None:
                height, width, _ = frame.shape
                # Right-side panel width. This was increased slightly after adding
                # the per-class color legend squares so that long labels don't
                # collide with the probability column.
                side_panel_width = 300
                new_width = width + side_panel_width
                output_height = height + int(timeline_total_height)
                out = cv2.VideoWriter(output_movie_path, fourcc, float(fps_out), (new_width, output_height))

                # Compute a timestamp font scale that looks reasonable across resolutions.
                # Baseline: ~720p -> 0.8. Clamp to keep it readable but not huge.
                timestamp_font_scale = max(0.5, min(1.2, (float(height) / 720.0) * 0.8))
                timestamp_thickness = 2 if timestamp_font_scale >= 0.9 else 1

                # Ensure timeline label text fits within the side panel width.
                # This avoids clipping long labels like "Orientation timeline" when
                # timeline_height is large, without needing to expand the output frame.
                if bars and int(timeline_total_height) > 0:
                    avail_w = max(50, int(side_panel_width) - 12)  # ~10px left margin, ~2px right margin
                    max_w = 0
                    for _b in bars:
                        _txt = str(_b.get("display_name", "timeline"))
                        (_w, _h), _bl = cv2.getTextSize(
                            _txt,
                            cv2.FONT_HERSHEY_COMPLEX,
                            float(timeline_label_scale),
                            int(timeline_label_thickness),
                        )
                        if int(_w) > int(max_w):
                            max_w = int(_w)
                    if int(max_w) > int(avail_w) and int(max_w) > 0:
                        shrink = float(avail_w) / float(max_w)
                        timeline_label_scale = max(0.35, float(timeline_label_scale) * float(shrink))
                        timeline_label_thickness = 1 if float(timeline_label_scale) <= 0.7 else 2

                for bar in bars:
                    bar["buffer"] = np.zeros((timeline_height, width, 3), dtype=np.uint8)
                    # Match timeline background to the chosen panel background.
                    bar["buffer"][:] = panel_bg_color
                    bar["last_x"] = -1

            # Update timeline bar buffers for *every* source frame after the skip.
            # Even if we end up dropping this frame for FPS-capping, we still want the
            # timeline to reflect the true sequence of labels/probabilities.
            if int(timeline_total_height) > 0 and width > 1 and n_render > 0 and bars:
                x_current = int(round(src_offset * (width - 1) / max(1, n_render - 1)))

                for bar in bars:
                    buf = bar.get("buffer")
                    lbl_idx = bar.get("label_index")
                    if buf is None or lbl_idx is None:
                        continue
                    try:
                        lbl = row[int(lbl_idx)].strip()
                    except Exception:
                        lbl = ""
                    col = bar["color_map"].get(lbl, (0, 0, 0))
                    last_x = int(bar.get("last_x", -1))
                    if x_current > last_x:
                        buf[:, last_x + 1 : x_current + 1] = col
                        bar["last_x"] = x_current

            # If we are dropping frames to meet max_output_fps, skip the expensive drawing work
            # for non-written frames.
            if not should_write:
                source_frame_idx += 1
                continue

            output_height = height + int(timeline_total_height)
            processed_frame = np.zeros((output_height, new_width, 3), dtype=np.uint8)
            # Fill the "extra" regions (right-side panel + timeline area) with the desired background.
            # We then overwrite the left video region with the actual frame.
            processed_frame[:] = panel_bg_color
            processed_frame[:height, :width] = frame

            font = cv2.FONT_HERSHEY_COMPLEX

            cv2.putText(
                processed_frame,
                "Behavior",
                (width + 10, 30),
                font,
                0.7,
                panel_text_color,
                1,
                cv2.LINE_AA,
            )
            cv2.line(processed_frame, (width + 10, 40), (width + 150, 40), panel_text_color, 1)

            # Probability column x-position. Keep a fixed right margin so we can
            # safely adjust the side panel width without re-tuning all text.
            prob_x = width + (side_panel_width - 80)

            for idx, label in enumerate(pose_labels):
                y_position = 60 + idx * 30
                prob_str = row[pose_column_indices[idx]]
                prob = float(prob_str) if prob_str.strip() else 0.0
                color = interpolate_color(prob)
                # Optional legend square for the timeline color (only shown when timeline is enabled).
                label_x = width + 10
                # Optional legend square for the timeline color.
                show_behavior_squares = any(b.get("kind") == "behavior" for b in bars)
                if int(timeline_total_height) > 0 and show_behavior_squares:
                    sq = 12
                    sq_x0 = width + 10
                    sq_y0 = y_position - sq + 4
                    sq_color = pose_color_map.get(label, (127, 127, 127))
                    cv2.rectangle(processed_frame, (sq_x0, sq_y0), (sq_x0 + sq, sq_y0 + sq), sq_color, thickness=-1)
                    label_x = width + 10 + sq + 8

                cv2.putText(processed_frame, label, (label_x, y_position), font, 0.6, color, 1, cv2.LINE_AA)
                cv2.putText(processed_frame, f"{prob:.2f}", (prob_x, y_position), font, 0.6, color, 1, cv2.LINE_AA)

            orient_start_y = 60 + len(pose_labels) * 30 + 20
            cv2.putText(
                processed_frame,
                "Orientation",
                (width + 10, orient_start_y),
                font,
                0.7,
                panel_text_color,
                1,
                cv2.LINE_AA,
            )
            cv2.line(processed_frame, (width + 10, orient_start_y + 10), (width + 150, orient_start_y + 10), panel_text_color, 1)

            for idx, label in enumerate(orient_labels):
                y_position = orient_start_y + 30 + idx * 30
                prob_str = row[orient_column_indices[idx]]
                prob = float(prob_str) if prob_str.strip() else 0.0
                color = interpolate_color(prob)
                label_x = width + 10
                show_orient_squares = any(b.get("kind") == "orientation" for b in bars)
                if int(timeline_total_height) > 0 and show_orient_squares:
                    sq = 12
                    sq_x0 = width + 10
                    sq_y0 = y_position - sq + 4
                    sq_color = orient_color_map.get(label, (127, 127, 127))
                    cv2.rectangle(processed_frame, (sq_x0, sq_y0), (sq_x0 + sq, sq_y0 + sq), sq_color, thickness=-1)
                    label_x = width + 10 + sq + 8

                cv2.putText(processed_frame, label, (label_x, y_position), font, 0.6, color, 1, cv2.LINE_AA)
                cv2.putText(processed_frame, f"{prob:.2f}", (prob_x, y_position), font, 0.6, color, 1, cv2.LINE_AA)

            if predicted_pose_label_index is not None:
                predicted_pose_label = row[predicted_pose_label_index]
                text_size_pose = cv2.getTextSize(predicted_pose_label, font, 1, 2)[0]
                text_x_pose = (width // 4) - (text_size_pose[0] // 2)
                text_y = height - 10
                cv2.putText(processed_frame, predicted_pose_label, (text_x_pose, text_y), font, 1, (0, 0, 0), 2, cv2.LINE_AA)

            if predicted_orient_label_index is not None:
                predicted_orient_label = row[predicted_orient_label_index]
                text_size_orient = cv2.getTextSize(predicted_orient_label, font, 1, 2)[0]
                text_x_orient = (3 * width // 4) - (text_size_orient[0] // 2)
                cv2.putText(processed_frame, predicted_orient_label, (text_x_orient, text_y), font, 1, (0, 0, 0), 2, cv2.LINE_AA)

            # Optional timestamp overlay (seconds).
            if show_timestamp and fps_in > 0 and width > 1 and height > 1:
                # Timestamp is relative to the first rendered frame (after skip_first_n_frames),
                # expressed in *source time*.
                t_sec = float(src_offset) / float(fps_in)
                if abs(float(speed_factor) - 1.0) > 1e-9:
                    ts_text = f"t={t_sec:.{timestamp_decimals}f}s ({speed_factor:g}x)"
                else:
                    ts_text = f"t={t_sec:.{timestamp_decimals}f}s"

                (tw, th), bl = cv2.getTextSize(
                    ts_text,
                    font,
                    float(timestamp_font_scale),
                    int(timestamp_thickness),
                )
                margin = 10
                if timestamp_position == "top_left":
                    x = margin
                    y = margin + th
                elif timestamp_position == "top_right":
                    x = max(margin, int(width - tw - margin))
                    y = margin + th
                elif timestamp_position == "bottom_left":
                    x = margin
                    y = max(margin + th, int(height - margin))
                else:  # bottom_right
                    x = max(margin, int(width - tw - margin))
                    y = max(margin + th, int(height - margin))

                # Draw with outline for readability on any background.
                cv2.putText(
                    processed_frame,
                    ts_text,
                    (x, y),
                    font,
                    float(timestamp_font_scale),
                    (0, 0, 0),
                    int(timestamp_thickness) + 2,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    processed_frame,
                    ts_text,
                    (x, y),
                    font,
                    float(timestamp_font_scale),
                    (255, 255, 255),
                    int(timestamp_thickness),
                    cv2.LINE_AA,
                )

            # Optional timeline bar(s) appended below the main frame.
            if int(timeline_total_height) > 0 and width > 1 and n_render > 0 and bars:
                # Composite bars into the output frame.
                y0 = height
                for i, bar in enumerate(bars):
                    buf = bar.get("buffer")
                    if buf is None:
                        continue
                    processed_frame[y0 : y0 + timeline_height, 0:width] = buf
                    label_txt = str(bar.get("display_name", "timeline"))
                    (lw, lh), lbase = cv2.getTextSize(
                        label_txt,
                        font,
                        float(timeline_label_scale),
                        int(timeline_label_thickness),
                    )
                    # Vertically center the label within the bar height.
                    label_y = y0 + int((timeline_height + lh - lbase) / 2)
                    cv2.putText(
                        processed_frame,
                        label_txt,
                        (width + 10, label_y),
                        font,
                        float(timeline_label_scale),
                        panel_text_color,
                        int(timeline_label_thickness),
                        cv2.LINE_AA,
                    )
                    y0 += timeline_height
                    if i < len(bars) - 1:
                        if bars[i]["kind"] != bars[i + 1]["kind"]:
                            y0 += group_sep
                        else:
                            y0 += bar_sep

            out.write(processed_frame)
            written_frames += 1
            next_write_offset = float(next_write_offset) + float(write_interval)
            source_frame_idx += 1

        cap.release()
        if out:
            out.release()

    print(f"Finished processing. Output saved to: {output_movie_path}")


def _temp_path_for_atomic_write(final_path: str) -> str:
    """Return a sibling temporary path for atomic video writes.

    We keep the extension (e.g. .mp4) so some encoders behave consistently.
    """

    root, ext = os.path.splitext(final_path)
    return root + ".tmp" + ext


def render_overlay_movie(
    inf_csv_file_path: str,
    movie_file_path: str,
    output_movie_path: str,
    *,
    start_frame: int = 0,
    end_frame_exclusive: int | None = None,
    skip_first_n_frames: int = 5,
    timeline: str = "none",
    timeline_compare: bool = False,
    timeline_labels: str = "predicted",
    timeline_height: int = 18,
    white_mode: bool = False,
    speed_factor: float = 1.0,
    max_output_fps: float = 40.0,
    show_timestamp: bool = False,
    timestamp_position: str = "top_right",
    timestamp_decimals: int = 1,
    timestamp_reference: str = "rendered",
    atomic_write: bool = True,
) -> None:
    """Render an inference overlay movie for a specific movie/CSV pair.

    This is a more flexible backend than :func:`process_csv_and_movie`:
      * explicit movie path (no filename matching)
      * optional frame range (start/end) for clip generation
      * optional atomic output write

    Args:
        inf_csv_file_path: Path to Autobehaver inference CSV.
        movie_file_path: Path to the source .mp4 movie.
        output_movie_path: Path to write the rendered overlay movie (.mp4).
        start_frame: 0-based source frame index to start reading from.
        end_frame_exclusive: 0-based exclusive end frame index. None => render to end.
        timestamp_reference: "movie" | "clip" | "rendered".
            - movie: seconds since start of the source movie.
            - clip: seconds since `start_frame`.
            - rendered: seconds since the first *rendered* frame (after skip_first_n_frames).

    Notes:
        The CSV and video are assumed to be frame-aligned by row order.
    """

    try:
        import cv2
    except Exception as e:  # pragma: no cover
        raise ImportError(
            "OpenCV (cv2) is required for inference overlay movie generation. "
            "Install opencv-python (or equivalent) in this environment."
        ) from e

    # --- sanitize inputs ---
    inf_csv_file_path = str(inf_csv_file_path)
    movie_file_path = str(movie_file_path)
    output_movie_path = str(output_movie_path)

    if not os.path.exists(inf_csv_file_path):
        print(f"[infmovie] CSV not found: {inf_csv_file_path}")
        return
    if not os.path.exists(movie_file_path):
        print(f"[infmovie] Movie not found: {movie_file_path}")
        return

    try:
        start_frame = int(start_frame)
    except Exception:
        start_frame = 0
    start_frame = max(0, start_frame)

    # Determine total CSV rows (excluding header)
    try:
        with open(inf_csv_file_path, "r", newline="") as _f:
            n_csv_rows_total = max(0, sum(1 for _ in _f) - 1)
    except Exception:
        n_csv_rows_total = 0

    # Determine total video frames
    cap_probe = cv2.VideoCapture(movie_file_path)
    if not cap_probe.isOpened():
        print(f"[infmovie] Could not open movie: {movie_file_path}")
        return

    n_vid_total = int(cap_probe.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps_in = float(cap_probe.get(cv2.CAP_PROP_FPS) or 0.0)
    cap_probe.release()

    if (not np.isfinite(fps_in)) or fps_in <= 0:
        print(
            f"WARNING: could not read FPS for movie {os.path.basename(movie_file_path)}; "
            "defaulting to 30.0 FPS for timestamp/speed calculations."
        )
        fps_in = 30.0

    # Total aligned frames (best effort)
    if n_vid_total > 0 and n_csv_rows_total > 0:
        n_total_aligned = min(n_vid_total, n_csv_rows_total)
    else:
        n_total_aligned = max(n_vid_total, n_csv_rows_total)

    if n_total_aligned <= 0:
        print(f"[infmovie] Could not determine total frames for {movie_file_path}")
        return

    # Clamp end_frame
    if end_frame_exclusive is None:
        end_frame_exclusive = int(n_total_aligned)
    else:
        try:
            end_frame_exclusive = int(end_frame_exclusive)
        except Exception:
            end_frame_exclusive = int(n_total_aligned)

    end_frame_exclusive = max(0, min(int(end_frame_exclusive), int(n_total_aligned)))

    if start_frame >= end_frame_exclusive:
        print(
            f"[infmovie] Empty clip range for {os.path.basename(movie_file_path)}: "
            f"start_frame={start_frame} >= end_frame_exclusive={end_frame_exclusive}"
        )
        return

    # Normalize timeline options.
    timeline = (timeline or "none").strip().lower()
    if timeline not in {"none", "pose", "orient", "both"}:
        timeline = "none"

    timeline_compare = bool(timeline_compare)
    timeline_mode = "compare" if timeline_compare else "single"

    timeline_labels = (timeline_labels or "predicted").strip().lower()
    if timeline_labels not in {"smoothed", "predicted"}:
        timeline_labels = "predicted"

    draw_pose_timeline_req = timeline in {"pose", "both"}
    draw_orient_timeline_req = timeline in {"orient", "both"}

    timeline_height = max(2, int(timeline_height))

    # Timeline label text scales with bar height.
    timeline_label_scale = 0.125 + (float(timeline_height) / 48.0)
    timeline_label_scale = max(0.4, min(1.3, timeline_label_scale))
    timeline_label_thickness = 1 if timeline_label_scale <= 0.7 else 2

    # Playback speed.
    try:
        speed_factor = float(speed_factor)
    except Exception:
        speed_factor = 1.0
    if (not np.isfinite(speed_factor)) or speed_factor <= 0:
        print(
            f"WARNING: invalid speed_factor={speed_factor!r} for {os.path.basename(inf_csv_file_path)}; "
            "falling back to 1.0"
        )
        speed_factor = 1.0

    # Output FPS cap.
    try:
        max_output_fps = float(max_output_fps)
    except Exception:
        max_output_fps = 40.0
    if (not np.isfinite(max_output_fps)) or max_output_fps <= 0:
        print(
            f"WARNING: invalid max_output_fps={max_output_fps!r} for {os.path.basename(inf_csv_file_path)}; "
            "treating as no cap"
        )
        max_output_fps = float("inf")
    if max_output_fps > 240.0:
        print(f"WARNING: max_output_fps={max_output_fps:.2f} is very high; clamping to 240.0")
        max_output_fps = 240.0

    show_timestamp = bool(show_timestamp)
    timestamp_position = (timestamp_position or "top_right").strip().lower()
    if timestamp_position not in {"top_left", "top_right", "bottom_left", "bottom_right"}:
        timestamp_position = "top_right"
    try:
        timestamp_decimals = int(timestamp_decimals)
    except Exception:
        timestamp_decimals = 1
    timestamp_decimals = max(0, min(6, timestamp_decimals))

    timestamp_reference = (timestamp_reference or "rendered").strip().lower()
    if timestamp_reference not in {"movie", "clip", "rendered"}:
        timestamp_reference = "rendered"

    white_mode = bool(white_mode)
    # Colors are in OpenCV BGR order.
    panel_bg_color = (255, 255, 255) if white_mode else (0, 0, 0)
    panel_text_color = (0, 0, 0) if white_mode else (255, 255, 255)

    # Compute output fps and frame-drop interval.
    fps_out, write_interval, was_capped = _compute_output_fps_and_interval(
        fps_in=float(fps_in),
        speed_factor=float(speed_factor),
        max_output_fps=float(max_output_fps),
    )

    desired_fps = float(fps_in) * float(speed_factor)
    if was_capped or (write_interval > 1.0001):
        if np.isfinite(max_output_fps) and max_output_fps > 0 and max_output_fps < float("inf"):
            print(
                f"INFO: requested {speed_factor:g}x at {fps_in:.2f} FPS -> desired FPS {desired_fps:.2f}. "
                f"Capping output FPS at {fps_out:.2f} (max_output_fps={max_output_fps:.2f}) and "
                f"dropping frames (~{write_interval:.3g} source frames/output frame) to preserve speed."
            )

    # Frame-range bookkeeping.
    try:
        skip_first_n_frames = int(skip_first_n_frames)
    except Exception:
        skip_first_n_frames = 0
    skip_first_n_frames = max(0, int(skip_first_n_frames))

    # Render starts after the configured frame skip.
    render_start_frame = max(int(start_frame), int(skip_first_n_frames))
    render_end_frame = int(end_frame_exclusive)

    n_render = max(0, int(render_end_frame) - int(render_start_frame))
    if n_render <= 0:
        print(
            f"[infmovie] Nothing to render after skip_first_n_frames={skip_first_n_frames} "
            f"for clip range start={start_frame}, end={end_frame_exclusive}"
        )
        return

    # Ensure output directory exists.
    out_dir = os.path.dirname(output_movie_path) or "."
    os.makedirs(out_dir, exist_ok=True)

    tmp_out_path = _temp_path_for_atomic_write(output_movie_path) if atomic_write else output_movie_path

    # --- Open CSV and parse columns ---
    with open(inf_csv_file_path, newline="") as csvfile:
        csv_reader = csv.reader(csvfile)
        try:
            headers = next(csv_reader)
        except StopIteration:
            print(f"[infmovie] Empty CSV: {inf_csv_file_path}")
            return

        pose_column_indices = []
        pose_labels = []
        orient_column_indices = []
        orient_labels = []
        predicted_pose_label_index = None
        predicted_orient_label_index = None
        smoothed_pose_label_index = None
        smoothed_orient_label_index = None

        for i, header in enumerate(headers):
            if header.startswith("Pose-") and header.endswith("-Prob"):
                label = header.replace("Pose-", "").replace("-Prob", "")
                pose_labels.append(label)
                pose_column_indices.append(i)
            elif header.startswith("Orient-") and header.endswith("-Prob"):
                label = header.replace("Orient-", "").replace("-Prob", "")
                orient_labels.append(label)
                orient_column_indices.append(i)
            elif header == "Predicted Pose Label":
                predicted_pose_label_index = i
            elif header == "Predicted Orient Label":
                predicted_orient_label_index = i
            elif header == "Smoothed Pose Label":
                smoothed_pose_label_index = i
            elif header == "Smoothed Orient Label":
                smoothed_orient_label_index = i

        # Build deterministic categorical color palettes for behavior (pose) and orientation.
        def _hue_order(n: int) -> list[int]:
            if n <= 0:
                return []
            half = (n + 1) // 2
            order: list[int] = []
            for k in range(half):
                order.append(k)
                j = k + half
                if j < n:
                    order.append(j)
            return order[:n]

        def _generate_palette(
            n: int,
            *,
            alternate_value: bool,
            sat_bright: float = 0.85,
            val_bright: float = 0.95,
            sat_dim: float = 0.80,
            val_dim: float = 0.70,
        ) -> list[tuple[int, int, int]]:
            import colorsys

            if n <= 0:
                return []
            order = _hue_order(n)
            out_colors: list[tuple[int, int, int]] = []
            for i, idx in enumerate(order):
                h = (idx / float(n)) % 1.0
                if alternate_value and (i % 2 == 1):
                    sat = sat_dim
                    val = val_dim
                else:
                    sat = sat_bright
                    val = val_bright
                r, g, b = colorsys.hsv_to_rgb(h, sat, val)
                out_colors.append((int(b * 255), int(g * 255), int(r * 255)))
            return out_colors

        pose_palette = _generate_palette(len(pose_labels), alternate_value=True)
        orient_palette = _generate_palette(len(orient_labels), alternate_value=False)

        pose_color_map = {lbl: pose_palette[i] for i, lbl in enumerate(pose_labels)}
        orient_color_map = {lbl: orient_palette[i] for i, lbl in enumerate(orient_labels)}

        bars: list[dict] = []

        def _maybe_add_bar(kind: str, source: str, label_index: int | None) -> None:
            if label_index is None:
                return
            if kind == "behavior":
                color_map = pose_color_map
                base = "Behavior"
            else:
                color_map = orient_color_map
                base = "Orientation"

            if timeline_mode == "compare":
                display_name = f"{base} {source}"
            else:
                if source == "smoothed":
                    display_name = f"{base} timeline (smoothed)"
                else:
                    display_name = f"{base} timeline"

            bars.append(
                {
                    "kind": kind,
                    "source": source,
                    "label_index": int(label_index),
                    "color_map": color_map,
                    "buffer": None,
                    "last_x": -1,
                    "display_name": display_name,
                }
            )

        if timeline != "none":
            if timeline_mode == "single":
                if draw_pose_timeline_req:
                    pose_idx = predicted_pose_label_index
                    pose_src = "predicted"
                    if timeline_labels == "smoothed" and smoothed_pose_label_index is not None:
                        pose_idx = smoothed_pose_label_index
                        pose_src = "smoothed"
                    _maybe_add_bar("behavior", pose_src, pose_idx)

                if draw_orient_timeline_req:
                    orient_idx = predicted_orient_label_index
                    orient_src = "predicted"
                    if timeline_labels == "smoothed" and smoothed_orient_label_index is not None:
                        orient_idx = smoothed_orient_label_index
                        orient_src = "smoothed"
                    _maybe_add_bar("orientation", orient_src, orient_idx)
            else:
                if draw_pose_timeline_req:
                    _maybe_add_bar("behavior", "predicted", predicted_pose_label_index)
                    if smoothed_pose_label_index is not None:
                        _maybe_add_bar("behavior", "smoothed", smoothed_pose_label_index)
                    else:
                        print(
                            f"WARNING: compare mode requested, but no Smoothed Pose Label column found in "
                            f"{os.path.basename(inf_csv_file_path)}. Drawing predicted only."
                        )

                if draw_orient_timeline_req:
                    _maybe_add_bar("orientation", "predicted", predicted_orient_label_index)
                    if smoothed_orient_label_index is not None:
                        _maybe_add_bar("orientation", "smoothed", smoothed_orient_label_index)
                    else:
                        print(
                            f"WARNING: compare mode requested, but no Smoothed Orient Label column found in "
                            f"{os.path.basename(inf_csv_file_path)}. Drawing predicted only."
                        )

        # Timeline geometry.
        bar_sep = 2
        group_sep = 6
        timeline_total_height = 0
        if bars:
            timeline_total_height += len(bars) * timeline_height
            for i in range(len(bars) - 1):
                if bars[i]["kind"] != bars[i + 1]["kind"]:
                    timeline_total_height += group_sep
                else:
                    timeline_total_height += bar_sep

        # Skip CSV rows up to start_frame.
        # The first data row corresponds to source frame 0.
        skipped_rows = 0
        if start_frame > 0:
            for _ in range(start_frame):
                try:
                    next(csv_reader)
                    skipped_rows += 1
                except StopIteration:
                    break

        if skipped_rows < start_frame:
            print(
                f"[infmovie] CSV shorter than requested start_frame={start_frame}: "
                f"only skipped {skipped_rows} rows."
            )

        # Open movie and seek to start_frame.
        cap = cv2.VideoCapture(movie_file_path)
        if not cap.isOpened():
            print(f"[infmovie] Could not open movie: {movie_file_path}")
            return

        try:
            cap.set(cv2.CAP_PROP_POS_FRAMES, float(start_frame))
        except Exception:
            pass

        fps_in_cap = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        if np.isfinite(fps_in_cap) and fps_in_cap > 0:
            fps_in = fps_in_cap

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = None

        timestamp_font_scale = 0.8
        timestamp_thickness = 1

        def interpolate_color(prob: float):
            """Interpolate between gray and green based on probability."""

            p = float(prob)
            if p < 0.0:
                p = 0.0
            if p > 1.0:
                p = 1.0

            if white_mode:
                gray = 40
                green_max = 170
                blue = int(gray * (1 - p) + 0 * p)
                green = int(gray * (1 - p) + green_max * p)
                red = int(gray * (1 - p) + 0 * p)
                return (blue, green, red)

            gray_value = 127
            red_intensity = int(gray_value * (1 - p))
            green_intensity = int(gray_value + (255 - gray_value) * p)
            blue_intensity = int(gray_value * (1 - p))
            return (blue_intensity, green_intensity, red_intensity)

        # Source frame index is absolute (0-based) in the source movie.
        source_frame_idx = int(start_frame)
        next_write_offset = 0.0
        written_frames = 0

        # Iterate rows in lockstep with video frames.
        for row in csv_reader:
            if source_frame_idx >= int(end_frame_exclusive):
                break

            ret, frame = cap.read()
            if not ret:
                break

            if all(not cell.strip() for cell in row):
                break

            if source_frame_idx < int(skip_first_n_frames):
                source_frame_idx += 1
                continue

            if source_frame_idx < int(render_start_frame):
                # This can happen when start_frame < skip_first_n_frames.
                source_frame_idx += 1
                continue

            render_offset = int(source_frame_idx) - int(render_start_frame)

            force_last_frame = bool(n_render > 0 and render_offset >= (n_render - 1))
            should_write = (float(render_offset) + 1e-9) >= float(next_write_offset) or force_last_frame

            if out is None:
                height, width, _ = frame.shape
                side_panel_width = 300
                new_width = width + side_panel_width
                output_height = height + int(timeline_total_height)

                out = cv2.VideoWriter(tmp_out_path, fourcc, float(fps_out), (new_width, output_height))

                timestamp_font_scale = max(0.5, min(1.2, (float(height) / 720.0) * 0.8))
                timestamp_thickness = 2 if timestamp_font_scale >= 0.9 else 1

                # Ensure timeline label text fits.
                if bars and int(timeline_total_height) > 0:
                    avail_w = max(50, int(side_panel_width) - 12)
                    max_w = 0
                    for _b in bars:
                        _txt = str(_b.get("display_name", "timeline"))
                        (_w, _h), _bl = cv2.getTextSize(
                            _txt,
                            cv2.FONT_HERSHEY_COMPLEX,
                            float(timeline_label_scale),
                            int(timeline_label_thickness),
                        )
                        if int(_w) > int(max_w):
                            max_w = int(_w)
                    if int(max_w) > int(avail_w) and int(max_w) > 0:
                        shrink = float(avail_w) / float(max_w)
                        timeline_label_scale = max(0.35, float(timeline_label_scale) * float(shrink))
                        timeline_label_thickness = 1 if float(timeline_label_scale) <= 0.7 else 2

                for bar in bars:
                    bar["buffer"] = np.zeros((timeline_height, width, 3), dtype=np.uint8)
                    bar["buffer"][:] = panel_bg_color
                    bar["last_x"] = -1

            # Update timeline buffers for every renderable source frame.
            if int(timeline_total_height) > 0 and width > 1 and n_render > 0 and bars:
                x_current = int(round(render_offset * (width - 1) / max(1, n_render - 1)))

                for bar in bars:
                    buf = bar.get("buffer")
                    lbl_idx = bar.get("label_index")
                    if buf is None or lbl_idx is None:
                        continue
                    try:
                        lbl = row[int(lbl_idx)].strip()
                    except Exception:
                        lbl = ""
                    col = bar["color_map"].get(lbl, (0, 0, 0))
                    last_x = int(bar.get("last_x", -1))
                    if x_current > last_x:
                        buf[:, last_x + 1 : x_current + 1] = col
                        bar["last_x"] = x_current

            if not should_write:
                source_frame_idx += 1
                continue

            output_height = height + int(timeline_total_height)
            processed_frame = np.zeros((output_height, new_width, 3), dtype=np.uint8)
            processed_frame[:] = panel_bg_color
            processed_frame[:height, :width] = frame

            font = cv2.FONT_HERSHEY_COMPLEX

            cv2.putText(
                processed_frame,
                "Behavior",
                (width + 10, 30),
                font,
                0.7,
                panel_text_color,
                1,
                cv2.LINE_AA,
            )
            cv2.line(processed_frame, (width + 10, 40), (width + 150, 40), panel_text_color, 1)

            prob_x = width + (side_panel_width - 80)

            for idx, label in enumerate(pose_labels):
                y_position = 60 + idx * 30
                prob_str = row[pose_column_indices[idx]]
                prob = float(prob_str) if prob_str.strip() else 0.0
                color = interpolate_color(prob)

                label_x = width + 10
                show_behavior_squares = any(b.get("kind") == "behavior" for b in bars)
                if int(timeline_total_height) > 0 and show_behavior_squares:
                    sq = 12
                    sq_x0 = width + 10
                    sq_y0 = y_position - sq + 4
                    sq_color = pose_color_map.get(label, (127, 127, 127))
                    cv2.rectangle(processed_frame, (sq_x0, sq_y0), (sq_x0 + sq, sq_y0 + sq), sq_color, thickness=-1)
                    label_x = width + 10 + sq + 8

                cv2.putText(processed_frame, label, (label_x, y_position), font, 0.6, color, 1, cv2.LINE_AA)
                cv2.putText(processed_frame, f"{prob:.2f}", (prob_x, y_position), font, 0.6, color, 1, cv2.LINE_AA)

            orient_start_y = 60 + len(pose_labels) * 30 + 20
            cv2.putText(
                processed_frame,
                "Orientation",
                (width + 10, orient_start_y),
                font,
                0.7,
                panel_text_color,
                1,
                cv2.LINE_AA,
            )
            cv2.line(
                processed_frame,
                (width + 10, orient_start_y + 10),
                (width + 150, orient_start_y + 10),
                panel_text_color,
                1,
            )

            for idx, label in enumerate(orient_labels):
                y_position = orient_start_y + 30 + idx * 30
                prob_str = row[orient_column_indices[idx]]
                prob = float(prob_str) if prob_str.strip() else 0.0
                color = interpolate_color(prob)

                label_x = width + 10
                show_orient_squares = any(b.get("kind") == "orientation" for b in bars)
                if int(timeline_total_height) > 0 and show_orient_squares:
                    sq = 12
                    sq_x0 = width + 10
                    sq_y0 = y_position - sq + 4
                    sq_color = orient_color_map.get(label, (127, 127, 127))
                    cv2.rectangle(processed_frame, (sq_x0, sq_y0), (sq_x0 + sq, sq_y0 + sq), sq_color, thickness=-1)
                    label_x = width + 10 + sq + 8

                cv2.putText(processed_frame, label, (label_x, y_position), font, 0.6, color, 1, cv2.LINE_AA)
                cv2.putText(processed_frame, f"{prob:.2f}", (prob_x, y_position), font, 0.6, color, 1, cv2.LINE_AA)

            # Bottom labels are always drawn from predicted columns.
            if predicted_pose_label_index is not None:
                predicted_pose_label = row[predicted_pose_label_index]
                text_size_pose = cv2.getTextSize(predicted_pose_label, font, 1, 2)[0]
                text_x_pose = (width // 4) - (text_size_pose[0] // 2)
                text_y = height - 10
                cv2.putText(processed_frame, predicted_pose_label, (text_x_pose, text_y), font, 1, (0, 0, 0), 2, cv2.LINE_AA)

            if predicted_orient_label_index is not None:
                predicted_orient_label = row[predicted_orient_label_index]
                text_size_orient = cv2.getTextSize(predicted_orient_label, font, 1, 2)[0]
                text_x_orient = (3 * width // 4) - (text_size_orient[0] // 2)
                text_y = height - 10
                cv2.putText(processed_frame, predicted_orient_label, (text_x_orient, text_y), font, 1, (0, 0, 0), 2, cv2.LINE_AA)

            # Timestamp overlay.
            if show_timestamp and fps_in > 0 and width > 1 and height > 1:
                if timestamp_reference == "movie":
                    t_sec = float(source_frame_idx) / float(fps_in)
                elif timestamp_reference == "clip":
                    t_sec = float(source_frame_idx - int(start_frame)) / float(fps_in)
                else:
                    t_sec = float(render_offset) / float(fps_in)

                if abs(float(speed_factor) - 1.0) > 1e-9:
                    ts_text = f"t={t_sec:.{timestamp_decimals}f}s ({speed_factor:g}x)"
                else:
                    ts_text = f"t={t_sec:.{timestamp_decimals}f}s"

                (tw, th), bl = cv2.getTextSize(
                    ts_text,
                    font,
                    float(timestamp_font_scale),
                    int(timestamp_thickness),
                )
                margin = 10
                if timestamp_position == "top_left":
                    x = margin
                    y = margin + th
                elif timestamp_position == "top_right":
                    x = max(margin, int(width - tw - margin))
                    y = margin + th
                elif timestamp_position == "bottom_left":
                    x = margin
                    y = max(margin + th, int(height - margin))
                else:
                    x = max(margin, int(width - tw - margin))
                    y = max(margin + th, int(height - margin))

                cv2.putText(
                    processed_frame,
                    ts_text,
                    (x, y),
                    font,
                    float(timestamp_font_scale),
                    (0, 0, 0),
                    int(timestamp_thickness) + 2,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    processed_frame,
                    ts_text,
                    (x, y),
                    font,
                    float(timestamp_font_scale),
                    (255, 255, 255),
                    int(timestamp_thickness),
                    cv2.LINE_AA,
                )

            # Timeline bars.
            if int(timeline_total_height) > 0 and width > 1 and n_render > 0 and bars:
                y0 = height
                for i, bar in enumerate(bars):
                    buf = bar.get("buffer")
                    if buf is None:
                        continue
                    processed_frame[y0 : y0 + timeline_height, 0:width] = buf
                    label_txt = str(bar.get("display_name", "timeline"))
                    (lw, lh), lbase = cv2.getTextSize(
                        label_txt,
                        font,
                        float(timeline_label_scale),
                        int(timeline_label_thickness),
                    )
                    label_y = y0 + int((timeline_height + lh - lbase) / 2)
                    cv2.putText(
                        processed_frame,
                        label_txt,
                        (width + 10, label_y),
                        font,
                        float(timeline_label_scale),
                        panel_text_color,
                        int(timeline_label_thickness),
                        cv2.LINE_AA,
                    )
                    y0 += timeline_height
                    if i < len(bars) - 1:
                        if bars[i]["kind"] != bars[i + 1]["kind"]:
                            y0 += group_sep
                        else:
                            y0 += bar_sep

            out.write(processed_frame)
            written_frames += 1
            next_write_offset = float(next_write_offset) + float(write_interval)
            source_frame_idx += 1

        cap.release()
        if out:
            out.release()

    # Atomic finalize.
    if atomic_write:
        try:
            if os.path.exists(tmp_out_path) and os.path.getsize(tmp_out_path) > 1024:
                os.replace(tmp_out_path, output_movie_path)
            else:
                print(f"[infmovie] WARNING: temp output looks invalid; keeping tmp: {tmp_out_path}")
        except Exception as e:
            print(f"[infmovie] WARNING: could not finalize atomic write ({e}); temp at: {tmp_out_path}")

    print(f"Finished processing. Output saved to: {output_movie_path}")
