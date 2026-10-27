"""Render inference-overlay movies and clips using explicit original-frame mappings.

OpenCV is required only when rendering runs.
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
    """Resolve output FPS and the source-frame write interval for the requested playback
    speed.

    Cap output FPS when valid and drop frames at interval = input_fps * speed_factor /
    output_fps. Return output FPS, interval, and whether capping occurred.
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
    source_frame_offset: int | None = None,
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
            source_frame_offset,
        )
        for csv_file in csv_files
    ]

    if not args:
        print(f"No inference CSV files found in: {inference_csv_dir}")
        return

    from .io import worker_count
    workers = worker_count(num_workers, len(args))
    if workers == 1:
        for payload in args:
            process_csv_and_movie(*payload)
    else:
        with Pool(processes=workers) as pool:
            pool.starmap(process_csv_and_movie, args)

    gc.collect()


def process_csv_and_movie(
    inf_csv_file_path, movie_dir, output_dir, skip_first_n_frames=5,
    timeline="none", timeline_compare=False, timeline_labels="predicted",
    timeline_height=18, white_mode=False, speed_factor=1.0, max_output_fps=40.0,
    show_timestamp=False, timestamp_position="top_right", timestamp_decimals=1,
    source_frame_offset=None, atomic_write=True,
):
    """Resolve a folder input pair and delegate to the checked original-frame renderer."""
    from .naming import extract_inf_csv_filename_part
    from .io import find_movie_file
    movie = find_movie_file(movie_dir, extract_inf_csv_filename_part(inf_csv_file_path))
    if movie is None:
        raise FileNotFoundError(f"No unique movie for {inf_csv_file_path}")
    stem = os.path.splitext(os.path.basename(inf_csv_file_path))[0]
    suffix = "_movie.mp4" if float(speed_factor) == 1.0 else f"_movie_{_format_speed_tag(speed_factor)}.mp4"
    render_overlay_movie(
        inf_csv_file_path, movie, os.path.join(output_dir, stem + suffix),
        source_frame_offset=source_frame_offset, skip_first_n_frames=skip_first_n_frames,
        timeline=timeline, timeline_compare=timeline_compare, timeline_labels=timeline_labels,
        timeline_height=timeline_height, white_mode=white_mode, speed_factor=speed_factor,
        max_output_fps=max_output_fps, show_timestamp=show_timestamp,
        timestamp_position=timestamp_position, timestamp_decimals=timestamp_decimals,
        atomic_write=atomic_write,
    )


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
    source_frame_offset: int | None = None,
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
    csv_summary=None,
    resolved_fps: float | None = None,
    default_movie_fps: float = 40.0,
) -> dict:
    """Render an overlay for an explicit inference CSV and source movie pair.

    Clip bounds use zero-based original-frame indices with an exclusive end. Validate
    the CSV frame map and seek the CSV and video independently; never infer alignment
    from row count alone. Timestamps can refer to the source movie, requested clip, or
    first rendered frame.
    """

    try:
        import cv2
    except Exception as e:  # pragma: no cover
        raise ImportError(
            "OpenCV (cv2) is required for inference overlay movie generation. "
            "Install opencv-python (or equivalent) in this environment."
        ) from e

    # sanitize inputs
    inf_csv_file_path = str(inf_csv_file_path)
    movie_file_path = str(movie_file_path)
    output_movie_path = str(output_movie_path)

    if not os.path.exists(inf_csv_file_path):
        raise FileNotFoundError(f"CSV not found: {inf_csv_file_path}")
    if not os.path.exists(movie_file_path):
        raise FileNotFoundError(f"Movie not found: {movie_file_path}")

    try:
        start_frame = int(start_frame)
    except Exception:
        start_frame = 0
    start_frame = max(0, start_frame)

    # Determine total video frames
    from .io import open_video_capture, resolve_movie_fps
    cap_probe = open_video_capture(movie_file_path)
    if not cap_probe.isOpened():
        cap_probe.release()
        raise ValueError(f"Could not open movie: {movie_file_path}")

    n_vid_total = int(cap_probe.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps_in = float(cap_probe.get(cv2.CAP_PROP_FPS) or 0.0)
    cap_probe.release()

    fps_in = resolve_movie_fps(fps_in if resolved_fps is None else resolved_fps, default_movie_fps)

    from .frame_map import read_csv_summary
    if csv_summary is None:
        csv_summary = read_csv_summary(inf_csv_file_path, source_frame_offset=source_frame_offset, video_frames=n_vid_total)
    source_frames = csv_summary.frames
    if source_frames[-1] >= n_vid_total:
        raise ValueError('Inference frame mapping exceeds the source movie')
    source_first = int(source_frames[0])
    n_total_aligned = int(source_frames[-1]) + 1
    if n_total_aligned <= 0:
        raise ValueError(f"Could not determine total frames for {movie_file_path}")

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
        raise ValueError(
            f"Empty clip range for {os.path.basename(movie_file_path)}: "
            f"start_frame={start_frame} >= end_frame_exclusive={end_frame_exclusive}"
        )

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
    render_start_frame = max(int(start_frame), int(skip_first_n_frames), source_first)
    render_end_frame = int(end_frame_exclusive)

    n_render = max(0, int(render_end_frame) - int(render_start_frame))
    if n_render <= 0:
        raise ValueError(
            f"Nothing to render after skip_first_n_frames={skip_first_n_frames} "
            f"for clip range start={start_frame}, end={end_frame_exclusive}"
        )
    # Scheduled writes before the forced final frame, plus that final frame.
    expected_frames = 1 if n_render == 1 else min(n_render, int(np.floor((n_render - 2 + 1e-9) / write_interval)) + 2)

    # Ensure output directory exists.
    out_dir = os.path.dirname(output_movie_path) or "."
    os.makedirs(out_dir, exist_ok=True)

    tmp_out_path = _temp_path_for_atomic_write(output_movie_path) if atomic_write else output_movie_path

    # Open CSV and parse columns
    from contextlib import ExitStack
    aligned_start = max(int(start_frame), source_first)
    with ExitStack() as resources, csv_summary.reader(inf_csv_file_path, aligned_start - source_first) as csv_reader:
        try:
            headers = next(csv_reader)
        except StopIteration:
            raise ValueError(f"Empty CSV: {inf_csv_file_path}")

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
            """Assign deterministic categorical overlay colors in OpenCV BGR order."""
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
            """Add the requested timeline bar when its label columns are available."""
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

        # Seek row and movie independently through the explicit frame map.
        # Open movie and seek to start_frame.
        cap = open_video_capture(movie_file_path)
        resources.callback(cap.release)
        if not cap.isOpened():
            raise ValueError(f"Could not open movie: {movie_file_path}")

        try:
            cap.set(cv2.CAP_PROP_POS_FRAMES, float(aligned_start))
        except Exception:
            pass

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = None

        timestamp_font_scale = 0.8
        timestamp_thickness = 1

        def interpolate_color(prob: float):
            """Interpolate between gray and green based on probability."""

            p = float(prob)
            if not np.isfinite(p):
                p = 0.0  # Neutral color for missing predictions.
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
        source_frame_idx = int(aligned_start)
        next_write_offset = 0.0
        written_frames = 0

        # Iterate rows in lockstep with video frames.
        for row in csv_reader:
            if source_frame_idx >= int(end_frame_exclusive):
                break

            ret, frame = cap.read()
            if not ret:
                raise ValueError(f"Source movie ended before mapped frame {source_frame_idx}")

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
                resources.callback(out.release)
                if hasattr(out, 'isOpened') and not out.isOpened():
                    raise RuntimeError(f'Could not open overlay encoder: {tmp_out_path}')

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
                prob = float(prob_str) if prob_str.strip() else float('nan')
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
                probability_text = f"{prob:.2f}" if np.isfinite(prob) else 'N/A'
                cv2.putText(processed_frame, probability_text, (prob_x, y_position), font, 0.6, color, 1, cv2.LINE_AA)

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
                prob = float(prob_str) if prob_str.strip() else float('nan')
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
                probability_text = f"{prob:.2f}" if np.isfinite(prob) else 'N/A'
                cv2.putText(processed_frame, probability_text, (prob_x, y_position), font, 0.6, color, 1, cv2.LINE_AA)

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
            next_write_offset = float(written_frames) * float(write_interval)
            source_frame_idx += 1

        cap.release()
        if out:
            out.release()

    if source_frame_idx != render_end_frame or written_frames != expected_frames:
        raise RuntimeError(f'Overlay did not complete its requested frame schedule: '
                           f'{written_frames}/{expected_frames} output frames, '
                           f'source end {source_frame_idx}/{render_end_frame}')

    # Atomic finalize.
    if atomic_write:
        try:
            if os.path.exists(tmp_out_path) and os.path.getsize(tmp_out_path) > 1024:
                capture = open_video_capture(tmp_out_path)
                try:
                    if not capture.isOpened() or int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) != written_frames:
                        raise RuntimeError('Encoded overlay frame count does not match the requested rendered frames')
                    for index in sorted({0, written_frames - 1}):
                        capture.set(cv2.CAP_PROP_POS_FRAMES, index)
                        if not capture.read()[0]:
                            raise RuntimeError(f'Encoded overlay cannot decode frame {index}')
                finally:
                    capture.release()
                from autobehaver.completion import preserve_previous
                preserve_previous(output_movie_path)
                os.replace(tmp_out_path, output_movie_path)
            else:
                raise RuntimeError(f'Overlay encoder did not produce a complete output: {tmp_out_path}')
        except Exception as e:
            raise RuntimeError(f'Could not publish overlay ({e}); temporary output retained at {tmp_out_path}') from e

    print(f"Finished processing. Output saved to: {output_movie_path}")
    return {'output_movie_path': output_movie_path, 'frames': written_frames,
            'fps': fps_out, 'fps_in': fps_in, 'start_frame': render_start_frame,
            'end_frame_exclusive': render_end_frame}
