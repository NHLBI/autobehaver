"""Provide the OpenCV review interface for MovieChecker chamber exclusions."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import cv2
import numpy as np

from .exclusions import ExclusionsStore
from autobehaver.portable_paths import original_name

if TYPE_CHECKING:
    from ..autocropper.recording_db import RecordingDBIndex


VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".mpg", ".mpeg"}


def _is_video_file(path: str) -> bool:
    ext = os.path.splitext(path)[1].lower()
    return ext in VIDEO_EXTS


from autobehaver.recording_identity import derive_pair_key


def group_movie_pairs(movie_paths: List[str]) -> Dict[str, List[str]]:
    """Group movie paths by pair key and sort each group deterministically."""
    pairs: Dict[str, List[str]] = {}
    for p in movie_paths:
        key = derive_pair_key(p)
        pairs.setdefault(key, []).append(p)
    # Sort each group for deterministic selection
    for k in pairs:
        pairs[k] = sorted(pairs[k])
    return pairs


def choose_representative_movie(paths: List[str]) -> str:
    """Choose movie0 from a recording pair when present, otherwise the first path."""
    if not paths:
        raise ValueError("paths must be non-empty")
    # Prefer movie0 if present.
    for p in paths:
        if re.search(r"movie0", os.path.basename(p), flags=re.IGNORECASE):
            return p
    return paths[0]


@dataclass
class UILayout:
    """Store display rectangles and scaling in canvas pixel coordinates."""
    tile_rects: Dict[int, Tuple[int, int, int, int]]  # chamber -> (x1,y1,x2,y2) on the canvas
    seekbar_rect: Tuple[int, int, int, int]
    image_rect: Tuple[int, int, int, int]  # (x1,y1,x2,y2) where the resized frame is drawn
    header_rect: Tuple[int, int, int, int]
    footer_rect: Tuple[int, int, int, int]
    top_text_y: int
    scale: float
    display_size: Tuple[int, int]  # full canvas size (image + header/footer)


def compute_ui_layout(frame_w: int, frame_h: int, *, max_display_w: int = 1600, max_display_h: int = 1000) -> UILayout:
    """Calculate image scaling and review-control rectangles in canvas pixels.

    Reserve header and seekbar areas outside the movie image.
    """

    scale = min(1.0, max_display_w / float(frame_w), max_display_h / float(frame_h))
    disp_w = int(round(frame_w * scale))
    disp_h = int(round(frame_h * scale))

    # Reserve header/footer outside the image. Recompute once if needed so the
    # combined canvas still fits on typical screens.
    header_h = max(56, int(disp_h * 0.08))
    seekbar_h = max(18, int(disp_h * 0.03))
    footer_h = seekbar_h
    if disp_h + header_h + footer_h > max_display_h:
        scale = min(scale, max(0.1, (max_display_h - header_h - footer_h) / float(frame_h)))
        disp_w = int(round(frame_w * scale))
        disp_h = int(round(frame_h * scale))
        header_h = max(56, int(disp_h * 0.08))
        seekbar_h = max(18, int(disp_h * 0.03))
        footer_h = seekbar_h

    canvas_w = disp_w
    canvas_h = disp_h + header_h + footer_h

    # UI sizing inside the image area
    outer_margin = max(20, int(disp_w * 0.03))
    pad = max(10, int(disp_w * 0.01))

    avail_w = disp_w - 2 * outer_margin - 2 * pad
    avail_h = disp_h - 2 * outer_margin - pad
    tile_w = avail_w / 3.0
    tile_h = avail_h / 2.0
    tile = int(max(80, min(tile_w, tile_h)))

    grid_w = 3 * tile + 2 * pad
    grid_h = 2 * tile + pad

    x_start = int((disp_w - grid_w) / 2)
    y_start = int(header_h + (disp_h - grid_h) / 2)

    tile_rects: Dict[int, Tuple[int, int, int, int]] = {}
    chamber = 1
    for r in range(2):
        for c in range(3):
            x1 = x_start + c * (tile + pad)
            y1 = y_start + r * (tile + pad)
            x2 = x1 + tile
            y2 = y1 + tile
            tile_rects[chamber] = (x1, y1, x2, y2)
            chamber += 1

    seekbar_rect = (0, header_h + disp_h, canvas_w, canvas_h)
    image_rect = (0, header_h, disp_w, header_h + disp_h)
    header_rect = (0, 0, canvas_w, header_h)
    footer_rect = (0, header_h + disp_h, canvas_w, canvas_h)
    return UILayout(
        tile_rects=tile_rects,
        seekbar_rect=seekbar_rect,
        image_rect=image_rect,
        header_rect=header_rect,
        footer_rect=footer_rect,
        top_text_y=max(28, int(header_h * 0.5)),
        scale=scale,
        display_size=(canvas_w, canvas_h),
    )


def _draw_text_with_outline(img: np.ndarray, text: str, org: Tuple[int, int], font_scale: float, thickness: int) -> None:
    """Draw white text with a black outline on the image in place."""
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)


def _draw_ui_overlays(
    frame: np.ndarray,
    layout: UILayout,
    excluded_chambers: List[int],
    progress_idx: int,
    progress_total: int,
    file_label: str,
    *,
    paused: bool,
    playback_stride: int,
    frame_pos: int,
    frame_count: int,
    recording_excluded_chambers: Optional[List[int]] = None,
) -> np.ndarray:
    """Draw tiles, header text, and seekbar while keeping the full image visible."""

    canvas_w, canvas_h = layout.display_size
    ix1, iy1, ix2, iy2 = layout.image_rect
    img_w, img_h = ix2 - ix1, iy2 - iy1

    disp = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    # Header/footer background
    hx1, hy1, hx2, hy2 = layout.header_rect
    fx1, fy1, fx2, fy2 = layout.footer_rect
    disp[hy1:hy2, hx1:hx2] = (24, 24, 24)
    disp[fy1:fy2, fx1:fx2] = (20, 20, 20)

    if frame.shape[1] != img_w or frame.shape[0] != img_h:
        frame_img = cv2.resize(frame, (img_w, img_h), interpolation=cv2.INTER_AREA)
    else:
        frame_img = frame
    disp[iy1:iy2, ix1:ix2] = frame_img

    tile = next(iter(layout.tile_rects.values()))
    tile_size = tile[2] - tile[0]
    outline_thick = max(2, int(tile_size / 260))
    x_thick = max(outline_thick * 2, 4)

    for ch, (x1, y1, x2, y2) in layout.tile_rects.items():
        cv2.rectangle(disp, (x1, y1), (x2, y2), (255, 255, 255), outline_thick)
        cv2.putText(disp, str(ch), (x1 + 10, y1 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
        if recording_excluded_chambers and recording_excluded_chambers[ch - 1]:
            _draw_text_with_outline(disp, "Recording exclusion", (x1 + 10, y1 + 55), 0.5, 1)
        if excluded_chambers[ch - 1]:
            cv2.line(disp, (x1 + 15, y1 + 15), (x2 - 15, y2 - 15), (0, 0, 255), x_thick)
            cv2.line(disp, (x2 - 15, y1 + 15), (x1 + 15, y2 - 15), (0, 0, 255), x_thick)

    counter = f"{progress_idx + 1} / {progress_total}"
    _draw_text_with_outline(disp, counter, (20, layout.top_text_y), 1.0, 2)

    short = file_label
    if len(short) > 80:
        short = short[:77] + "..."
    _draw_text_with_outline(disp, short, (20, layout.top_text_y + 30), 0.7, 2)

    status = "PAUSED" if paused else f"PLAY x{max(1, int(playback_stride))}"
    _draw_text_with_outline(disp, status, (canvas_w - 190, layout.top_text_y), 0.8, 2)
    _draw_text_with_outline(disp, "Keys: [ / ] speed", (canvas_w - 250, layout.top_text_y + 30), 0.55, 1)

    x1, y1, x2, y2 = layout.seekbar_rect
    cv2.rectangle(disp, (x1, y1), (x2, y2), (20, 20, 20), -1)
    if frame_count > 1:
        frac = max(0.0, min(1.0, frame_pos / float(frame_count - 1)))
        fill_w = int(frac * (x2 - x1))
        cv2.rectangle(disp, (x1, y1), (x1 + fill_w, y2), (180, 180, 180), -1)

    return disp


class MovieCheckerApp:
    """Manage movie-pair selection, playback, and persistent chamber exclusions."""
    def __init__(
        self,
        input_dir: str,
        exclusions_csv_path: str,
        *,
        review_all: bool = False,
        max_display_w: int = 1600,
        max_display_h: int = 1000,
        playback_stride: int = 1,
        recording_index: Optional[RecordingDBIndex] = None,
    ):
        self.input_dir = input_dir
        self.exclusions_csv_path = exclusions_csv_path
        self.review_all = review_all
        self.max_display_w = max_display_w
        self.max_display_h = max_display_h
        self._playback_stride = max(1, int(playback_stride))
        self.recording_index = recording_index
        self._recording_exclusions: Dict[str, List[int]] = {}
        self.recording_skipped_keys: List[str] = []
        self._recording_flags: List[int] = [0] * 6

        self.store = ExclusionsStore(csv_path=exclusions_csv_path)
        self.store.load()

        self.movie_pairs: Dict[str, List[str]] = {}
        self.review_keys: List[str] = []
        self.review_movies: List[str] = []
        self.current_idx: int = 0

        # UI state
        self.layout: Optional[UILayout] = None
        self.excluded_flags: List[int] = [0, 0, 0, 0, 0, 0]
        self._current_pair_key: Optional[str] = None
        self._current_movie_path: Optional[str] = None
        self._cap: Optional[cv2.VideoCapture] = None
        self._paused: bool = False
        self._pending_seek_frame: Optional[int] = None

    def scan_movies(self) -> None:
        """Discover movie pairs and select those needing review after metadata exclusions."""
        if not os.path.isdir(self.input_dir):
            raise FileNotFoundError(f"input_dir does not exist: {self.input_dir}")

        all_files = [os.path.join(self.input_dir, f) for f in os.listdir(self.input_dir)]
        movie_files = sorted([p for p in all_files if os.path.isfile(p) and _is_video_file(p)])

        self.movie_pairs = group_movie_pairs(movie_files)

        # Resolve every movie before allowing any writes or automatic skips.
        self._recording_exclusions = {}
        self.recording_skipped_keys = []
        if self.recording_index is not None:
            for pair_key, paths in self.movie_pairs.items():
                flags = [self.recording_index.get_movie_exclusion_flags(Path(original_name(p))) for p in paths]
                if any(value != flags[0] for value in flags[1:]):
                    raise ValueError(f"Conflicting recording occupancy for movie pair: {pair_key}")
                self._recording_exclusions[pair_key] = flags[0]
                if all(flags[0]):
                    self.recording_skipped_keys.append(pair_key)

        # Build review list
        keys = sorted(set(self.movie_pairs) - set(self.recording_skipped_keys))
        if not self.review_all:
            keys = [k for k in keys if not self.store.has(k)]
        self.review_keys = keys
        self.review_movies = [choose_representative_movie(self.movie_pairs[k]) for k in self.review_keys]
        self.current_idx = 0

    def _load_state_for_pair(self, pair_key: str, reviewed_movie: str) -> None:
        """Combine saved manual and recording-time exclusions for the current pair."""
        rec = self.store.get(pair_key)
        if rec is None:
            self.excluded_flags = [0, 0, 0, 0, 0, 0]
        else:
            self.excluded_flags = list(rec.exclude_ch)
        self._recording_flags = list(self._recording_exclusions.get(pair_key, [0] * 6))
        self.excluded_flags = [int(manual or recorded) for manual, recorded in
                               zip(self.excluded_flags, self._recording_flags)]

        self._current_pair_key = pair_key
        self._current_movie_path = reviewed_movie

    def _save_recording_only_exclusions(self) -> None:
        """Keep skipped cameras covered for cropping, preserving existing reviews."""
        changed = False
        for pair_key in self.recording_skipped_keys:
            if not self.store.has(pair_key):
                self.store.upsert(
                    pair_key=pair_key,
                    reviewed_movie=os.path.basename(choose_representative_movie(self.movie_pairs[pair_key])),
                    exclude_ch=[1] * 6,
                )
                changed = True
        if changed:
            # Stop on save failure; otherwise cropping could see incomplete coverage.
            self.store.write_atomic()

    def _save_current_state(self) -> None:
        """Save the current pair exclusions and report recoverable storage failures."""
        if not self._current_pair_key or not self._current_movie_path:
            return
        self.store.upsert(
            pair_key=self._current_pair_key,
            reviewed_movie=os.path.basename(self._current_movie_path),
            exclude_ch=[int(manual or recorded) for manual, recorded in
                        zip(self.excluded_flags, self._recording_flags)],
        )
        try:
            self.store.write_atomic()
        except (OSError, RuntimeError) as exc:
            # Do not crash the OpenCV review session just because a Windows/SMB
            # share temporarily refuses to replace exclusions.csv.  The store
            # still has this record in memory, and write_atomic leaves a full
            # recovery CSV behind if it cannot update the destination.
            print(f"WARNING: moviechecker could not save exclusions.csv: {exc}")

    def _close_cap(self) -> None:
        """Release the current video capture, if open."""
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception:
                pass
        self._cap = None

    def _open_current_movie(self) -> None:
        """Open the selected movie, restore exclusions, and initialize its display layout."""
        self._close_cap()
        movie_path = self.review_movies[self.current_idx]
        pair_key = self.review_keys[self.current_idx]
        self._load_state_for_pair(pair_key, movie_path)

        cap = cv2.VideoCapture(movie_path)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open movie: {movie_path}")
        self._cap = cap
        self._paused = False
        self._pending_seek_frame = None

        # Initialize layout based on the first frame
        ok, frame = cap.read()
        if not ok or frame is None:
            raise RuntimeError(f"Failed to read first frame from: {movie_path}")
        frame_h, frame_w = frame.shape[:2]
        self.layout = compute_ui_layout(frame_w, frame_h, max_display_w=self.max_display_w, max_display_h=self.max_display_h)

        # Rewind to start so playback begins at frame 0
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    def _mouse_callback(self, event: int, x: int, y: int, flags: int, param: object) -> None:
        """Handle seeking and manual exclusion clicks while preserving recording exclusions."""
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        if self.layout is None:
            return

        # Seekbar click
        sx1, sy1, sx2, sy2 = self.layout.seekbar_rect
        if sy1 <= y <= sy2:
            if self._cap is None:
                return
            frame_count = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            if frame_count <= 1:
                return
            frac = max(0.0, min(1.0, (x - sx1) / float(max(1, (sx2 - sx1)))))
            target = int(frac * (frame_count - 1))
            self._pending_seek_frame = target
            return

        # Tile click
        for ch, (x1, y1, x2, y2) in self.layout.tile_rects.items():
            if x1 <= x <= x2 and y1 <= y <= y2:
                idx = ch - 1
                if self._recording_flags[idx]:
                    print(f"Chamber {ch} is excluded by the recording records. "
                          "Correct those records if its occupancy is wrong.")
                    return
                self.excluded_flags[idx] = 0 if self.excluded_flags[idx] else 1
                return

    def run(self) -> None:
        """Run movie review and persist exclusions on navigation or exit."""
        self.scan_movies()
        self._save_recording_only_exclusions()
        if self.recording_index is not None:
            print(f"Skipped {len(self.recording_skipped_keys)} cameras excluded by recording metadata; "
                  f"{len(self.review_movies)} cameras need review.", flush=True)
        if not self.review_movies:
            print("No movies to review (empty folder, already reviewed, or excluded by recording metadata).")
            return

        # AUTOSIZE keeps mouse coordinates aligned with the internally scaled image.
        cv2.namedWindow("moviechecker", cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback("moviechecker", self._mouse_callback)

        self._open_current_movie()

        last_frame: Optional[np.ndarray] = None

        while True:
            assert self._cap is not None
            assert self.layout is not None

            # Apply pending seek
            if self._pending_seek_frame is not None:
                self._cap.set(cv2.CAP_PROP_POS_FRAMES, int(self._pending_seek_frame))
                self._pending_seek_frame = None

                ok, frame = self._cap.read()
                if ok and frame is not None:
                    last_frame = frame

            if not self._paused:
                ok, frame = self._cap.read()
                if not ok or frame is None:
                    # Loop to start if we hit EOF
                    self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ok, frame = self._cap.read()
                    if not ok or frame is None:
                        frame = None

                if frame is not None:
                    last_frame = frame
                    if self._playback_stride > 1:
                        for _ in range(self._playback_stride - 1):
                            if not self._cap.grab():
                                break

            # If paused, keep showing the last frame.
            if last_frame is None:
                ix1, iy1, ix2, iy2 = self.layout.image_rect
                last_frame = np.zeros((iy2 - iy1, ix2 - ix1, 3), dtype=np.uint8)

            fps = float(self._cap.get(cv2.CAP_PROP_FPS) or 30.0)
            delay_ms = max(1, int(1000.0 / max(1.0, fps)))

            frame_count = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            frame_pos_next = int(self._cap.get(cv2.CAP_PROP_POS_FRAMES) or 0)
            frame_pos = max(0, frame_pos_next - 1)

            label = os.path.basename(self._current_movie_path or "")
            disp = _draw_ui_overlays(
                last_frame,
                layout=self.layout,
                excluded_chambers=self.excluded_flags,
                progress_idx=self.current_idx,
                progress_total=len(self.review_movies),
                file_label=label,
                paused=self._paused,
                playback_stride=self._playback_stride,
                frame_pos=frame_pos,
                frame_count=frame_count,
                recording_excluded_chambers=self._recording_flags,
            )

            cv2.imshow("moviechecker", disp)

            key = cv2.waitKey(delay_ms) & 0xFFFFFFFF

            # Quit
            if key in {27, ord("q"), ord("Q")}:
                self._save_current_state()
                break

            # Pause toggle
            if key == 32:  # space
                self._paused = not self._paused
                continue

            # Playback speed (frame skipping)
            if key in {ord(']'), ord('=')}:
                self._playback_stride = min(16, self._playback_stride + 1)
                continue
            if key in {ord('['), ord('-')}:
                self._playback_stride = max(1, self._playback_stride - 1)
                continue

            # Navigation
            forward_keys = {ord("f"), ord("F"), 83, 2555904}
            back_keys = {ord("d"), ord("D"), 81, 2424832}

            if key in forward_keys:
                self._save_current_state()
                if self.current_idx < len(self.review_movies) - 1:
                    self.current_idx += 1
                    self._open_current_movie()
                continue

            if key in back_keys:
                self._save_current_state()
                if self.current_idx > 0:
                    self.current_idx -= 1
                    self._open_current_movie()
                continue

        self._close_cap()
        cv2.destroyAllWindows()
