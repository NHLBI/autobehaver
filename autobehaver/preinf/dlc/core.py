"""Run DeepLabCut inference, validate outputs, and publish processing records.

Import DeepLabCut and TensorFlow when needed so other stages remain importable
without those optional dependencies.
"""

from __future__ import annotations

from autobehaver.cli_summary import reported_stage, completion_details, validation_details

from autobehaver.progress import progress, phase

import glob
import fnmatch
import atexit
import csv
import inspect
import os
import re
import sys
import tempfile
import hashlib
import uuid
from autobehaver.portable_paths import generated_path,original_name
import shutil
from autobehaver.completion import (receipt_matches, write_receipt, preserve_previous, verified_run,
                                   file_digest, file_signature, remember_verified_digest, native_path)
from autobehaver.scratch import scratch_workspace
from .contract import model_identity, dlc_request
from ..autocropper.selection import filter_included_paths, propagate_selection
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from autobehaver.validation import StageValidationReport

from .config import DLCInferencerConfig

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None


# Dependency helpers


def _require_deeplabcut():
    """Import DeepLabCut or raise an environment-specific dependency error."""
    try:
        import deeplabcut  # type: ignore

        return deeplabcut
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "DeepLabCut is not available in this Python environment. "
            "Keep Autobehaver in its supported main environment and pass "
            "--dlc_python with the existing DLC interpreter or launcher "
            "(for example, python-DLC after loading the cluster module)."
        ) from e


def _require_tensorflow():
    """Import TensorFlow or raise an environment-specific dependency error."""
    try:
        import tensorflow as tf  # type: ignore

        return tf
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "TensorFlow is not available in this Python environment. "
            "DeepLabCut typically depends on TF; ensure you're in the correct DLC environment."
        ) from e


# Utilities


def setup_gpu() -> None:
    """Enable TensorFlow GPU memory growth instead of reserving all device memory."""
    tf = _require_tensorflow()

    gpus = tf.config.experimental.list_physical_devices("GPU")
    if not gpus:
        return
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e, file=sys.stderr)


def find_processing_log(log_dir: str) -> Optional[str]:
    """Find the processing log file in a directory (stage output folder)."""
    try:
        for file in os.listdir(log_dir):
            if file.endswith("_processing_log.txt"):
                return os.path.join(log_dir, file)
    except FileNotFoundError:
        return None
    return None


def _safe_relpath(path: str, start: str) -> str:
    try:
        return os.path.relpath(path, start=start)
    except Exception:
        return path


def resolve_videos(
    input_dir: str,
    videotype: str = ".mp4",
    videos: Optional[Sequence[str]] = None,
    video_glob: Optional[str] = None,
    recursive: bool = False,
) -> List[str]:
    """Resolve explicit videos or scan the input directory with an optional filename glob.

    Basenames in the explicit list resolve under input_dir; relative paths with
    directory components resolve from the current working directory.
    """
    input_dir = os.path.abspath(input_dir)
    if not os.path.isdir(input_dir):
        raise FileNotFoundError(f"--input directory not found: {input_dir}")

    resolved: List[str] = []

    if videos:
        for v in videos:
            # Allow either absolute/relative paths or just basenames.
            cand = v
            if not os.path.isabs(cand) and os.path.sep not in cand:
                cand = os.path.join(input_dir, cand)
            cand = os.path.abspath(cand)
            if not os.path.exists(cand):
                raise FileNotFoundError(f"Video not found: {cand}")
            resolved.append(cand)
    else:
        pattern = f"*{videotype}"
        if recursive:
            search_pattern = os.path.join(input_dir, "**", pattern)
            matches = glob.glob(search_pattern, recursive=True)
        else:
            search_pattern = os.path.join(input_dir, pattern)
            matches = glob.glob(search_pattern, recursive=False)

        # Keep only files that look like videos and skip retry dirs.
        for m in matches:
            m = os.path.abspath(m)
            if os.path.isdir(m):
                continue
            if any(part == '_old' or part.startswith('retry_attempt_')
                   or part.startswith('.dlc-') for part in Path(m).relative_to(input_dir).parts[:-1]):
                continue
            if Path(m).stem.lower().endswith('_labeled'):
                continue
            if videotype and not m.lower().endswith(videotype.lower()):
                continue
            if video_glob and not fnmatch.fnmatch(original_name(m), video_glob):
                continue
            resolved.append(m)

    resolved = sorted(set(resolved))
    return resolved


def _validate_video_names(videos):
    """DLC stages all inputs together and resolves them by native video stem."""
    for label, name in (('native', lambda p: Path(p).stem),
                        ('logical', lambda p: Path(original_name(p)).stem)):
        seen = {}
        for video in videos:
            key = name(video).casefold()
            if key in seen and Path(seen[key]).resolve() != Path(video).resolve():
                raise ValueError(f'DLC {label} video name collision: {seen[key]}; {video}. '
                                 'Select uniquely named recording inputs before analysis.')
            seen[key] = video


def load_config_bodyparts(config_path: str) -> Optional[List[str]]:
    """Read the bodypart list from config.yaml, returning None when unavailable."""
    if yaml is None:
        return None
    try:
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)
        bps = cfg.get("bodyparts", None)
        if isinstance(bps, list) and all(isinstance(x, str) for x in bps):
            return bps
    except Exception:
        return None
    return None


def _project_model_root(config_path, cfg):
    """Resolve a configured model project, falling back to the config's folder."""
    source_root = Path(config_path).resolve().parent
    configured = cfg.get("project_path")
    if configured:
        candidate = Path(str(configured)).expanduser()
        if not candidate.is_absolute():
            candidate = source_root / candidate
        if any((candidate / name).is_dir() for name in ("dlc-models", "dlc-models-pytorch")):
            return candidate.resolve()
    return source_root


def _write_project_config(cfg, source_root, temporary_root) -> str:
    """Keep a temporary config beside its linked models for DLC path resolution."""
    source_root = Path(source_root).resolve()
    temporary_root = Path(temporary_root).resolve()
    temporary_root.mkdir(parents=True, exist_ok=True)
    for name in ("dlc-models", "dlc-models-pytorch"):
        source = source_root / name
        if not source.is_dir():
            continue
        target = temporary_root / name
        try:
            target.symlink_to(source.resolve(), target_is_directory=True)
        except OSError:
            if os.name != "nt":
                raise
            # Windows may require privileges for directory symlinks. Hardlink
            # the small model directory tree without duplicating model bytes.
            shutil.copytree(source.resolve(), target, copy_function=os.link)
    cfg = dict(cfg)
    cfg["project_path"] = str(temporary_root)
    path = temporary_root / "config.yaml"
    with path.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(cfg, stream, sort_keys=False)
    return str(path)


def _prepare_portable_dlc_config(config_path: str) -> str:
    """Stage an isolated DLC project with unchanged settings and linked models.

    DLC resolves project_path from the config's directory, so a standalone YAML
    in scratch would hide the trained model tree. Preserve the source project
    and register temporary-project cleanup for process exit.
    """
    config_path = os.path.abspath(str(config_path))
    if yaml is None:
        return config_path
    try:
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f) or {}
        if not isinstance(cfg, dict):
            return config_path
    except Exception:
        return config_path

    if cfg.get("video_sets") is None:
        cfg["video_sets"] = {}
    project = tempfile.TemporaryDirectory(prefix="autobehaver_dlc_project_")
    try:
        path = _write_project_config(cfg, _project_model_root(config_path, cfg), project.name)
    except BaseException:
        project.cleanup()
        raise
    atexit.register(project.cleanup)
    return path


class _RetainDLCScratch(RuntimeError):
    pass


def build_dlc_output_index(directory):
    """Index CSV, HDF5, and metadata files by recording stem, retaining ambiguous candidates."""
    index = {}
    root = Path(directory)
    if not root.exists():
        return index
    for path in root.iterdir():
        name = original_name(path)
        suffix = next((s for s in ('_meta.pickle', '.h5', '.csv') if name.endswith(s)), None)
        if suffix is None:
            continue
        base = name[:-len(suffix)]
        stems = {base}
        for match in re.finditer('DLC', base):
            stems.add(base[:match.start()])
            if match.start() and base[match.start()-1] == '_':
                stems.add(base[:match.start()-1])
        key = {'_meta.pickle':'metadata', '.h5':'h5', '.csv':'csv'}[suffix]
        for stem in stems:
            index.setdefault(stem, {'h5': [], 'csv': [], 'metadata': []})[key].append(str(path))
    return index


def _copy_verified(source, target):
    """Hash source bytes while copying and independently verify the destination."""
    before = file_signature(source)
    digest = hashlib.sha256()
    with open(native_path(source), 'rb') as stream, open(native_path(target), 'wb') as output:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            output.write(block)
            digest.update(block)
    if file_signature(source) != before:
        raise OSError('DLC source changed during publication copy')
    value = digest.hexdigest()
    if file_digest(target) != value:
        raise OSError('DLC publication copy failed its checksum')
    remember_verified_digest(source, value, before)
    return value


def inspect_dlc_outputs(
    video_path: str,
    output_dir: Optional[str] = None,
    expected_bodyparts=None,
    output_index=None,
) -> tuple[bool, str, Dict[str, object]]:
    """Validate the unique DLC CSV/HDF5/metadata set for one source video."""

    video_path = os.path.abspath(video_path)
    stem = Path(original_name(video_path)).stem
    search_dir = os.path.abspath(output_dir) if output_dir else str(Path(video_path).parent)

    if output_index is None:
        output_index = build_dlc_output_index(search_dir)
    # Fresh DLC files use the physical input basename; published files restore
    # its complete recording identity through the name map. Accept both during
    # staging without treating two matching output sets as a unique result.
    candidates = [output_index.get(key, {}) for key in {stem, Path(video_path).stem}]
    h5s = sorted({p for group in candidates for p in group.get('h5', [])})
    csvs = sorted({p for group in candidates for p in group.get('csv', [])})
    metas = sorted({p for group in candidates for p in group.get('metadata', [])})

    counts = {"h5": len(h5s), "csv": len(csvs), "metadata": len(metas)}
    if counts != {"h5": 1, "csv": 1, "metadata": 1}:
        return (
            False,
            "expected exactly one matching DLC HDF5, CSV, and metadata file; "
            f"found {counts}",
            {"candidates": {"h5": h5s, "csv": csvs, "metadata": metas}},
        )

    try:
        import cv2
        import pandas as pd

        capture = cv2.VideoCapture(video_path)
        if not capture.isOpened():
            capture.release()
            return False, "source video cannot be decoded", {}
        video_rows = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        readable, _frame = capture.read()
        capture.release()
        if not readable:
            return False, "source video has no readable frames", {}

        csv_frame = pd.read_csv(csvs[0], header=[0, 1, 2], index_col=0)
        h5_frame = pd.read_hdf(h5s[0])
        csv_rows = int(len(csv_frame))
        h5_rows = int(len(h5_frame))
        if csv_rows < 1 or h5_rows < 1:
            return False, "DLC CSV or HDF5 output contains no frames", {}
        if csv_rows != h5_rows:
            return False, f"DLC CSV/HDF5 row mismatch ({csv_rows} != {h5_rows})", {}
        if video_rows > 0 and csv_rows != video_rows:
            return False, f"DLC row count {csv_rows} does not match video frame count {video_rows}", {}
        if csv_frame.columns.duplicated().any() or h5_frame.columns.duplicated().any():
            return False, "DLC output contains duplicate coordinate columns", {}
        from autobehaver.preinf.frame_identity import consecutive_frames
        import numpy as np
        consecutive_frames(csv_frame.index, source=csvs[0], require_origin=True)
        consecutive_frames(h5_frame.index, source=h5s[0], require_origin=True)
        if not csv_frame.columns.equals(h5_frame.columns):
            return False, 'DLC CSV/HDF5 coordinate schemas differ', {}
        coords = {}
        for scorer, part, coord in csv_frame.columns:
            coords.setdefault((scorer,part),set()).add(coord)
        if not coords or any(value != {'x','y','likelihood'} for value in coords.values()):
            return False, 'DLC output lacks complete coordinate triplets', {}
        if expected_bodyparts is not None and {part for _,part in coords} != set(expected_bodyparts):
            return False, 'DLC bodyparts differ from the requested model configuration', {}
        cv = csv_frame.to_numpy(dtype=float)
        hv = h5_frame.to_numpy(dtype=float)
        if np.isinf(cv).any() or np.isinf(hv).any() or not np.allclose(cv,hv,equal_nan=True,rtol=1e-5,atol=1e-5):
            return False, 'DLC CSV/HDF5 values differ or contain infinity', {}
        likelihood = csv_frame.loc[:, csv_frame.columns.get_level_values(2) == 'likelihood'].to_numpy(dtype=float)
        finite = likelihood[np.isfinite(likelihood)]
        if len(finite) and ((finite < 0) | (finite > 1)).any():
            return False, 'DLC likelihood outside [0,1]', {}
        if os.path.getsize(metas[0]) <= 0:
            return False, "DLC metadata file is empty", {}
    except Exception as exc:
        return False, f"DLC output integrity check failed: {exc}", {}

    return True, "ok", {
        "video_frames": video_rows,
        "rows": csv_rows,
        "h5": h5s[0],
        "csv": csvs[0],
        "metadata": metas[0],
    }


def check_dlc_outputs(video_path: str, output_dir: Optional[str] = None) -> bool:
    """Return true only when the required DLC output set passes integrity checks."""

    ok, _reason, _details = inspect_dlc_outputs(video_path, output_dir=output_dir)
    return ok


def strip_dlc_tag_from_csvs(
    directory: str,
    dlc_tag: str,
    *,
    log_subdir: str = "filename_changes_log",
    log_filename: str = "filename_changes_log.csv",
) -> int:
    """Remove an exact DeepLabCut tag from CSV filenames and log each rename.

    Preserve HDF5/metadata names and skip collisions. Return the number renamed.
    """

    directory = os.path.abspath(directory)
    if not dlc_tag or not dlc_tag.strip():
        return 0
    if not os.path.isdir(directory):
        return 0

    # Compile a literal match pattern (user provides the exact substring).
    pattern = re.compile(re.escape(dlc_tag.strip()))
    ts = datetime.now().isoformat(timespec="seconds")

    log_dir = os.path.join(directory, log_subdir)
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, log_filename)

    write_header = not os.path.exists(log_path)

    renamed = 0
    with open(log_path, mode="a", newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(["Timestamp", "Old Filename", "New Filename", "Status"])

        for filename in sorted(os.listdir(directory)):
            # Skip the log itself.
            if filename == log_filename and os.path.abspath(directory) == os.path.abspath(log_dir):
                continue
            if not filename.lower().endswith(".csv"):
                continue
            if dlc_tag not in filename:
                continue

            old_path = os.path.join(directory, filename)
            if not os.path.exists(old_path):
                w.writerow([ts, filename, "", "missing_source"])
                continue

            new_filename = re.sub(pattern, "", filename)
            # Cleanup: collapse multiple underscores and strip leading/trailing underscores.
            new_filename = re.sub(r"__+", "_", new_filename)
            new_filename = new_filename.strip("_")

            if new_filename == filename:
                w.writerow([ts, filename, new_filename, "nochange"])
                continue

            new_path = os.path.join(directory, new_filename)
            if os.path.exists(new_path):
                w.writerow([ts, filename, new_filename, "collision_exists"])
                continue

            try:
                os.rename(old_path, new_path)
                w.writerow([ts, filename, new_filename, "renamed"])
                renamed += 1
            except Exception as e:
                w.writerow([ts, filename, new_filename, f"error:{type(e).__name__}"])

    return renamed


def _call_dlc_func(func, *args, **kwargs):
    """Call a DeepLabCut function with only the keyword arguments it accepts."""
    try:
        sig = inspect.signature(func)
        accepts_var_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
        allowed = set(sig.parameters.keys())
        filtered = kwargs if accepts_var_kwargs else {k: v for k, v in kwargs.items() if k in allowed}
    except (TypeError, ValueError):
        # If introspection fails for any reason, fall back to calling directly.
        filtered = kwargs
    # A backend failure belongs to the explicit retry loop. Never execute the
    # same expensive backend call again merely because it raised an exception.
    with phase('DeepLabCut ' + getattr(func, '__name__', 'processing')):
        return func(*args, **filtered)


def _write_temp_config(source_config: str, updates: Dict[str, object], temp_dir: str) -> str:
    """Write rendering overrides in an isolated project beside linked models."""
    if yaml is None:
        raise RuntimeError(
            "PyYAML not available; cannot create a temporary config. "
            "Install pyyaml or upgrade DLC and pass plotting options directly."
        )
    with open(source_config, "r") as f:
        cfg = yaml.safe_load(f)

    for k, v in updates.items():
        cfg[k] = v

    return _write_project_config(cfg, _project_model_root(source_config, cfg), temp_dir)


# Core pipeline


def _matching_dlc_files(video, directory):
    """Return indexed artifacts matching a video identity or its physical filename."""
    index = build_dlc_output_index(directory)
    keys = {Path(original_name(video)).stem, Path(video).stem}
    return sorted({Path(path) for key in keys for group in index.get(key, {}).values() for path in group})


def analyze_videos_with_retries(
    *,
    config_path: str,
    videos: Sequence[str],
    videotype: str,
    shuffle: int,
    save_as_csv: bool,
    max_retries: int,
    output_dir: Optional[str] = None,
    log_prefix: str = "",
    expected_bodyparts=None,
    backend=None,
) -> Dict[str, Dict[str, object]]:
    """Run an initial DeepLabCut batch and retry failed videos individually.

    Return per-video attempts, validation details, and errors. max_retries bounds
    total attempts, including the initial batch.
    """
    if not videos:
        print("No videos found to analyze.")
        return {}
    _validate_video_names(videos)
    dlc = backend if backend is not None else _require_deeplabcut()

    resolved_output_dir = os.path.abspath(output_dir) if output_dir else None

    print(f"{log_prefix}Starting DLC analysis for {len(videos)} video(s).")
    if resolved_output_dir:
        print(f"{log_prefix}Outputs will be written to output_dir: {resolved_output_dir}")

    # Initial batch analysis (may process some even if an exception occurs)
    try:
        _call_dlc_func(
            dlc.analyze_videos,
            config_path,
            list(videos),
            shuffle=shuffle,
            save_as_csv=save_as_csv,
            videotype=videotype,
            destfolder=output_dir,
        )
    except Exception as e:
        print(f"{log_prefix}Initial analyze_videos() raised: {e}", file=sys.stderr)

    video_status: Dict[str, Dict[str, object]] = {}
    indexes = {}
    for v in progress(videos, desc='DLC checking video outputs'):
        directory = resolved_output_dir or str(Path(v).parent)
        if directory not in indexes:
            indexes[directory] = build_dlc_output_index(directory)
        valid, reason, details = inspect_dlc_outputs(v, output_dir=directory,
            expected_bodyparts=expected_bodyparts, output_index=indexes[directory])
        video_status[v] = {'attempts':1, 'success':valid, 'details':details, 'error':reason if not valid else None}

    attempt = 1
    while attempt < max_retries:
        failed_videos = [v for v, s in video_status.items() if not s["success"]]
        if not failed_videos:
            print(f"{log_prefix}Analysis finished; final output checks follow.")
            break

        attempt += 1
        print(f"{log_prefix}Retry attempt {attempt} for {len(failed_videos)} video(s)")

        # Retry individually so one bad file can't block the rest.
        for v in failed_videos:
            try:
                directory=resolved_output_dir or str(Path(v).parent)
                for path in _matching_dlc_files(v,directory):
                    preserve_previous(path)
                _call_dlc_func(
                    dlc.analyze_videos,
                    config_path,
                    [v],
                    shuffle=shuffle,
                    save_as_csv=save_as_csv,
                    videotype=videotype,
                    destfolder=output_dir,
                )
            except Exception as e:
                print(f"{log_prefix}Retry analyze_videos() failed for {v}: {e}", file=sys.stderr)

            video_status[v]["attempts"] = int(video_status[v]["attempts"]) + 1
            indexes[directory] = build_dlc_output_index(directory)
            valid, reason, details = inspect_dlc_outputs(v, output_dir=directory,
                expected_bodyparts=expected_bodyparts, output_index=indexes[directory])
            video_status[v].update(success=valid, details=details, error=reason if not valid else None)

    return video_status


def create_labeled_movies(
    *,
    config_path: str,
    videos: Sequence[str],
    videotype: str,
    shuffle: int,
    dotsize: int,
    pcutoff: float,
    colormap: str,
    exclude_bodyparts: Sequence[str],
    color_by: str = "bodypart",
    draw_skeleton: bool = False,
    trailpoints: int = 0,
    fastmode: bool = True,
    codec: str = "mp4v",
    overwrite: bool = True,
    output_dir: Optional[str] = None,
    log_prefix: str = "",
    backend=None,
    analyzed_outputs=None,
    scratch_dir=None,
) -> Dict[str, str]:
    """Render and verify overlays using native DLC names in owned scratch.

    Published prediction files can have portable names that DLC cannot resolve.
    Restore its video/scorer naming only in scratch, leaving published data and
    model files untouched.
    """
    dlc = backend if backend is not None else _require_deeplabcut()

    if not videos:
        print(f"{log_prefix}No videos to label.")
        return {}

    displayed_bodyparts: object = "all"
    bodyparts = load_config_bodyparts(config_path)
    if bodyparts:
        excluded = set(exclude_bodyparts)
        displayed = [bp for bp in bodyparts if bp not in excluded]
        displayed_bodyparts = displayed if displayed else "all"
        if excluded:
            print(f"{log_prefix}Excluding bodyparts: {sorted(excluded)}")
            if displayed_bodyparts != "all":
                print(f"{log_prefix}Displaying {len(displayed_bodyparts)}/{len(bodyparts)} bodyparts")

    kwargs = dict(
        videotype=videotype,
        shuffle=shuffle,
        draw_skeleton=draw_skeleton,
        displayedbodyparts=displayed_bodyparts,
        color_by=color_by,
        trailpoints=trailpoints,
        fastmode=fastmode,
        codec=codec,
        overwrite=overwrite,
        keypoints_only=False,
        # plotting controls
        dotsize=dotsize,
        pcutoff=pcutoff,
        colormap=colormap,
    )

    from autobehaver.preinf.autocropper.contract import video_shape
    import pandas as pd

    with scratch_workspace('dlc_labels', scratch_dir) as staging:
        # DLC 2.3.9 reads these settings from config even when its signature
        # accepts plotting keywords. Keep the config beside linked models.
        cfg_to_use = _write_temp_config(config_path, {
            'dotsize': dotsize, 'pcutoff': pcutoff, 'colormap': colormap,
        }, str(staging / 'model'))
        with open(cfg_to_use, encoding='utf-8') as stream:
            render_config = yaml.safe_load(stream)
        suffix = ('_id' if color_by == 'individual' else '_bp') if render_config.get('multianimalproject') else ''
        plans = []
        seen = set()
        published = {}
        for video in videos:
            details = (analyzed_outputs or {}).get(video)
            if details is None:
                valid, reason, details = inspect_dlc_outputs(video, output_dir=output_dir)
                if not valid:
                    raise RuntimeError(f'Cannot label {video}: {reason}')
            header = pd.read_csv(details['csv'], header=[0, 1, 2], index_col=0, nrows=0)
            scorers = set(header.columns.get_level_values(0))
            if len(scorers) != 1:
                raise ValueError(f'Expected one DLC scorer for labeled video: {video}')
            scorer = str(next(iter(scorers)))
            if not scorer or Path(scorer).name != scorer or '\\' in scorer:
                raise ValueError(f'Invalid DLC scorer filename component: {scorer}')
            native_stem = Path(video).stem + scorer
            if native_stem in seen:
                raise ValueError(f'Labeled video staging name collision: {native_stem}')
            seen.add(native_stem)
            logical_stem = Path(original_name(video)).stem
            scorer_token = hashlib.sha256(scorer.encode()).hexdigest()[:12]
            target_dir = Path(output_dir) if output_dir else Path(video).parent
            target_dir.mkdir(parents=True, exist_ok=True)
            target = generated_path(target_dir, logical_stem + 'DLC' + scorer_token + suffix + '_labeled.mp4',
                                    original=logical_stem + scorer + suffix + '_labeled.mp4')
            shape = video_shape(video)
            from autobehaver.completion import request_identity
            render_request = request_identity('dlc-label-v1',
                [video, details['h5'], details['csv'], details['metadata']],
                {'config': {k: v for k, v in render_config.items()
                            if k not in ('project_path', 'video_sets', 'video_sets_original')},
                 'render': {k: v for k, v in kwargs.items() if k != 'overwrite'}}, [__file__])
            if target.exists() and not overwrite:
                if video_shape(target) != shape or not receipt_matches(target, render_request):
                    raise RuntimeError(f'Existing labeled video does not match current tracking/render settings '
                                       f'and overwrite is disabled: {target}')
                published[video] = str(target)
                continue
            for key, extension in (('h5', '.h5'), ('csv', '.csv'), ('metadata', '_meta.pickle')):
                source = Path(details[key]).resolve()
                link = staging / (native_stem + extension)
                try:
                    link.symlink_to(source)
                except OSError:
                    if os.name != 'nt':
                        raise
                    _copy_verified(source, link)
            plans.append((video, staging / (native_stem + suffix + '_labeled.mp4'), target, shape, render_request))
        if plans:
            print(f"{log_prefix}Creating labeled videos for {len(plans)} video(s)...")
            _call_dlc_func(dlc.create_labeled_video, cfg_to_use,
                           [item[0] for item in plans], destfolder=str(staging), **kwargs)
        # Native DLC may return False without raising. Require a complete,
        # readable overlay for every request before publishing any new movie.
        for video, rendered, target, shape, _request in plans:
            try:
                if video_shape(rendered) != shape:
                    raise ValueError('frame count or dimensions differ from the source')
            except Exception as exc:
                raise RuntimeError(f'Labeled video generation failed for {video}; scratch retained at {staging}: {exc}') from exc
        for video, rendered, target, _shape, render_request in plans:
            temporary = target.parent / ('.dlc-label-' + uuid.uuid4().hex[:12] + '.tmp')
            try:
                digest = _copy_verified(rendered, temporary)
                if target.exists():
                    preserve_previous(target)
                os.replace(temporary, target)
                remember_verified_digest(target, digest)
                write_receipt(target, render_request)
            finally:
                temporary.unlink(missing_ok=True)
            published[video] = str(target)
        return published


def write_summary_log(
    log_dir: str,
    *,
    input_dir: str,
    video_status: Dict[str, Dict[str, object]],
    made_movies: bool,
    skipped_existing: int = 0,
    file_filter: str | None = None,
) -> None:
    """Append a summary to the *_processing_log.txt in the stage output directory."""

    log_dir = os.path.abspath(log_dir)
    os.makedirs(log_dir, exist_ok=True)

    log_file = find_processing_log(log_dir)
    if not log_file:
        log_filename = f"{os.path.basename(os.path.normpath(log_dir))}_processing_log.txt"
        log_file = os.path.join(log_dir, log_filename)
        print(f"No processing log found. Creating a new one at: {log_file}")

    total_videos = len(video_status)
    successful = sum(1 for s in video_status.values() if s.get("success"))
    failed = total_videos - successful

    with open(log_file, "a") as f:
        f.write("\n3. DLC\n")
        f.write(f"   Timestamp: {datetime.now().isoformat(timespec='seconds')}\n")
        f.write(f"   Input directory: {input_dir}\n")
        f.write(f"   Output directory: {log_dir}\n")
        if file_filter:
            f.write(f"   File filter: {file_filter}\n")
        f.write(f"   Total videos targeted: {total_videos}\n")
        f.write(f"   Successfully analyzed: {successful}\n")
        if skipped_existing:
            f.write(f"   Skipped (existing outputs): {skipped_existing}\n")
        if failed > 0:
            f.write(f"   Failed to analyze: {failed}\n")
            f.write("   Failed videos:\n")
            for video, status in video_status.items():
                if not status.get("success"):
                    f.write(f"   - {os.path.basename(video)} (attempts: {status.get('attempts')})\n")
        f.write(f"   Labeled videos requested: {'yes' if made_movies else 'no'}\n")


# Orchestration


def _monitor_h5_signature(details):
    """Return the unique HDF5 path, size, and modification time when available."""
    path = details.get("h5")
    if not path:
        candidates = details.get("candidates", {}).get("h5", [])
        path = candidates[0] if len(candidates) == 1 else None
    if not path:
        return None
    try:
        metadata = Path(path).stat()
        return str(path), metadata.st_size, metadata.st_mtime_ns
    except OSError:
        return None


@reported_stage("DLC", exit_code=True)
@verified_run
def run_dlc(cfg: DLCInferencerConfig) -> int:
    """Run DeepLabCut inference with the supplied runtime options.

    Return 0 when validation succeeds, 1 when it fails, or 2 when no videos match.
    """
    input_dir = os.path.abspath(cfg.input_dir)

    output_dir = os.path.abspath(cfg.output_dir) if cfg.output_dir else None
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    log_dir = output_dir or input_dir

    # Resolve videos to process
    # Priority:
    # explicit --videos
    # registry (if enabled)
    # filesystem scan of input_dir
    used_registry = False

    if cfg.videos:
        videos = resolve_videos(
            input_dir=input_dir,
            videotype=cfg.videotype,
            videos=cfg.videos,
            video_glob=cfg.video_glob,
            recursive=cfg.recursive,
        )
    else:
        reg_csv: str | None = None
        if bool(getattr(cfg, "registry_enabled", False)):
            try:
                # Provided by DLCInferencerConfig in pipeline mode.
                reg_csv = cfg.resolved_registry_csv()  # type: ignore[attr-defined]
            except Exception:
                reg_csv = None

        if reg_csv:
            reg_path = Path(reg_csv).expanduser().resolve()
            if not reg_path.exists():
                raise FileNotFoundError(
                    f"Registry enabled but CSV not found: {reg_path}. "
                    "Run autocropper first (to create registry entries) or disable registry for this run."
                )

            project_root = reg_path.parent

            try:
                from autobehaver.registry import read_registry, resolve_registry_path  # lazy import; pulls in pandas
            except Exception as e:
                raise RuntimeError(
                    "Failed to import registry helpers (pandas required). "
                    "Install pandas in this environment or disable registry."
                ) from e

            df = read_registry(reg_path)

            vids: List[str] = []
            if not df.empty and "cropped_movie_relpath" in df.columns:
                # Handle null active flags without changing compatibility defaults.
                try:
                    import pandas as pd  # type: ignore
                except Exception:  # pragma: no cover
                    pd = None  # type: ignore

                def _is_active(val: object) -> bool:
                    if val is None:
                        return True
                    if pd is not None:
                        try:
                            if pd.isna(val):  # type: ignore[attr-defined]
                                return True
                        except Exception:
                            pass
                    if isinstance(val, bool):
                        return bool(val)
                    s = str(val).strip().lower()
                    if s in {"0", "false", "no", "n", "off"}:
                        return False
                    if s in {"1", "true", "yes", "y", "on"}:
                        return True
                    # Preserve the legacy active default for unrecognized values.
                    return True

                try:
                    df2 = df[df["cropped_movie_relpath"].notna()].copy()
                except Exception:
                    df2 = df.copy()

                if "active" in df2.columns:
                    try:
                        df2 = df2[df2["active"].apply(_is_active)]
                    except Exception:
                        pass

                for _, registry_row in df2.iterrows():
                    rel = str(registry_row["cropped_movie_relpath"]).replace("\\", "/")
                    if not rel.strip():
                        continue
                    p = resolve_registry_path(project_root, registry_row, "cropped_movie_relpath")
                    if cfg.videotype and not str(p).lower().endswith(str(cfg.videotype).lower()):
                        continue
                    vids.append(str(p))

            videos = sorted(set(vids))
            used_registry = True

            # Optional glob filtering (applies to basenames)
            if videos and cfg.video_glob:
                pat = str(cfg.video_glob)
                videos = [v for v in videos if fnmatch.fnmatch(original_name(v), pat)]
        else:
            videos = resolve_videos(
                input_dir=input_dir,
                videotype=cfg.videotype,
                videos=None,
                video_glob=cfg.video_glob,
                recursive=cfg.recursive,
            )

    if cfg.file_filter:
        before = len(videos)
        needle = cfg.file_filter.lower()
        videos = [v for v in videos if needle in original_name(v).lower()]
        print(f"Applied file_filter='{cfg.file_filter}': {len(videos)}/{before} videos selected")

    source_directories = {Path(input_dir), *(Path(video).parent for video in videos)}
    if output_dir:
        propagate_selection(source_directories, output_dir)
    selected_before_exclusions = len(videos)
    videos = filter_included_paths(videos)
    from autobehaver.batch_inputs import select_batch_inputs
    videos = select_batch_inputs(videos, getattr(cfg, "file_list", None))

    if len(videos) != selected_before_exclusions:
        print(f"Excluded {selected_before_exclusions - len(videos)} previously cropped chamber(s) according to current crop selection.")

    if not videos:
        completion_details(status="nothing to process", output=output_dir or input_dir)
        print(f"No videos matched in: {input_dir}")
        return 2

    missing = [v for v in videos if not Path(v).is_file()]
    if missing:
        raise FileNotFoundError(f'Selected cropped movies are missing at their declared paths: {missing[:25]}')
    _validate_video_names(videos)
    if cfg.dlc_python:
        from .backend import SubprocessDLC
        dlc = SubprocessDLC(cfg.dlc_python)
    else:
        dlc = _require_deeplabcut()
        setup_gpu()
    runtime_config_path = _prepare_portable_dlc_config(cfg.dlc_config)

    print(f"Found {len(videos)} video(s) to process under: {input_dir}")
    for v in videos[:10]:
        print(f"  - {_safe_relpath(v, input_dir)}")
    if len(videos) > 10:
        print(f"  ... and {len(videos)-10} more")

    # Build a baseline status map from existing outputs.
    from autobehaver.resource_monitor import current_monitor, record_work

    before_h5 = {}
    requested_model = model_identity(runtime_config_path, cfg.shuffle)
    prepare_model = getattr(dlc, "prepare_model", None)
    if callable(prepare_model):
        prepare_model(runtime_config_path, cfg.shuffle)
    requests = {v:dlc_request(v,requested_model) for v in videos}
    source_signatures = {v:file_signature(v) for v in videos}
    existing_ok = {}
    existing_indexes = {}
    validated_outputs = {}
    for video in videos:
        directory = output_dir or str(Path(video).parent)
        if directory not in existing_indexes:
            existing_indexes[directory] = build_dlc_output_index(directory)
        valid, _, details = inspect_dlc_outputs(video, output_dir=output_dir,
                                              expected_bodyparts=requested_model['settings'].get('bodyparts'),
                                              output_index=existing_indexes[directory])
        paths = [details[k] for k in ('h5','csv','metadata')] if valid else []
        existing_ok[video] = bool(valid and receipt_matches(details['csv'],requests[video],paths))
        if existing_ok[video]:
            validated_outputs[video] = (details, [file_signature(p) for p in paths], file_signature(video))
        if current_monitor() is not None:
            before_h5[video] = _monitor_h5_signature(details)
    video_status: Dict[str, Dict[str, object]] = {
        v: {"attempts": 0, "success": bool(existing_ok[v])} for v in videos
    }

    skipped_existing = 0
    if not cfg.skip_analysis:
        videos_to_process = list(videos)
        if not cfg.clobber:
            videos_to_process = [v for v in videos if not existing_ok[v]]
            skipped_existing = sum(1 for v in videos if existing_ok[v])
            if skipped_existing:
                print(f"Skipping {skipped_existing} video(s) with existing DLC outputs.")

        if videos_to_process:
            processed = {}
            # Keep DLC away from old files: its own internal skip logic otherwise
            # prevents genuine regeneration even when Autobehaver requests it.
            destination = Path(output_dir) if output_dir else Path(videos_to_process[0]).parent
            destination.mkdir(parents=True,exist_ok=True)
            try:
                with scratch_workspace('dlc', getattr(cfg, 'scratch_dir', None)) as staging:
                    processed = analyze_videos_with_retries(
                        config_path=runtime_config_path, videos=videos_to_process,
                        videotype=cfg.videotype, shuffle=cfg.shuffle, save_as_csv=True,
                        max_retries=cfg.max_retries, output_dir=str(staging),
                        expected_bodyparts=requested_model['settings'].get('bodyparts'), backend=dlc)
                    for video, status in processed.items():
                        valid, reason, details = bool(status.get('success')), status.get('error'), status.get('details', {})
                        if valid and not details:
                            valid, reason, details = inspect_dlc_outputs(video, output_dir=str(staging),
                                expected_bodyparts=requested_model['settings'].get('bodyparts'))
                        if not valid:
                            status['success'] = False
                            status['error'] = reason
                            continue
                        if file_signature(video) != source_signatures[video]:
                            status.update(success=False, error='Source video changed during DLC processing')
                            continue
                        target = Path(output_dir) if output_dir else Path(video).parent
                        # Existing output candidates are preserved only after the new
                        # complete set has passed validation.
                        original_stem=Path(original_name(video)).stem
                        scorer_suffix=Path(details['h5']).stem[len(Path(video).stem):]
                        scorer_token=hashlib.sha256(scorer_suffix.encode()).hexdigest()[:12]
                        pending=[]
                        published=[]
                        temporary_paths = []
                        try:
                            for key,suffix in [('h5','.h5'),('csv','.csv'),('metadata','_meta.pickle')]:
                                source=Path(details[key])
                                output=generated_path(target,original_stem+'DLC'+scorer_token+suffix,
                                                      original=original_stem+scorer_suffix+suffix)
                                temporary=target/('.dlc-'+uuid.uuid4().hex[:12]+'.tmp')
                                temporary_paths.append(temporary)
                                digest = _copy_verified(source, temporary)
                                pending.append((temporary,output,digest))
                                published.append(str(output))
                            index = existing_indexes.get(str(target), {})
                            previous = {Path(p) for stem in {original_stem, Path(video).stem}
                                        for group in index.get(stem, {}).values() for p in group}
                            for path in sorted(previous):
                                preserve_previous(path)
                            for temporary,output,digest in pending:
                                os.replace(temporary,output)
                                remember_verified_digest(output, digest)
                            write_receipt(published[1],requests[video],published,rows=details['rows'])
                            final_details = {**details, **dict(zip(('h5','csv','metadata'), published))}
                            validated_outputs[video] = (final_details, [file_signature(p) for p in published], file_signature(video))
                        except Exception as exc:
                            status.update(success=False, error=f'DLC publication failed: {exc}')
                            validated_outputs.pop(video, None)
                        finally:
                            for temporary in temporary_paths:
                                temporary.unlink(missing_ok=True)
                    if any(not status.get('success') for status in processed.values()):
                        raise _RetainDLCScratch(f'DLC failed outputs retained in {staging}')
            except _RetainDLCScratch as exc:
                print(str(exc), file=sys.stderr)
            except Exception as exc:
                for video in videos_to_process:
                    if video not in validated_outputs:
                        processed.setdefault(video, {}).update(success=False, error=f'DLC processing failed: {exc}')
            # Merge results back into the full status map.
            for v, s in processed.items():
                video_status[v] = s
        else:
            print("All selected videos already have DLC outputs; nothing to analyze.")
    else:
        # Skip analysis mode: just report existing outputs.
        pass

    total = len(video_status)
    ok = sum(1 for s in video_status.values() if s.get("success"))
    bad = total - ok
    print("\nFinal Analysis Status:")
    print(f"  Total videos: {total}")
    print(f"  Successful:   {ok}")
    print(f"  Failed:       {bad}")
    if bad:
        print("  Failed videos:")
        for v, s in video_status.items():
            if not s.get("success"):
                print(f"    - {_safe_relpath(v, input_dir)} (attempts: {s.get('attempts')})")

    labeled_outputs = {}
    labeled_error = None
    if cfg.make_movie:
        # Only label the successful ones (need .h5/.csv/_meta.pickle).
        label_videos = [v for v, s in video_status.items() if s.get("success")]
        if not label_videos:
            print("No successful analyses found; skipping labeled video creation.")
        else:
            try:
                labeled_outputs = create_labeled_movies(
                    config_path=runtime_config_path,
                    videos=label_videos,
                    videotype=cfg.videotype,
                    shuffle=cfg.shuffle,
                    dotsize=cfg.dotsize,
                    pcutoff=cfg.pcutoff,
                    colormap=cfg.colormap,
                    exclude_bodyparts=cfg.exclude_bodyparts,
                    color_by=cfg.color_by,
                    draw_skeleton=False,
                    trailpoints=cfg.trailpoints,
                    fastmode=cfg.fastmode,
                    codec=cfg.codec,
                    overwrite=cfg.overwrite,
                    output_dir=cfg.output_dir,
                    backend=dlc,
                    analyzed_outputs={v: validated_outputs[v][0] for v in label_videos if v in validated_outputs},
                    scratch_dir=cfg.scratch_dir,
                )
            except Exception as exc:
                labeled_error = str(exc)
                print(f'Labeled video generation failed: {exc}', file=sys.stderr)

    # Optional: strip a user-provided DLC tag from CSV filenames.
    if (not getattr(cfg, "keep_dlc_tag", False)) and cfg.dlc_tag_to_remove and cfg.dlc_tag_to_remove.strip():
        rename_dir = output_dir or input_dir
        n = strip_dlc_tag_from_csvs(rename_dir, cfg.dlc_tag_to_remove)
        if n:
            print(f"Stripped DLC tag from {n} CSV file(s) in: {rename_dir}")
        for video in videos:
            valid,_,details=inspect_dlc_outputs(video,output_dir=output_dir)
            if valid and video_status[video].get('success'):
                paths=[details[k] for k in ('h5','csv','metadata')]
                write_receipt(details['csv'],requests[video],paths,rows=details['rows'])


    # Registry update (optional)
    registry_error: str | None = None
    if bool(getattr(cfg, "registry_enabled", False)):
        try:
            reg_csv = cfg.resolved_registry_csv()  # type: ignore[attr-defined]
        except Exception:
            reg_csv = None

        if reg_csv:
            reg_path = Path(reg_csv).expanduser().resolve()
            if reg_path.exists():
                try:
                    from .registry_update import update_registry_from_dlc_outputs

                    update_registry_from_dlc_outputs(
                        registry_csv=reg_path,
                        project_root=reg_path.parent,
                        videos=[Path(v) for v in videos if video_status[v].get("success")],
                        output_dir=Path(output_dir) if output_dir else None,
                        lock_timeout_sec=int(getattr(cfg, "registry_lock_timeout_sec", 120)),
                        lock_stale_sec=int(getattr(cfg, "registry_lock_stale_sec", 60 * 60)),
                    )
                except Exception as e:
                    registry_error = str(e)
                    print(f"WARNING: Registry update skipped/failed for DLC stage: {e}", file=sys.stderr)
            else:
                registry_error = f"registry CSV not found: {reg_path}"
                print(
                    f"WARNING: Registry enabled but CSV not found (skipping registry update): {reg_path}",
                    file=sys.stderr,
                )

    # Log
    write_summary_log(
        log_dir,
        input_dir=input_dir,
        video_status=video_status,
        made_movies=bool(labeled_outputs),
        skipped_existing=skipped_existing,
        file_filter=cfg.file_filter,
    )

    validation = StageValidationReport.for_items(
        "dlc",
        videos,
        metadata={"output_directory": str(log_dir), "make_labeled_movies": bool(cfg.make_movie),
                  "labeled_movies": labeled_outputs},
    )
    if labeled_error is not None:
        validation.add_issue('labeled_video_failed', 'error',
                             'Requested labeled videos were not generated and verified.',
                             details={'error': labeled_error})
    output_stems = [Path(video).stem for video in videos]
    duplicate_stems = sorted(
        stem for stem, count in Counter(output_stems).items() if count > 1
    )
    if duplicate_stems:
        validation.add_issue(
            "output_path_collision",
            "error",
            "Selected videos share output stems and would overwrite one another.",
            details={"stems": duplicate_stems[:25]},
        )
    for video in videos:
        checked = validated_outputs.get(video)
        try:
            unchanged = checked is not None and checked[1] == [file_signature(checked[0][key]) for key in ('h5','csv','metadata')]
            unchanged = unchanged and checked[2] == file_signature(video)
        except OSError:
            unchanged = False
        if unchanged:
            ok_output, reason, details = True, 'ok', checked[0]
        else:
            ok_output, reason, details = inspect_dlc_outputs(video, output_dir=output_dir)
        if not video_status[video].get("success"):
            ok_output = False
            reason = str(video_status[video].get("error") or
                         "No successful analysis or verified existing output for the current request")
        elif ok_output and not receipt_matches(details['csv'], requests[video],
                                               [details[key] for key in ('h5', 'csv', 'metadata')]):
            ok_output = False
            reason = "DLC output does not match the requested input/model or its published content"
        if current_monitor() is not None:
            attempted = int(video_status[video].get("attempts", 0)) > 0
            unchanged = before_h5.get(video) is not None and before_h5[video] == _monitor_h5_signature(details)
            if not attempted or (ok_output and unchanged):
                record_work("dlc_videos", skipped=1)
            elif not ok_output:
                record_work("dlc_videos", failed=1)
            else:
                record_work("dlc_videos", completed=1)
                record_work("dlc_output_frames", completed=int(details["rows"]))
        if not ok_output:
            validation.add_outcome(video, "failed", reason=reason, details=details)
        elif existing_ok.get(video, False) and (not cfg.clobber or cfg.skip_analysis):
            validation.add_outcome(
                video,
                "reused",
                reason="existing DLC output set passed integrity checks",
                details=details,
            )
        else:
            validation.add_outcome(video, "succeeded", details=details)
    if registry_error is not None:
        validation.add_issue(
            "registry_update_failed",
            "error",
            "DLC outputs were produced, but the project registry could not be updated.",
            details={"error": registry_error},
        )
    validation.write_run_reports(Path(log_dir) / "DLC_VALIDATION.json")

    validation_details(validation, unit="Videos", output=output_dir or input_dir)
    return 0 if validation.status != "failed" else 1
