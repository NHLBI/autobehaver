"""Core logic for running DeepLabCut inference.

DeepLabCut and TensorFlow are imported lazily inside functions so that
Autobehaver can be imported in environments where those optional dependencies
are not installed.
"""

# NOTE: This module intentionally avoids importing DeepLabCut at import-time.

from __future__ import annotations

import glob
import fnmatch
import atexit
import csv
import inspect
import os
import re
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from .config import DLCInferencerConfig

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None


# ---------------------------
# Dependency helpers
# ---------------------------


def _require_deeplabcut():
    try:
        import deeplabcut  # type: ignore

        return deeplabcut
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "DeepLabCut is not available in this Python environment. "
            "On the HPC, load the DLC module / activate the DLC conda env, then rerun."
        ) from e


def _require_tensorflow():
    try:
        import tensorflow as tf  # type: ignore

        return tf
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "TensorFlow is not available in this Python environment. "
            "DeepLabCut typically depends on TF; ensure you're in the correct DLC environment."
        ) from e


# ---------------------------
# Utilities
# ---------------------------


def setup_gpu() -> None:
    """Configure GPU memory growth to prevent TF from reserving all GPU memory."""
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
    """Resolve the list of video files to process.

    Priority:
      1) explicit list (``videos``)
      2) glob pattern (``video_glob``)
      3) default: ``*{videotype}`` in input_dir
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
        pattern = video_glob if video_glob else f"*{videotype}"
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
            if "retry_attempt_" in Path(m).parts:
                continue
            if videotype and not m.lower().endswith(videotype.lower()):
                continue
            resolved.append(m)

    resolved = sorted(set(resolved))
    return resolved


def load_config_bodyparts(config_path: str) -> Optional[List[str]]:
    """Load bodyparts list from config.yaml if possible."""
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


def _prepare_portable_dlc_config(config_path: str) -> str:
    """Return a temporary DLC config with project_path set to this extracted project.

    Bundled DLC resources are portable after extraction, but DLC's config.yaml
    stores ``project_path`` as an absolute path. At runtime we write a small
    temporary config with ``project_path`` pointing at the extracted DLC resource
    folder.
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

    project_root = os.path.abspath(os.path.dirname(config_path))
    if str(cfg.get("project_path", "")).strip() == project_root:
        return config_path

    cfg["project_path"] = project_root
    if cfg.get("video_sets") is None:
        cfg["video_sets"] = {}

    fd, tmp_path = tempfile.mkstemp(prefix="autobehaver_dlc_config_", suffix=".yaml")
    with os.fdopen(fd, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    def _cleanup(path: str = tmp_path) -> None:
        try:
            if os.path.exists(path):
                os.unlink(path)
        except Exception:
            pass

    atexit.register(_cleanup)
    return tmp_path


def check_dlc_outputs(video_path: str, output_dir: Optional[str] = None) -> bool:
    """Check if required DLC outputs exist for a given video."""
    video_path = os.path.abspath(video_path)
    stem = Path(video_path).stem
    search_dir = os.path.abspath(output_dir) if output_dir else str(Path(video_path).parent)

    # DLC outputs start with the video stem; use filename startswith for robustness.
    h5s = glob.glob(os.path.join(search_dir, f"{stem}*.h5"))
    csvs = glob.glob(os.path.join(search_dir, f"{stem}*.csv"))
    metas = glob.glob(os.path.join(search_dir, f"{stem}*_meta.pickle"))

    has_h5 = any(Path(p).name.startswith(stem) for p in h5s)
    has_csv = any(Path(p).name.startswith(stem) for p in csvs)
    has_meta = any(Path(p).name.startswith(stem) for p in metas)

    return has_h5 and has_csv and has_meta


def strip_dlc_tag_from_csvs(
    directory: str,
    dlc_tag: str,
    *,
    log_subdir: str = "filename_changes_log",
    log_filename: str = "filename_changes_log.csv",
) -> int:
    """Remove a user-provided DLC tag substring from DLC CSV filenames in a directory.

    DLC produces filenames like:
        <video_stem>DLC_resnet50_...shuffle1_1000000.csv
    This function removes the provided tag, cleans up double underscores, and
    logs the changes.

    Notes:
      - We only rename *.csv files (to avoid interfering with DLC behaviors that
        may rely on scorer-tagged .h5/_meta.pickle names).
      - Collisions are not overwritten; they are logged as skipped.

    Returns: number of files renamed.
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
    """Call a DLC function while only passing kwargs it accepts."""
    try:
        sig = inspect.signature(func)
        accepts_var_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
        if accepts_var_kwargs:
            return func(*args, **kwargs)

        allowed = set(sig.parameters.keys())
        filtered = {k: v for k, v in kwargs.items() if k in allowed}
        return func(*args, **filtered)
    except Exception:
        # If introspection fails for any reason, fall back to calling directly.
        return func(*args, **kwargs)


def _write_temp_config(source_config: str, updates: Dict[str, object], temp_dir: str) -> str:
    """Write a temporary config.yaml with updated top-level keys."""
    if yaml is None:
        raise RuntimeError(
            "PyYAML not available; cannot create a temporary config. "
            "Install pyyaml or upgrade DLC and pass plotting options directly."
        )
    with open(source_config, "r") as f:
        cfg = yaml.safe_load(f)

    for k, v in updates.items():
        cfg[k] = v

    tmp_cfg_path = os.path.join(temp_dir, "config_temp_for_labeled_video.yaml")
    with open(tmp_cfg_path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    return tmp_cfg_path


# ---------------------------
# Core pipeline
# ---------------------------


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
) -> Dict[str, Dict[str, object]]:
    """Run DLC inference, then retry failed videos individually up to max_retries."""
    dlc = _require_deeplabcut()

    if not videos:
        print("No videos found to analyze.")
        return {}

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

    # Track status
    video_status: Dict[str, Dict[str, object]] = {}
    for v in videos:
        video_status[v] = {"attempts": 1, "success": check_dlc_outputs(v, output_dir=resolved_output_dir)}

    attempt = 1
    while attempt < max_retries:
        failed_videos = [v for v, s in video_status.items() if not s["success"]]
        if not failed_videos:
            print(f"{log_prefix}All videos analyzed successfully!")
            break

        attempt += 1
        print(f"{log_prefix}Retry attempt {attempt} for {len(failed_videos)} video(s)")

        # Retry individually so one bad file can't block the rest.
        for v in failed_videos:
            try:
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
            video_status[v]["success"] = check_dlc_outputs(v, output_dir=resolved_output_dir)

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
) -> None:
    """Create DLC labeled videos (source video + keypoint dots)."""
    dlc = _require_deeplabcut()

    if not videos:
        print(f"{log_prefix}No videos to label.")
        return

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

    # Build kwargs for create_labeled_video.
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
        destfolder=output_dir,
        # plotting controls
        dotsize=dotsize,
        pcutoff=pcutoff,
        colormap=colormap,
    )

    # If this DLC build doesn't accept dotsize/pcutoff/colormap kwargs, fall back
    # to a temporary config copy that sets them as top-level config keys.
    try:
        sig = inspect.signature(dlc.create_labeled_video)
        accepts_var_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
        accepted = set(sig.parameters.keys())
    except Exception:
        accepts_var_kwargs = True
        accepted = set()

    need_temp_config = False
    updates_for_temp_cfg: Dict[str, object] = {}
    for k, v in (("dotsize", dotsize), ("pcutoff", pcutoff), ("colormap", colormap)):
        if not accepts_var_kwargs and k not in accepted:
            need_temp_config = True
            updates_for_temp_cfg[k] = v

    cfg_to_use = config_path
    tmpdir_obj = None
    if need_temp_config:
        print(
            f"{log_prefix}Your DLC version doesn't accept some plotting kwargs directly; "
            f"using a temporary config copy for dotsize/pcutoff/colormap."
        )
        tmpdir_obj = tempfile.TemporaryDirectory(prefix="dlc_cfg_tmp_")
        cfg_to_use = _write_temp_config(config_path, updates_for_temp_cfg, tmpdir_obj.name)

    print(f"{log_prefix}Creating labeled videos for {len(videos)} video(s)...")
    _call_dlc_func(dlc.create_labeled_video, cfg_to_use, list(videos), **kwargs)

    if tmpdir_obj is not None:
        tmpdir_obj.cleanup()


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


# ---------------------------
# Orchestration
# ---------------------------


def run_dlc(cfg: DLCInferencerConfig) -> int:
    """Run DLC inference according to the provided config.

    Returns an exit code (0 = all successful, 1 = some failed, 2 = no videos).
    """
    # Make sure DLC is importable early so failures are obvious.
    _require_deeplabcut()

    setup_gpu()

    runtime_config_path = _prepare_portable_dlc_config(cfg.dlc_config)

    input_dir = os.path.abspath(cfg.input_dir)

    output_dir = os.path.abspath(cfg.output_dir) if cfg.output_dir else None
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    log_dir = output_dir or input_dir

    # -------------------------
    # Resolve videos to process
    # -------------------------
    # Priority:
    #   1) explicit --videos
    #   2) registry (if enabled)
    #   3) filesystem scan of input_dir
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
                from autobehaver.registry import read_registry  # lazy import; pulls in pandas
            except Exception as e:
                raise RuntimeError(
                    "Failed to import registry helpers (pandas required). "
                    "Install pandas in this environment or disable registry."
                ) from e

            df = read_registry(reg_path)

            vids: List[str] = []
            if not df.empty and "cropped_movie_relpath" in df.columns:
                # pandas is optional here; used only for isna convenience.
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
                    # Default: treat unknown as active
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

                rels = [
                    str(x).replace("\\", "/")
                    for x in df2["cropped_movie_relpath"].tolist()
                    if str(x).strip() != ""
                ]

                for rel in rels:
                    p = (project_root / rel).resolve()
                    if not p.exists():
                        # Fallback: registry may have been created on a different mirror.
                        cand = Path(input_dir) / Path(rel).name
                        if cand.exists():
                            p = cand.resolve()
                    if not p.exists():
                        continue
                    if cfg.videotype and not str(p).lower().endswith(str(cfg.videotype).lower()):
                        continue
                    vids.append(str(p))

            videos = sorted(set(vids))
            used_registry = True

            # Optional glob filtering (applies to basenames)
            if videos and cfg.video_glob:
                pat = str(cfg.video_glob)
                videos = [v for v in videos if fnmatch.fnmatch(Path(v).name, pat)]
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
        videos = [v for v in videos if needle in Path(v).name.lower()]
        print(f"Applied file_filter='{cfg.file_filter}': {len(videos)}/{before} videos selected")

    if not videos:
        print(f"No videos matched in: {input_dir}")
        return 2

    print(f"Found {len(videos)} video(s) to process under: {input_dir}")
    for v in videos[:10]:
        print(f"  - {_safe_relpath(v, input_dir)}")
    if len(videos) > 10:
        print(f"  ... and {len(videos)-10} more")

    # Build a baseline status map from existing outputs.
    existing_ok: Dict[str, bool] = {v: check_dlc_outputs(v, output_dir=output_dir) for v in videos}
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
            processed = analyze_videos_with_retries(
                config_path=runtime_config_path,
                videos=videos_to_process,
                videotype=cfg.videotype,
                shuffle=cfg.shuffle,
                save_as_csv=True,
                max_retries=cfg.max_retries,
                output_dir=cfg.output_dir,
            )
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

    if cfg.make_movie:
        # Only label the successful ones (need .h5/.csv/_meta.pickle).
        label_videos = [v for v, s in video_status.items() if s.get("success")]
        if not label_videos:
            print("No successful analyses found; skipping labeled video creation.")
        else:
            create_labeled_movies(
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
            )

    # Optional: strip a user-provided DLC tag from CSV filenames.
    if (not getattr(cfg, "keep_dlc_tag", False)) and cfg.dlc_tag_to_remove and cfg.dlc_tag_to_remove.strip():
        rename_dir = output_dir or input_dir
        n = strip_dlc_tag_from_csvs(rename_dir, cfg.dlc_tag_to_remove)
        if n:
            print(f"Stripped DLC tag from {n} CSV file(s) in: {rename_dir}")

    # -------------------------
    # Registry update (optional)
    # -------------------------
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
                        videos=[Path(v) for v in videos],
                        output_dir=Path(output_dir) if output_dir else None,
                        lock_timeout_sec=int(getattr(cfg, "registry_lock_timeout_sec", 120)),
                        lock_stale_sec=int(getattr(cfg, "registry_lock_stale_sec", 60 * 60)),
                    )
                except Exception as e:
                    print(f"WARNING: Registry update skipped/failed for DLC stage: {e}", file=sys.stderr)
            else:
                print(
                    f"WARNING: Registry enabled but CSV not found (skipping registry update): {reg_path}",
                    file=sys.stderr,
                )

    # Log
    write_summary_log(
        log_dir,
        input_dir=input_dir,
        video_status=video_status,
        made_movies=bool(cfg.make_movie),
        skipped_existing=skipped_existing,
        file_filter=cfg.file_filter,
    )

    return 0 if bad == 0 else 1
