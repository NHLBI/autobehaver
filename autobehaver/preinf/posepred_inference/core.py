"""Coordinate PosePred inference and validate per-frame predictions and provenance."""

from __future__ import annotations

import json
import os
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, List

from autobehaver.preinf.frame_identity import h5_frame_timeline
from autobehaver.portable_paths import prediction_path, original_name
import h5py
import numpy as np
import pandas as pd

from autobehaver.validation import StageValidationReport

from .config import PosePredInferenceConfig

from ...inference import (
    _build_inference_provenance,
    _completed_inference_matches,
    _provenance_file_identity,
    run_inference,
)

logger = logging.getLogger(__name__)


def _ensure_list(x):
    """Normalize one or more supplied values into a list of strings."""
    if x is None:
        return []
    if isinstance(x, (list, tuple)):
        return [str(v) for v in x]
    return [str(x)]


def _validate_inference_csv(source_h5: str, inference_csv: Path, expected_heads=None) -> tuple[bool, str, dict]:
    """Validate frame alignment, probability bounds, and emitted label vocabularies."""

    try:
        with h5py.File(source_h5, "r") as handle:
            if "data" not in handle or getattr(handle["data"], "ndim", None) != 2:
                return False, "source HDF5 is missing a 2D data dataset", {}
            expected_rows, _, _ = h5_frame_timeline(handle)
    except Exception as exc:
        return False, f"source HDF5 is unreadable: {exc}", {}
    try:
        frame = pd.read_csv(inference_csv)
    except Exception as exc:
        return False, f"inference CSV is unreadable: {exc}", {}

    if len(frame) != expected_rows:
        return False, f"row count {len(frame)} does not match source row count {expected_rows}", {}
    if "Frame Number" not in frame.columns:
        return False, "inference CSV is missing 'Frame Number'", {}
    expected_frames = np.arange(1, expected_rows + 1, dtype=np.int64)
    try:
        actual_frames = pd.to_numeric(frame["Frame Number"], errors="raise").to_numpy()
    except Exception as exc:
        return False, f"Frame Number contains non-numeric values: {exc}", {}
    if not np.array_equal(actual_frames, expected_frames):
        return False, "Frame Number must be the unique consecutive sequence 1..N", {}

    present_heads = [p for p in ('Pose', 'Orient') if any(str(c).startswith(p + '-') and str(c).endswith('-Prob') for c in frame)]
    if not present_heads:
        return False, 'inference CSV contains no prediction head', {}
    if expected_heads is not None:
        if set(present_heads) != set(expected_heads):
            return False, 'prediction heads differ from the requested checkpoint', {}
        for head,labels in expected_heads.items():
            actual={str(c)[len(head)+1:-5] for c in frame if str(c).startswith(head+'-') and str(c).endswith('-Prob')}
            if actual != set(labels):
                return False, f'{head} vocabulary differs from the requested checkpoint', {}

    if 'Source Frame Index' in frame and not np.array_equal(pd.to_numeric(frame['Source Frame Index'], errors='coerce'), expected_frames - 1):
        return False, 'original frame identity does not match Frame Number', {}

    for prefix, label_column in (
        ("Pose", "Predicted Pose Label"),
        ("Orient", "Predicted Orient Label"),
    ):
        probability_columns = [
            column
            for column in frame.columns
            if str(column).startswith(f"{prefix}-") and str(column).endswith("-Prob")
        ]
        if not probability_columns:
            continue
        try:
            probabilities = frame[probability_columns].apply(pd.to_numeric, errors="raise")
        except Exception as exc:
            return False, f"{prefix} probability columns contain non-numeric values: {exc}", {}
        values = probabilities.to_numpy(dtype=np.float64, copy=False)
        finite = np.isfinite(values)
        if ((values[finite] < 0.0) | (values[finite] > 1.0)).any():
            return False, f"{prefix} probabilities fall outside [0, 1]", {}
        complete_rows = finite.all(axis=1)
        if complete_rows.any():
            sums = values[complete_rows].sum(axis=1)
            tolerance = max(0.002, len(probability_columns) * 0.00011)
            if not np.allclose(sums, 1.0, atol=tolerance, rtol=0.0):
                return False, f"{prefix} probabilities do not sum to one", {}
        if not complete_rows.any():
            return False, f'{prefix} head has no usable predictions', {}
        partial_rows = finite.any(axis=1) & ~complete_rows
        if partial_rows.any():
            return False, f"{prefix} probability rows are only partially populated", {}
        if label_column not in frame.columns:
            return False, f"inference CSV is missing {label_column!r}", {}
        vocabulary = {
            str(column)[len(prefix) + 1 : -len("-Prob")]
            for column in probability_columns
        }
        labels = frame[label_column]
        populated_labels = labels[complete_rows].dropna().map(str)
        if len(populated_labels) != int(complete_rows.sum()):
            return False, f"{label_column} is blank on a frame with probabilities", {}
        unknown = sorted(set(populated_labels) - vocabulary)
        if unknown:
            return False, f"{label_column} contains labels outside the probability vocabulary: {unknown[:10]}", {}

    return True, "ok", {"rows": expected_rows, "columns": int(frame.shape[1])}


from autobehaver.completion import verified_run


@verified_run
def run_posepred_inference(cfg: PosePredInferenceConfig) -> None:
    """Run selected PosePred checkpoints and validate prediction outputs before registry updates."""

    input_dir_path = Path(cfg.input_dir).expanduser().resolve()
    input_dir = str(input_dir_path)
    if not input_dir_path.is_dir():
        raise FileNotFoundError(f"Input directory not found: {input_dir_path}")

    model_paths = _ensure_list(cfg.inference_model)
    if not model_paths:
        raise ValueError("No inference_model checkpoint(s) provided.")

    # Validate model files
    for p in model_paths:
        if not os.path.isfile(p):
            raise FileNotFoundError(f"Inference checkpoint not found: {p}")

    output_dir: Optional[str] = str(Path(cfg.output_dir).expanduser().resolve()) if cfg.output_dir else None

    # Output layout:
    # - If output_dir is not provided, write under input_dir/inferences.
    # - If output_dir is provided, default to flat output for the stage folder.
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        use_inferences_subdir = (not bool(cfg.flat_output))
        output_root_for_log = output_dir
    else:
        use_inferences_subdir = True
        output_root_for_log = input_dir

    # Resolve input HDF5s
    # Priority:
    # registry (if enabled)
    # filesystem scan of input_dir
    # Restrict registry model updates to the inputs selected for this run.
    used_registry = False
    registry_csv_path: Optional[str] = None
    registry_project_root: Optional[Path] = None

    h5_files: List[str] = []

    if bool(getattr(cfg, "registry_enabled", False)) and getattr(cfg, "registry_csv", None):
        registry_csv_path = str(getattr(cfg, "registry_csv"))
        reg_path = Path(registry_csv_path).expanduser().resolve()
        if not reg_path.exists():
            raise FileNotFoundError(
                f"Registry enabled but CSV not found: {reg_path}. "
                "Run autocropper + dlc + keypre + keyprenorm first (to create registry entries), "
                "or disable registry for this run."
            )

        registry_project_root = reg_path.parent

        try:
            from autobehaver.registry import read_registry, resolve_registry_path
        except Exception as e:
            raise RuntimeError(
                "Failed to import registry helpers (pandas required). "
                "Install pandas in this environment or disable registry."
            ) from e

        df = read_registry(reg_path)
        if df.empty or "keyprenorm_h5_relpath" not in df.columns:
            raise FileNotFoundError(
                "Registry enabled but no keyprenorm_h5_relpath entries were found. "
                "Run the keyprenorm stage first, or disable registry for this run."
            )

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
            return True

        try:
            df2 = df[df["keyprenorm_h5_relpath"].notna()].copy()
        except Exception:
            df2 = df.copy()

        if "active" in df2.columns:
            try:
                df2 = df2[df2["active"].apply(_is_active)]
            except Exception:
                pass

        resolved: List[str] = []
        for _, row in df2.iterrows():
            rel = str(row.get("keyprenorm_h5_relpath") or "").replace("\\", "/")
            if not rel.strip():
                continue
            p = resolve_registry_path(registry_project_root, row, "keyprenorm_h5_relpath")
            if str(p).lower().endswith((".h5", ".hdf5")) and "metadata" not in p.name.lower():
                resolved.append(str(p))

        h5_files = sorted(set(resolved))
        used_registry = True
        logger.info("Using registry for posepred inputs: %s (%d H5 files)", reg_path, len(h5_files))
    else:
        # Filesystem scan
        files: list[str] = []
        for f in os.listdir(input_dir):
            lf = f.lower()
            if (lf.endswith(".h5") or lf.endswith(".hdf5")) and ("metadata" not in lf):
                files.append(str((input_dir_path / f).resolve()))

        h5_files = sorted(set(files))

    # Optional substring filter (bundling without subfolders)
    if getattr(cfg, "file_filter", None):
        needle = str(getattr(cfg, "file_filter")).lower()
        before = len(h5_files)
        h5_files = [p for p in h5_files if needle in original_name(p).lower()]
        h5_files.sort()
        logger.info("Applied file_filter='%s': %d -> %d H5 files", needle, before, len(h5_files))

    from autobehaver.preinf.autocropper.selection import filter_included_paths, propagate_selection
    selection_output_root = Path(output_dir) if output_dir else input_dir_path
    selection_output = selection_output_root / "inferences" if use_inferences_subdir else selection_output_root
    propagate_selection([input_dir_path] + [Path(p).parent for p in h5_files], selection_output)
    h5_files = filter_included_paths(h5_files)
    from autobehaver.batch_inputs import select_batch_inputs
    h5_files = select_batch_inputs(h5_files, getattr(cfg, "file_list", None))
    missing = [p for p in h5_files if not Path(p).is_file()]
    if missing:
        raise FileNotFoundError(f'Selected normalized HDF5 inputs are missing at their declared paths: {missing[:25]}')

    if not h5_files:
        if used_registry:
            raise FileNotFoundError(
                "No normalized HDF5 files discovered from registry. "
                "Check that the registry has keyprenorm_h5_relpath entries (run the keyprenorm stage), "
                "or disable registry for this run."
            )
        raise FileNotFoundError("No HDF5 files to process in input_dir (after filtering).")

    # Capture compatible existing outputs before inference to classify reuse.
    output_root_for_outputs = Path(output_dir).resolve() if output_dir else input_dir_path
    inference_dir = (output_root_for_outputs / "inferences") if use_inferences_subdir else output_root_for_outputs
    preexisting_outputs: set[str] = set()
    preexisting_compatible: set[str] = set()
    from autobehaver.augmentation import TTA_RANDOM_ALGORITHM
    provenance_settings = {
        "inf_average": int(cfg.inf_average),
        "sequence_length_argument": int(cfg.sequence_length),
        "min_pose_bout": int(cfg.min_pose_bout),
        "min_orient_bout": int(cfg.min_orient_bout),
        "use_tta": bool(cfg.use_tta),
        "tta_noise_level": float(cfg.tta_noise_level),
        "tta_seed": int(cfg.tta_seed),
        "tta_random_algorithm": TTA_RANDOM_ALGORITHM,
        "max_inference_cvs": len(model_paths),
    }
    import torch
    from autobehaver.inference import load_inference_checkpoint
    checkpoint = load_inference_checkpoint(model_paths[0])
    variables = checkpoint.get('global_vars', {})
    expected_heads = {head:list(checkpoint.get(key,{})) for head,key,flag in
                      [('Pose','pose_dict','pose_pred'),('Orient','orient_dict','orient_pred')]
                      if variables.get(flag, bool(checkpoint.get(key)))}
    del checkpoint
    checkpoint_identities = [_provenance_file_identity(path) for path in model_paths]
    expected_provenance: dict[str, dict] = {}
    for fp in h5_files:
        stem = Path(fp).stem
        out_path = prediction_path(fp, inference_dir)
        expected = _build_inference_provenance(
            fp,
            model_paths,
            provenance_settings,
            checkpoint_identities=checkpoint_identities,
        )
        expected_provenance[fp] = expected
        if out_path.exists():
            try:
                preexisting_outputs.add(str(out_path.resolve()))
            except Exception:
                preexisting_outputs.add(str(out_path))
            if _completed_inference_matches(str(out_path) + ".provenance.json", expected):
                preexisting_compatible.add(fp)

    # Optional run log.
    if cfg.write_run_log:
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_path = os.path.join(output_root_for_log, f"posepred_inference_run_{ts}.json")
        payload = {
            "timestamp": ts,
            "input_dir": input_dir,
            "output_dir": output_dir,
            "flat_output": bool(cfg.flat_output),
            "use_inferences_subdir": bool(use_inferences_subdir),
            "inference_model": model_paths,
            "file_filter": cfg.file_filter,
            "clobber": bool(cfg.clobber),
            "archive_existing": bool(cfg.archive_existing),
            "num_workers": int(cfg.num_workers),
            "batch": int(cfg.batch),
            "memory_gb": cfg.memory_gb, "noise_workers": cfg.noise_workers, "vram_fraction": cfg.vram_fraction,
            "inf_average": int(cfg.inf_average),
            "use_tta": bool(cfg.use_tta),
            "tta_noise_level": float(cfg.tta_noise_level),
            "tta_seed": int(cfg.tta_seed),
            "tta_random_algorithm": TTA_RANDOM_ALGORITHM,
            "min_pose_bout": int(cfg.min_pose_bout),
            "min_orient_bout": int(cfg.min_orient_bout),
            "sequence_length_fallback": int(cfg.sequence_length),
        }
        try:
            with open(log_path, 'w') as f:
                json.dump(payload, f, indent=2)
        except Exception as e:
            # Never fail the run just because logging failed.
            print(f"[WARN] Could not write run log: {log_path} ({e})")

    validation = StageValidationReport.for_items(
        "posepred_inference",
        h5_files,
        metadata={
            "output_directory": str(inference_dir),
            "model_count": len(model_paths),
        },
    )
    try:
        run_inference(
            directory=input_dir,
            inference_model_paths=model_paths,
            num_workers=int(cfg.num_workers),
            batch=int(cfg.batch),
            memory_gb=cfg.memory_gb, noise_workers=cfg.noise_workers, vram_fraction=cfg.vram_fraction,
            inf_average=int(cfg.inf_average),
            sequence_length=int(cfg.sequence_length),
            min_pose_bout=int(cfg.min_pose_bout),
            min_orient_bout=int(cfg.min_orient_bout),
            use_tta=bool(cfg.use_tta),
            tta_noise_level=float(cfg.tta_noise_level),
            tta_seed=int(cfg.tta_seed),
            output_dir=output_dir,
            file_filter=None,  # already applied above
            clobber=bool(cfg.clobber),
            archive_existing=bool(cfg.archive_existing),
            use_inferences_subdir=bool(use_inferences_subdir),
            explicit_files=list(h5_files),
        )
    except Exception as exc:
        for fp in h5_files:
            validation.add_outcome(fp, "failed", reason=f"inference aborted: {exc}")
        validation.write_run_reports(output_root_for_outputs / "POSEPRED_VALIDATION.json")
        raise

    verified_sources = []
    for fp in h5_files:
        output_csv = prediction_path(fp, inference_dir)
        manifest_path = str(output_csv) + ".provenance.json"
        if not _completed_inference_matches(manifest_path, expected_provenance[fp]):
            validation.add_outcome(
                fp,
                "failed",
                reason="output provenance is missing, incompatible, or does not match published files",
                details={"output": str(output_csv)},
            )
            continue
        ok, reason, details = _validate_inference_csv(fp, output_csv, expected_heads)
        if ok:
            verified_sources.append(fp)
        reused = fp in preexisting_compatible and not cfg.clobber
        validation.add_outcome(
            fp,
            "failed" if not ok else ("reused" if reused else "succeeded"),
            reason=reason if not ok else ("existing output and provenance matched" if reused else None),
            details={"output": str(output_csv), **details},
        )

    # Registry update
    registry_error: str | None = None
    if bool(getattr(cfg, "registry_enabled", False)) and registry_csv_path and registry_project_root:
        try:
            from .registry_update import update_registry_from_posepred_inference_outputs

            update_registry_from_posepred_inference_outputs(
                registry_csv=Path(registry_csv_path),
                project_root=Path(registry_project_root),
                normalized_h5_files=[Path(p) for p in verified_sources],
                output_root=output_root_for_outputs,
                use_inferences_subdir=bool(use_inferences_subdir),
                model_paths=model_paths,
                only_set_models_if_new=False,
                existing_outputs_before=preexisting_outputs,
                lock_timeout_sec=int(getattr(cfg, "registry_lock_timeout_sec", 120)),
                lock_stale_sec=int(getattr(cfg, "registry_lock_stale_sec", 60 * 60)),
            )
        except Exception as exc:
            registry_error = str(exc)
            logger.exception("Registry update (posepred_inference) failed")

    if registry_error is not None:
        validation.add_issue(
            "registry_update_failed",
            "error",
            "Inference outputs were created, but the project registry could not be updated.",
            details={"error": registry_error},
        )
    validation.write_run_reports(output_root_for_outputs / "POSEPRED_VALIDATION.json")
    validation.raise_for_errors()
