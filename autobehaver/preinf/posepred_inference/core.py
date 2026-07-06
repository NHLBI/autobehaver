"""Core runner for the posepred inference pipeline stage."""

from __future__ import annotations

import json
import os
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, List

from .config import PosePredInferenceConfig

from ...inference import run_inference

logger = logging.getLogger(__name__)


def _ensure_list(x):
    if x is None:
        return []
    if isinstance(x, (list, tuple)):
        return [str(v) for v in x]
    return [str(x)]


def run_posepred_inference(cfg: PosePredInferenceConfig) -> None:
    """Run posepred inference from a prepared config."""

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

    # ----------------------------------------------------------------------------------
    # Resolve input HDF5s
    # Priority:
    #   1) registry (if enabled)
    #   2) filesystem scan of input_dir
    #
    # We pre-filter here so we can safely update the registry without overwriting
    # posepred_models for files that were not part of the selected run.
    # ----------------------------------------------------------------------------------
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
            from autobehaver.registry import read_registry  # lazy import; pandas required
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

        # pandas is optional; used only for isna convenience.
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

        rels = [
            str(x).replace("\\", "/")
            for x in df2["keyprenorm_h5_relpath"].tolist()
            if str(x).strip() != ""
        ]

        resolved: List[str] = []
        for rel in rels:
            p = (registry_project_root / rel).resolve()
            if not p.exists():
                bn = Path(rel).name
                cand = input_dir_path / bn
                if cand.exists():
                    p = cand.resolve()
            if not p.exists():
                continue
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
        h5_files = [p for p in h5_files if needle in os.path.basename(p).lower()]
        h5_files.sort()
        logger.info("Applied file_filter='%s': %d -> %d H5 files", needle, before, len(h5_files))

    if not h5_files:
        if used_registry:
            raise FileNotFoundError(
                "No normalized HDF5 files discovered from registry. "
                "Check that the registry has keyprenorm_h5_relpath entries (run the keyprenorm stage), "
                "or disable registry for this run."
            )
        print("No HDF5 files to process in input_dir (after filtering).")
        return

    # Determine expected output locations *before* running inference (for safe registry updates).
    output_root_for_outputs = Path(output_dir).resolve() if output_dir else input_dir_path
    inference_dir = (output_root_for_outputs / "inferences") if use_inferences_subdir else output_root_for_outputs
    preexisting_outputs: set[str] = set()
    for fp in h5_files:
        stem = Path(fp).stem
        out_path = inference_dir / f"{stem}_inference.csv"
        if out_path.exists():
            try:
                preexisting_outputs.add(str(out_path.resolve()))
            except Exception:
                preexisting_outputs.add(str(out_path))

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
            "inf_average": int(cfg.inf_average),
            "use_tta": bool(cfg.use_tta),
            "tta_noise_level": float(cfg.tta_noise_level),
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

    run_inference(
        directory=input_dir,
        inference_model_paths=model_paths,
        num_workers=int(cfg.num_workers),
        batch=int(cfg.batch),
        inf_average=int(cfg.inf_average),
        sequence_length=int(cfg.sequence_length),
        min_pose_bout=int(cfg.min_pose_bout),
        min_orient_bout=int(cfg.min_orient_bout),
        use_tta=bool(cfg.use_tta),
        tta_noise_level=float(cfg.tta_noise_level),
        output_dir=output_dir,
        file_filter=None,  # already applied above
        clobber=bool(cfg.clobber),
        archive_existing=bool(cfg.archive_existing),
        use_inferences_subdir=bool(use_inferences_subdir),
        explicit_files=list(h5_files),
    )

    # -------------------------
    # Registry update (best-effort)
    # -------------------------
    if bool(getattr(cfg, "registry_enabled", False)) and registry_csv_path and registry_project_root:
        try:
            from .registry_update import update_registry_from_posepred_inference_outputs

            update_registry_from_posepred_inference_outputs(
                registry_csv=Path(registry_csv_path),
                project_root=Path(registry_project_root),
                normalized_h5_files=[Path(p) for p in h5_files],
                output_root=output_root_for_outputs,
                use_inferences_subdir=bool(use_inferences_subdir),
                model_paths=model_paths,
                only_set_models_if_new=not bool(cfg.clobber),
                existing_outputs_before=preexisting_outputs,
                lock_timeout_sec=int(getattr(cfg, "registry_lock_timeout_sec", 120)),
                lock_stale_sec=int(getattr(cfg, "registry_lock_stale_sec", 60 * 60)),
            )
        except Exception:
            logger.exception("Registry update (posepred_inference) failed")
