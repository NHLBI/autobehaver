"""CLI for the posepred inference pipeline stage.

Preferred usage (pipeline config):
    python -m autobehaver.preinf.posepred_inference --config /path/to/config.yaml

The global config file should contain a top-level section:

posepred_inference:
  input_dir: 5_normalized_features
  output_dir: 6_posepred_inference
  inference_model:
    - /path/to/checkpoint.pth

CLI args always override config defaults.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import yaml

from autobehaver.argparse_utils import add_bool_pair
from autobehaver.pipeline_layout import apply_main_dir_overrides, config_main_dir_default

from .config import PosePredInferenceConfig
from .core import run_posepred_inference


def _load_config_defaults(config_path: str) -> Dict[str, Any]:
    """Load the posepred_inference section from a global YAML config file."""
    cfg_path = Path(config_path).expanduser().resolve()
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")

    with cfg_path.open("r") as f:
        data = yaml.safe_load(f) or {}

    if not isinstance(data, dict):
        raise ValueError("Pipeline config must be a YAML mapping (dict) at the top level")

    section = data.get("posepred_inference", {})

    if not isinstance(section, dict):
        raise ValueError("Expected 'posepred_inference' section in config to be a mapping (dict)")

    base_dir = cfg_path.parent

    # ------------------------------------------------------------------
    # Global registry settings (project-wide)
    # ------------------------------------------------------------------
    resolved: Dict[str, Any] = {}

    global_main_dir = config_main_dir_default(data, base_dir)
    if global_main_dir:
        resolved["main_dir"] = global_main_dir

    reg = data.get("registry", {})
    if reg is None:
        reg = {}
    if isinstance(reg, dict):
        if "enabled" in reg:
            resolved["registry_enabled"] = bool(reg.get("enabled"))
        if "csv" in reg and isinstance(reg.get("csv"), str) and str(reg.get("csv")).strip() != "":
            p = Path(str(reg.get("csv")).strip())
            if not p.is_absolute():
                p = (base_dir / p).resolve()
            resolved["registry_csv"] = str(p)
        if "lock_timeout_sec" in reg and reg.get("lock_timeout_sec") is not None:
            try:
                resolved["registry_lock_timeout_sec"] = int(reg.get("lock_timeout_sec"))
            except Exception:
                pass
        if "lock_stale_sec" in reg and reg.get("lock_stale_sec") is not None:
            try:
                resolved["registry_lock_stale_sec"] = int(reg.get("lock_stale_sec"))
            except Exception:
                pass

    def _resolve_path(v: str) -> str:
        p = Path(str(v).strip())
        if not p.is_absolute():
            p = (base_dir / p).resolve()
        return str(p)

    # Direct path keys
    for k in ("main_dir", "input_dir", "output_dir"):
        v = section.get(k)
        if isinstance(v, str) and v.strip():
            resolved[k] = _resolve_path(v)

    # Stage-level override for registry_csv (optional)
    v = section.get("registry_csv")
    if isinstance(v, str) and v.strip():
        resolved["registry_csv"] = _resolve_path(v)

    # Model list keys
    models = section.get("inference_model", None)
    if models is None:
        models = section.get("models", None)
    if models is not None:
        if isinstance(models, (list, tuple)):
            resolved["inference_model"] = [_resolve_path(m) for m in models]
        elif isinstance(models, str) and models.strip():
            resolved["inference_model"] = [_resolve_path(models)]

    # Pass-through simple keys
    passthrough = {
        # Registry
        "registry_enabled",
        "registry_lock_timeout_sec",
        "registry_lock_stale_sec",
        "num_workers",
        "batch",
        "inf_average",
        "use_tta",
        "tta_noise_level",
        "min_pose_bout",
        "min_orient_bout",
        "sequence_length",
        "file_filter",
        "clobber",
        "archive_existing",
        "flat_output",
        "write_run_log",
    }
    for k in passthrough:
        if k in section:
            resolved[k] = section[k]

    return resolved


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run PosePred transformer inference on normalized feature HDF5 files (pipeline stage).",
    )

    p.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to global pipeline YAML config (reads the 'posepred_inference:' section).",
    )

    # Project registry (optional)
    add_bool_pair(
        p,
        "registry_enabled",
        default=False,
        help="Enable project registry integration (discover inputs + upsert outputs).",
        dest="registry_enabled",
    )
    p.add_argument(
        "--registry_csv",
        type=str,
        default=None,
        help="Path to project registry CSV (usually at project root).",
    )
    p.add_argument(
        "--registry_lock_timeout_sec",
        type=int,
        default=120,
        help="Timeout (seconds) waiting for the registry directory lock.",
    )
    p.add_argument(
        "--registry_lock_stale_sec",
        type=int,
        default=60 * 60,
        help="Consider the registry lock stale after this many seconds.",
    )


    p.add_argument(
        "--main_dir",
        type=str,
        default=None,
        help=(
            "Optional project root containing the standardized numbered pipeline folders. "
            "For posepred_inference this fills input_dir=<main_dir>/5_normalized_features and output_dir=<main_dir>/6_posepred_inference."
        ),
    )

    p.add_argument(
        "--input_dir",
        dest="input_dir",
        type=str,
        default=None,
        help="Directory containing normalized feature HDF5 files (e.g., 5_normalized_features).",
    )

    p.add_argument(
        "--output_dir",
        dest="output_dir",
        type=str,
        default=None,
        help="Directory to write inference outputs (e.g., 6_posepred_inference).",
    )

    p.add_argument(
        "--inference_model",
        nargs="+",
        default=None,
        help="Path(s) to model checkpoint(s) (.pth). Provide multiple for ensembling.",
    )

    # Selection / incremental processing
    p.add_argument(
        "--file_filter",
        type=str,
        default=None,
        help="Only process HDF5 files whose filename contains this substring (case-insensitive).",
    )

    add_bool_pair(
        p,
        "clobber",
        default=False,
        help="Overwrite existing inference outputs (default: disabled / safe).",
        dest="clobber",
    )

    add_bool_pair(
        p,
        "archive_existing",
        default=False,
        help="Archive existing inference CSV outputs before running.",
        dest="archive_existing",
    )

    add_bool_pair(
        p,
        "flat_output",
        default=True,
        help=(
            "Write inference CSVs directly into output_dir (and details into output_dir/inference_details). "
            "If disabled, write outputs under output_dir/inferences/."
        ),
        dest="flat_output",
    )

    add_bool_pair(
        p,
        "write_run_log",
        default=True,
        help="Write a small JSON run log to the output directory.",
        dest="write_run_log",
    )

    # Inference parameters
    p.add_argument("--num_workers", type=int, default=14, help="PyTorch DataLoader workers (default: 14).")
    p.add_argument("--batch", type=int, default=1024, help="Batch size (default: 1024).")
    p.add_argument("--inf_average", type=int, default=10, help="Forward-pass averaging count (default: 10).")

    add_bool_pair(
        p,
        "use_tta",
        default=False,
        help="Enable test-time augmentation (noise injection).",
        dest="use_tta",
    )
    p.add_argument("--tta_noise_level", type=float, default=0.01, help="TTA noise scale (default: 0.01).")

    p.add_argument("--min_pose_bout", type=int, default=5, help="Pose smoothing min bout length (default: 5).")
    p.add_argument("--min_orient_bout", type=int, default=5, help="Orient smoothing min bout length (default: 5).")

    p.add_argument(
        "--sequence_length",
        type=int,
        default=41,
        help="Fallback sequence length if checkpoint does not specify it (default: 41).",
    )

    return p


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = build_arg_parser()
    argv_list = list(argv) if argv is not None else sys.argv[1:]

    # Two-pass parse so config defaults apply cleanly.
    pre = parser.parse_known_args(argv_list)[0]

    if pre.config:
        defaults = _load_config_defaults(pre.config)
        parser.set_defaults(**defaults)

    args = parser.parse_args(argv_list)
    apply_main_dir_overrides(args, stage="posepred_inference")

    if not args.input_dir:
        raise SystemExit(
            "Missing --input_dir. Provide --input_dir explicitly, or use --config with a 'posepred_inference:' section that sets input_dir."
        )

    if not args.inference_model:
        raise SystemExit(
            "Missing --inference_model. Provide checkpoint path(s) explicitly, or set inference_model in the 'posepred_inference:' config section."
        )

    clobber = bool(getattr(args, "clobber", False))

    cfg = PosePredInferenceConfig(
        input_dir=str(args.input_dir),
        output_dir=str(args.output_dir) if args.output_dir else None,
        inference_model=[str(x) for x in (args.inference_model or [])],
        num_workers=int(args.num_workers),
        batch=int(args.batch),
        inf_average=int(args.inf_average),
        use_tta=bool(args.use_tta),
        tta_noise_level=float(args.tta_noise_level),
        min_pose_bout=int(args.min_pose_bout),
        min_orient_bout=int(args.min_orient_bout),
        sequence_length=int(args.sequence_length),
        file_filter=str(args.file_filter) if args.file_filter else None,
        clobber=clobber,
        archive_existing=bool(args.archive_existing),
        flat_output=bool(args.flat_output),
        write_run_log=bool(args.write_run_log),
        registry_enabled=bool(getattr(args, "registry_enabled", False)),
        registry_csv=str(args.registry_csv) if getattr(args, "registry_csv", None) else None,
        registry_lock_timeout_sec=int(getattr(args, "registry_lock_timeout_sec", 120)),
        registry_lock_stale_sec=int(getattr(args, "registry_lock_stale_sec", 60 * 60)),
    )

    run_posepred_inference(cfg)


if __name__ == "__main__":
    main()
