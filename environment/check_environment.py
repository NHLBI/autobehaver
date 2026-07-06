#!/usr/bin/env python3
"""Check the active Autobehaver Python environment.

Run from the repository root:
    python environment/check_environment.py

For the optional DeepLabCut/TensorFlow environment:
    python environment/check_environment.py --dlc-only --require-dlc
"""

from __future__ import annotations

import argparse
import importlib
import subprocess
import sys
from pathlib import Path
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def check_import(label: str, module_name: str, *, optional: bool = False) -> tuple[bool, object | None]:
    try:
        module = importlib.import_module(module_name)
        version = getattr(module, "__version__", "version unknown")
        print(f"OK       {label:<16} {version}")
        return True, module
    except Exception as exc:
        tag = "OPTIONAL" if optional else "FAIL"
        print(f"{tag:<8} {label:<16} {type(exc).__name__}: {exc}")
        return optional, None


def check_help(command: list[str]) -> bool:
    try:
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print("OK      ", " ".join(command))
        return True
    except subprocess.CalledProcessError as exc:
        print("FAIL    ", " ".join(command), f"exit={exc.returncode}")
        return False


def _check_core_imports() -> tuple[bool, dict[str, object | None]]:
    print("\nCore imports:")
    core_modules = [
        ("numpy", "numpy"),
        ("pandas", "pandas"),
        ("scipy", "scipy"),
        ("sklearn", "sklearn"),
        ("statsmodels", "statsmodels"),
        ("matplotlib", "matplotlib"),
        ("cv2", "cv2"),
        ("h5py", "h5py"),
        ("yaml", "yaml"),
        ("torch", "torch"),
        ("xgboost", "xgboost"),
        ("shap", "shap"),
        ("gseapy", "gseapy"),
        ("autobehaver", "autobehaver"),
    ]
    core_ok = True
    imported: dict[str, object | None] = {}
    for label, module_name in core_modules:
        ok, module = check_import(label, module_name)
        core_ok = core_ok and ok
        imported[module_name] = module
    return core_ok, imported


def _check_dlc_imports(*, require_dlc: bool) -> tuple[bool, bool, dict[str, object | None]]:
    print("\nDLC/TensorFlow imports:")
    optional = not require_dlc
    tf_ok, tf_mod = check_import("tensorflow", "tensorflow", optional=optional)
    dlc_ok, dlc_mod = check_import("deeplabcut", "deeplabcut", optional=optional)
    available = tf_mod is not None and dlc_mod is not None

    if tf_mod is not None:
        try:
            gpus = tf_mod.config.list_physical_devices("GPU")
            print(f"TensorFlow visible GPUs: {len(gpus)}")
        except Exception as exc:
            print(f"TensorFlow GPU check failed: {type(exc).__name__}: {exc}")

    return bool(tf_ok and dlc_ok), available, {"tensorflow": tf_mod, "deeplabcut": dlc_mod}


def _check_torch(imported: dict[str, object | None]) -> None:
    torch_mod = imported.get("torch")
    if torch_mod is None:
        return
    print("\nPyTorch device check:")
    try:
        cuda_available = bool(torch_mod.cuda.is_available())
        print(f"torch.cuda.is_available(): {cuda_available}")
        print(f"torch.version.cuda: {getattr(torch_mod.version, 'cuda', None)}")
        print(f"CUDA device count: {torch_mod.cuda.device_count()}")
        if cuda_available:
            for idx in range(torch_mod.cuda.device_count()):
                print(f"  GPU {idx}: {torch_mod.cuda.get_device_name(idx)}")
    except Exception as exc:
        print(f"PyTorch CUDA check failed: {type(exc).__name__}: {exc}")


def _run_help_checks(commands: Iterable[list[str]]) -> list[bool]:
    print("\nCLI help checks:")
    return [check_help(command) for command in commands]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Check the active Autobehaver environment.")
    parser.add_argument(
        "--require-dlc",
        action="store_true",
        help="Fail if TensorFlow or DeepLabCut cannot be imported.",
    )
    parser.add_argument(
        "--dlc-only",
        action="store_true",
        help="Only check the Autobehaver DLC wrapper plus TensorFlow/DeepLabCut imports.",
    )
    parser.add_argument(
        "--skip-cli-help",
        action="store_true",
        help="Skip module --help smoke checks.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    print(f"Python: {sys.version.split()[0]}")
    print(f"Executable: {sys.executable}")
    print(f"Repo root: {REPO_ROOT}")

    overall_ok = True

    if args.dlc_only:
        print("\nDLC wrapper imports:")
        wrapper_ok, _ = check_import("autobehaver", "autobehaver")
        dlc_wrapper_ok, _ = check_import("autobehaver.preinf.dlc", "autobehaver.preinf.dlc")
        overall_ok = overall_ok and wrapper_ok and dlc_wrapper_ok
        dlc_ok, dlc_available, _ = _check_dlc_imports(require_dlc=args.require_dlc)
        overall_ok = overall_ok and dlc_ok
        if not args.skip_cli_help:
            help_results = _run_help_checks([[sys.executable, "-m", "autobehaver.preinf.dlc", "--help"]])
            overall_ok = overall_ok and all(help_results)
        else:
            help_results = []

        print("\nSummary:")
        print(f"DLC wrapper: {'OK' if wrapper_ok and dlc_wrapper_ok else 'FAILED'}")
        dlc_status = "OK" if dlc_available else ("FAILED" if args.require_dlc else "optional missing")
        print(f"DLC imports: {dlc_status}")
        if help_results:
            print(f"CLI helps:   {sum(help_results)}/{len(help_results)} OK")
        return 0 if overall_ok else 1

    core_ok, imported = _check_core_imports()
    overall_ok = overall_ok and core_ok
    _check_torch(imported)
    dlc_ok, dlc_available, _ = _check_dlc_imports(require_dlc=args.require_dlc)
    overall_ok = overall_ok and dlc_ok

    if not args.skip_cli_help:
        commands = [
            [sys.executable, "-m", "autobehaver", "--version"],
            [sys.executable, "-m", "autobehaver.preinf.moviechecker", "--help"],
            [sys.executable, "-m", "autobehaver.preinf.autocropper", "--help"],
            [sys.executable, "-m", "autobehaver.preinf.dlc", "--help"],
            [sys.executable, "-m", "autobehaver.preinf.keypre", "--help"],
            [sys.executable, "-m", "autobehaver.preinf.keyprenorm", "--help"],
            [sys.executable, "-m", "autobehaver.preinf.posepred_inference", "--help"],
            [sys.executable, "-m", "autobehaver.training", "--help"],
            [sys.executable, "-m", "autobehaver.training.posepred_training", "--help"],
            [sys.executable, "-m", "autobehaver.postinf.mmgen", "--help"],
            [sys.executable, "-m", "autobehaver.postinf.featureimp", "--help"],
            [sys.executable, "-m", "autobehaver.postinf.infmovie", "--help"],
        ]
        help_results = _run_help_checks(commands)
        overall_ok = overall_ok and all(help_results)
    else:
        help_results = []

    print("\nSummary:")
    print(f"Core imports: {'OK' if core_ok else 'FAILED'}")
    dlc_status = "OK" if dlc_available else ("FAILED" if args.require_dlc else "optional missing")
    print(f"DLC imports:  {dlc_status}")
    if help_results:
        print(f"CLI helps:    {sum(help_results)}/{len(help_results)} OK")

    return 0 if overall_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
