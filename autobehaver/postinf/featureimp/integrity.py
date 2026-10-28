"""Content-bound completion receipts for scientific FeatureImp stages."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping

from autobehaver.atomic_io import sha256_file
from autobehaver.completion import code_digest


def strict_bool(value: Any) -> bool:
    """Parse an explicit boolean; never treat the string 'False' as true."""
    if isinstance(value, bool):
        return value
    if value in (0, 1) and not isinstance(value, str):
        return bool(value)
    if isinstance(value, str) and value.strip().lower() in {"true", "false", "1", "0"}:
        return value.strip().lower() in {"true", "1"}
    raise ValueError(f"Invalid boolean flag: {value!r}")


def flatten_paths(value: Any) -> list[str]:
    """Yield artifact paths from nested mappings and sequences."""
    if isinstance(value, Mapping):
        return [p for item in value.values() for p in flatten_paths(item)]
    if isinstance(value, (list, tuple)):
        return [p for item in value for p in flatten_paths(item)]
    return [str(value)] if value else []


def safe_artifact(base: str | Path, relative: str) -> Path:
    """Require a declared artifact to exist inside its analysis directory."""
    base = Path(base).resolve()
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"Unsafe artifact path: {relative!r}")
    target = (base / path).resolve()
    if not target.is_relative_to(base) or not target.is_file():
        raise ValueError(f"Missing scientific artifact: {relative!r}")
    return target


def stage_inputs(base: str | Path, marker_name: str) -> dict[str, str]:
    """Bind frozen inputs and upstream artifacts, excluding own stage outputs."""
    base = Path(base).resolve()
    paths = [p for p in (base / "_featureimp_plan").rglob("*") if p.is_file()]
    plan_path = base / '_featureimp_plan' / 'PLAN_MANIFEST.json'
    if plan_path.is_file():
        manifest = json.loads(plan_path.read_text(encoding='utf-8'))
        paths.extend(safe_artifact(base, p) for p in flatten_paths(manifest.get('artifacts', {})))
    for name in ("FEATUREIMP_RUN_SETTINGS.json", "NORMALIZATION_PARAMS.json", "NORMALIZED_FEATURE_MATRIX.csv"):
        if (base / name).is_file():
            paths.append(base / name)
    if marker_name != "SHARD_DONE.json":
        for marker in sorted((base / "_featureimp_shards").glob("*/SHARD_DONE.json")):
            paths.append(marker)
            data = json.loads(marker.read_text(encoding="utf-8"))
            paths.extend(safe_artifact(base, p) for p in flatten_paths(data.get("artifacts", {})))
    if marker_name not in {"SHARD_DONE.json", "WEIGHTS_DONE.json"}:
        marker = base / "_featureimp_merge" / "WEIGHTS_DONE.json"
        if not marker.is_file():
            raise ValueError("Weighted stage requires a completed weights receipt")
        paths.append(marker)
        data = json.loads(marker.read_text(encoding="utf-8"))
        paths.extend(safe_artifact(base, p) for p in flatten_paths(data.get("artifacts", {})))
    # Each file is hashed once per validation, with bounded I/O concurrency.
    from concurrent.futures import ThreadPoolExecutor
    from autobehaver.preinf.execution import cpu_limit
    paths = sorted(set(paths))
    with ThreadPoolExecutor(max_workers=min(4, cpu_limit())) as pool:
        result = dict(zip((p.relative_to(base).as_posix() for p in paths), pool.map(sha256_file, paths)))
    source = Path(__file__).parent
    result["__featureimp_source__"] = code_digest(source.glob('*.py'))
    # Runtime parallelism is operational; the persisted full snapshot remains
    # fingerprinted above, while a changed in-memory worker count does not alter it.
    if not any("PLAN_MANIFEST" in p for p in result):
        raise ValueError("Completion requires a frozen FeatureImp plan")
    return result


def seal_receipt(payload: Mapping[str, Any], destination: str, *, verified_inputs=None) -> dict[str, Any]:
    """Bind stage outputs and upstream identities into a completion receipt."""
    out = dict(payload)
    base = out.get("analysis_folder")
    if not base:
        raise ValueError("Completion receipt requires analysis_folder")
    artifacts = flatten_paths(out.get("artifacts", {}))
    if not artifacts:
        raise ValueError("Completion receipt has no required scientific artifacts")
    out["receipt_version"] = 2
    out["input_sha256"] = dict(verified_inputs) if verified_inputs is not None else stage_inputs(base, Path(destination).name)
    out["artifact_sha256"] = {p: sha256_file(safe_artifact(base, p)) for p in artifacts}
    return out


def validate_receipt(destination: str, base: str, *, verified_inputs=None) -> dict[str, Any]:
    """Verify a stage receipt against its current artifacts and upstream identities."""
    out = json.loads(Path(destination).read_text(encoding="utf-8"))
    if out.get("status") != "completed" or out.get("receipt_version") != 2:
        raise ValueError("Unverified or legacy completion marker; use a fresh analysis folder")
    expected = verified_inputs if verified_inputs is not None else stage_inputs(base, Path(destination).name)
    if out.get("input_sha256") != expected:
        raise ValueError("Scientific inputs/source/dependencies changed after completion")
    paths = flatten_paths(out.get("artifacts", {}))
    if not paths or set(paths) != set(out.get("artifact_sha256", {})):
        raise ValueError("Completion marker has incomplete artifact fingerprints")
    for path in paths:
        if sha256_file(safe_artifact(base, path)) != out["artifact_sha256"][path]:
            raise ValueError(f"Scientific output changed after completion: {path}")
    return out


def baseline_inputs(base: str | Path, *, include_shards: bool) -> dict[str, str]:
    """Bind baseline calculations to their frozen repeat/fold plan and inputs."""
    base = Path(base).resolve()
    result = stage_inputs(base, "SHARD_DONE.json")
    plan = base / "PLAIN_XGBOOST_BASELINE" / "_STAGED_PLAN"
    manifest = plan / "PLAIN_XGBOOST_STAGED_PLAN_MANIFEST.json"
    if not manifest.is_file():
        raise ValueError("Baseline completion requires its frozen staged plan")
    paths = [p for p in plan.rglob("*") if p.is_file() and p.name != "PLAIN_XGBOOST_STAGED_MERGE_DONE.json"]
    if include_shards:
        for marker in sorted((base / "PLAIN_XGBOOST_BASELINE" / "_STAGED_SHARDS").glob("*/SHARD_DONE.json")):
            paths.append(marker)
            payload = json.loads(marker.read_text(encoding="utf-8"))
            paths.extend(safe_artifact(base, p) for p in flatten_paths(payload.get("artifacts", {})))
    result.update({p.relative_to(base).as_posix(): sha256_file(p) for p in paths})
    return result


def seal_baseline_receipt(payload: Mapping[str, Any], destination: str) -> dict[str, Any]:
    """Bind baseline outputs to the frozen fold plan and input identities."""
    out = dict(payload)
    base = out.get("analysis_folder")
    if not base:
        raise ValueError("Baseline completion receipt requires analysis_folder")
    paths = flatten_paths(out.get("artifacts", {}))
    if not paths:
        raise ValueError("Baseline completion receipt requires its scientific artifacts")
    out["receipt_version"] = 2
    out["input_sha256"] = baseline_inputs(base, include_shards=Path(destination).name != "SHARD_DONE.json")
    out["artifact_sha256"] = {p: sha256_file(safe_artifact(base, p)) for p in paths}
    return out


def validate_baseline_receipt(destination: str, base: str) -> dict[str, Any]:
    """Verify baseline artifacts and frozen inputs against their completion receipt."""
    out = json.loads(Path(destination).read_text(encoding="utf-8"))
    if out.get("status") != "completed" or out.get("receipt_version") != 2:
        raise ValueError("Unverified baseline completion; use fresh baseline shards")
    if out.get("input_sha256") != baseline_inputs(base, include_shards=Path(destination).name != "SHARD_DONE.json"):
        raise ValueError("Baseline inputs/source/plan changed after completion")
    paths = flatten_paths(out.get("artifacts", {}))
    if not paths or set(paths) != set(out.get("artifact_sha256", {})):
        raise ValueError("Baseline completion lacks required artifact fingerprints")
    for path in paths:
        if sha256_file(safe_artifact(base, path)) != out["artifact_sha256"][path]:
            raise ValueError(f"Baseline scientific output changed: {path}")
    return out
