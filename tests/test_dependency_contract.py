from __future__ import annotations

import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _read(relative: str) -> str:
    return (REPO_ROOT / relative).read_text(encoding="utf-8")


def _required_version(text: str, distribution: str) -> str:
    match = re.search(rf"(?m)^{re.escape(distribution)}==([^\s;]+)", text)
    assert match, f"{distribution} must have one exact version pin"
    return match.group(1)


def test_local_scientific_dependency_contract():
    conda = _read("environment/environment.yml")
    shared = _read("environment/requirements-core.txt")
    local = _read("environment/requirements-local-core.txt")

    assert "python=3.12" in conda
    assert "-r requirements-core.txt" in local

    _required_version(local, "torch")
    _required_version(local, "opencv-python")

    # Shared direct dependencies are exact pins (environment markers allowed).
    unpinned = [
        line
        for line in shared.splitlines()
        if line.strip() and not line.lstrip().startswith("#") and "==" not in line
    ]
    assert unpinned == []
