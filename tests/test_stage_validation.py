from __future__ import annotations

import json

import pytest

from autobehaver.validation import StageValidationError, StageValidationReport


def test_exact_inventory_passes_with_success_reuse_and_exclusion(tmp_path) -> None:
    report = StageValidationReport.for_items("demo", ["a", "b", "c"])
    report.add_outcome("a", "succeeded")
    report.add_outcome("b", "reused", reason="matching provenance")
    report.add_outcome("c", "excluded", reason="recording metadata marks chamber empty")

    assert report.status == "passed"
    path = report.write(tmp_path / "validation.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["expected_count"] == 3
    assert payload["outcome_counts"]["excluded"] == 1


def test_missing_or_failed_items_make_stage_fail() -> None:
    report = StageValidationReport.for_items("demo", ["a", "b"])
    report.add_outcome("a", "failed", reason="broken input")

    with pytest.raises(StageValidationError, match="demo"):
        report.raise_for_errors()

    codes = {issue.code for issue in report.issues}
    assert codes == {"failed_items", "missing_item_outcome"}


def test_warning_does_not_change_item_accounting_to_failure() -> None:
    report = StageValidationReport.for_items("demo", ["a"])
    report.add_outcome("a", "succeeded")
    report.add_issue("qc_signal", "warning", "A non-fatal QC threshold was exceeded")

    report.raise_for_errors()
    assert report.status == "warning"


def test_duplicate_expected_items_are_rejected() -> None:
    report = StageValidationReport.for_items("demo", ["a", "a"])
    report.add_outcome("a", "succeeded")

    with pytest.raises(StageValidationError, match="duplicate"):
        report.raise_for_errors()
