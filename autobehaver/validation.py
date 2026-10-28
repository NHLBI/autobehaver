"""Record stage integrity outcomes separately from quality-control warnings.

Require an outcome for every requested item without changing scientific data.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping
from uuid import uuid4

from .atomic_io import atomic_write_json
from .resource_monitor import record_validation


VALIDATION_SCHEMA = "autobehaver.stage_validation.v1"
TERMINAL_ITEM_STATUSES = frozenset({"succeeded", "reused", "excluded", "failed"})


class StageValidationError(RuntimeError):
    """Report a failed stage integrity contract."""

    def __init__(self, report: "StageValidationReport") -> None:
        self.report = report
        errors = [issue.message for issue in report.issues if issue.severity == "error"]
        preview = "; ".join(errors[:3]) or "stage validation failed"
        if len(errors) > 3:
            preview += f"; and {len(errors) - 3} more error(s)"
        super().__init__(f"{report.stage}: {preview}")


@dataclass(frozen=True)
class ValidationIssue:
    """Store a validation observation with a stable code and severity."""

    code: str
    severity: str
    message: str
    item_id: str | None = None
    details: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.severity not in {"info", "warning", "error"}:
            raise ValueError(f"Unknown validation severity: {self.severity!r}")
        if not str(self.code).strip():
            raise ValueError("Validation issue code must not be blank")
        if not str(self.message).strip():
            raise ValueError("Validation issue message must not be blank")

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": str(self.code),
            "severity": str(self.severity),
            "message": str(self.message),
            "item_id": self.item_id,
            "details": dict(self.details),
        }


@dataclass
class StageValidationReport:
    """Account for every requested item and retain quality-control findings."""

    stage: str
    expected_items: tuple[str, ...]
    outcomes: dict[str, dict[str, Any]] = field(default_factory=dict)
    issues: list[ValidationIssue] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def for_items(
        cls,
        stage: str,
        expected_items: Iterable[str],
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> "StageValidationReport":
        normalized = tuple(str(item) for item in expected_items)
        report = cls(
            stage=str(stage),
            expected_items=normalized,
            metadata=dict(metadata or {}),
        )
        duplicates = sorted(item for item, count in Counter(normalized).items() if count > 1)
        if duplicates:
            report.add_issue(
                "duplicate_expected_item",
                "error",
                f"The requested input inventory contains {len(duplicates)} duplicate item(s).",
                details={"examples": duplicates[:25]},
            )
        return report

    def add_outcome(
        self,
        item_id: str,
        status: str,
        *,
        reason: str | None = None,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        item = str(item_id)
        normalized_status = str(status).strip().lower()
        if normalized_status not in TERMINAL_ITEM_STATUSES:
            raise ValueError(f"Unknown stage item status: {status!r}")
        if item in self.outcomes:
            self.add_issue(
                "duplicate_item_outcome",
                "error",
                f"Item {item!r} received more than one terminal outcome.",
                item_id=item,
            )
            return
        self.outcomes[item] = {
            "status": normalized_status,
            "reason": None if reason is None else str(reason),
            "details": dict(details or {}),
        }

    def add_issue(
        self,
        code: str,
        severity: str,
        message: str,
        *,
        item_id: str | None = None,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        self.issues.append(
            ValidationIssue(
                code=str(code),
                severity=str(severity),
                message=str(message),
                item_id=None if item_id is None else str(item_id),
                details=dict(details or {}),
            )
        )

    def finalize(self) -> "StageValidationReport":
        # These issues describe the current accounting state, not a permanent
        # event. An earlier progress read must not poison the finished report.
        derived_codes = {"missing_item_outcome", "unexpected_item_outcome", "failed_items"}
        self.issues[:] = [issue for issue in self.issues if issue.code not in derived_codes]
        expected = set(self.expected_items)
        observed = set(self.outcomes)

        missing = sorted(expected - observed)
        if missing and not any(issue.code == "missing_item_outcome" for issue in self.issues):
            self.add_issue(
                "missing_item_outcome",
                "error",
                f"{len(missing)} expected item(s) have no terminal outcome.",
                details={"examples": missing[:25]},
            )

        unexpected = sorted(observed - expected)
        if unexpected and not any(issue.code == "unexpected_item_outcome" for issue in self.issues):
            self.add_issue(
                "unexpected_item_outcome",
                "error",
                f"{len(unexpected)} outcome(s) do not belong to the requested inventory.",
                details={"examples": unexpected[:25]},
            )

        failed = sorted(
            item for item, outcome in self.outcomes.items()
            if outcome.get("status") == "failed"
        )
        if failed and not any(issue.code == "failed_items" for issue in self.issues):
            self.add_issue(
                "failed_items",
                "error",
                f"{len(failed)} expected item(s) failed processing or output validation.",
                details={"examples": failed[:25]},
            )
        return self

    @property
    def status(self) -> str:
        self.finalize()
        severities = {issue.severity for issue in self.issues}
        if "error" in severities:
            return "failed"
        if "warning" in severities:
            return "warning"
        return "passed"

    def outcome_counts(self) -> dict[str, int]:
        counts = {status: 0 for status in sorted(TERMINAL_ITEM_STATUSES)}
        for outcome in self.outcomes.values():
            status = str(outcome.get("status", ""))
            if status in counts:
                counts[status] += 1
        return counts

    def to_dict(self) -> dict[str, Any]:
        self.finalize()
        return {
            "schema": VALIDATION_SCHEMA,
            "stage": self.stage,
            "status": self.status,
            "validated_at_utc": datetime.now(timezone.utc).isoformat(),
            "expected_count": len(self.expected_items),
            "outcome_counts": self.outcome_counts(),
            "outcomes": dict(sorted(self.outcomes.items())),
            "issues": [issue.to_dict() for issue in self.issues],
            "metadata": dict(self.metadata),
        }

    def write(self, path: str | Path) -> Path:
        destination = Path(path)
        atomic_write_json(self.to_dict(), destination)
        record_validation(self)
        return destination

    def write_run_reports(self, latest_path: str | Path) -> Path:
        """Retain this batch's outcome and update the conventional latest report."""
        latest = Path(latest_path)
        destination = latest.with_name(f"{latest.stem}.{uuid4().hex}{latest.suffix}")
        self.metadata["run_report"] = destination.name
        payload = self.to_dict()
        atomic_write_json(payload, destination)
        atomic_write_json(payload, latest)
        record_validation(self)
        return destination

    def raise_for_errors(self) -> None:
        self.finalize()
        if self.status == "failed":
            raise StageValidationError(self)


__all__ = [
    "StageValidationError",
    "StageValidationReport",
    "TERMINAL_ITEM_STATUSES",
    "VALIDATION_SCHEMA",
    "ValidationIssue",
]
