"""Resolve ordered feature schemas for PosePred inputs and checkpoint identity."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import logging
import re
from pathlib import Path
from typing import Any, Iterable, Sequence

__all__ = [
    "DEFAULT_PRESET",
    "PRESETS",
    "FeaturePlan",
    "compute_schema_hash",
    "parse_csv_list_arg",
    "read_columns_list_file",
    "resolve_feature_plan",
]


# Presets

NO_POS_EXCLUDE_COLUMNS: list[str] = [
    # Exclude absolute position and angle features in the no_pos preset.
    "centroid_x",
    "centroid_y",
    "angle_sin",
    "angle_cos",
    "radial_position",
    "radial_angle_sin",
    "radial_angle_cos",
    "relative_angle_sin",
    "relative_angle_cos",
]


PRESETS: dict[str, dict[str, list[str]]] = {
    "none": {
        "exclude_columns": [],
        "exclude_keywords": [],
    },
    # Exclude absolute position / angle information.
    "no_pos": {
        "exclude_columns": NO_POS_EXCLUDE_COLUMNS,
        "exclude_keywords": [],
    },
}


DEFAULT_PRESET: str = "no_pos"


# Data structures


@dataclass(frozen=True)
class FeaturePlan:
    """Store canonical, included and excluded feature names in order.

    ``schema_hash`` identifies the ordered model inputs; ``spec`` records the
    selection rules used to obtain them.
    """

    canonical_columns: list[str]
    included_columns: list[str]
    excluded_columns: list[str]
    schema_hash: str
    spec: dict[str, Any]

    def to_json_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return {
            "canonical_columns": list(self.canonical_columns),
            "included_columns": list(self.included_columns),
            "excluded_columns": list(self.excluded_columns),
            "schema_hash": str(self.schema_hash),
            "spec": dict(self.spec),
        }

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_json_dict(), indent=indent, sort_keys=True)


# Helpers


def compute_schema_hash(included_columns: Sequence[str]) -> str:
    """Compute a stable hash for an ordered included-column list."""
    # Feature order contributes to checkpoint identity.
    payload = "\n".join(str(c) for c in included_columns).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def read_columns_list_file(path: str | Path) -> list[str]:
    """Read a newline-delimited column list file.

    Lines beginning with '#' are treated as comments. Empty lines are ignored.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Column-list file not found: {p}")

    cols: list[str] = []
    with p.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            cols.append(line)
    return cols


def _flatten_items(value: Any) -> list[str]:
    """Normalize an argparse-style string/list value into a flat list of strings."""
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple)):
        out: list[str] = []
        for v in value:
            if v is None:
                continue
            if isinstance(v, str):
                out.append(v)
            else:
                out.append(str(v))
        return out
    return [str(value)]


def parse_csv_list_arg(value: str | Sequence[str] | None) -> list[str]:
    """Flatten comma- or whitespace-separated strings and lists into items."""
    flat = _flatten_items(value)
    items: list[str] = []
    for entry in flat:
        # Allow both comma-separated and whitespace-separated, but comma is primary.
        for part in entry.split(","):
            tok = part.strip()
            if tok:
                items.append(tok)
    return items


def _normalize_for_match(s: str, *, case_sensitive: bool) -> str:
    return s if case_sensitive else s.lower()


def _dedupe_preserve_order(items: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for x in items:
        if x in seen:
            continue
        seen.add(x)
        out.append(x)
    return out


def _validate_headers(headers: Sequence[str]) -> None:
    if not headers:
        raise ValueError("Empty headers list; cannot resolve feature plan.")
    # Duplicate header names create ambiguous mappings; disallow.
    seen: set[str] = set()
    dupes: set[str] = set()
    for h in headers:
        if h in seen:
            dupes.add(h)
        seen.add(h)
    if dupes:
        raise ValueError(
            "Duplicate feature names found in canonical headers; cannot resolve feature plan. "
            f"Duplicates: {sorted(dupes)[:20]}" + (" ..." if len(dupes) > 20 else "")
        )


def resolve_feature_plan(
    headers: Sequence[str],
    *,
    preset: str | None = None,
    exclude_columns: str | Sequence[str] | None = None,
    exclude_keywords: str | Sequence[str] | None = None,
    include_columns_file: str | Path | None = None,
    case_sensitive: bool = False,
) -> FeaturePlan:
    """Resolve an ordered input schema and its hash from canonical headers.

    An include file defines the order directly and bypasses exclusion rules.
    Otherwise apply preset, exact-name and substring exclusions in canonical
    order. Use ``case_sensitive`` for matching and reject an empty schema.
    """
    canonical_columns = [str(h) for h in headers]
    _validate_headers(canonical_columns)

    chosen_preset = preset or DEFAULT_PRESET
    if chosen_preset not in PRESETS:
        raise ValueError(
            f"Unknown ablation preset '{chosen_preset}'. Available: {sorted(PRESETS.keys())}"
        )

    # Explicit ordered schema.
    if include_columns_file is not None:
        included = read_columns_list_file(include_columns_file)
        if not included:
            raise ValueError(f"Included-columns file produced an empty list: {include_columns_file}")

        included = _dedupe_preserve_order([str(c) for c in included])
        missing = [c for c in included if c not in canonical_columns]
        if missing:
            raise ValueError(
                "Included-columns file references columns not present in canonical headers. "
                f"Missing: {missing[:20]}" + (" ..." if len(missing) > 20 else "")
            )

        excluded = [c for c in canonical_columns if c not in set(included)]
        schema_hash = compute_schema_hash(included)
        spec = {
            "mode": "include_file",
            "include_columns_file": str(include_columns_file),
            "preset": chosen_preset,
            "case_sensitive": bool(case_sensitive),
        }
        return FeaturePlan(
            canonical_columns=canonical_columns,
            included_columns=included,
            excluded_columns=excluded,
            schema_hash=schema_hash,
            spec=spec,
        )

    # Canonical-order exclusions.
    preset_spec = PRESETS[chosen_preset]
    preset_ex_cols = list(preset_spec.get("exclude_columns", []))
    preset_ex_kw = list(preset_spec.get("exclude_keywords", []))

    ex_cols = _dedupe_preserve_order(
        [str(x) for x in preset_ex_cols] + [str(x) for x in parse_csv_list_arg(exclude_columns)]
    )
    ex_kw = _dedupe_preserve_order(
        [str(x) for x in preset_ex_kw] + [str(x) for x in parse_csv_list_arg(exclude_keywords)]
    )

    canon_norm = [_normalize_for_match(c, case_sensitive=case_sensitive) for c in canonical_columns]
    ex_cols_norm = {_normalize_for_match(c, case_sensitive=case_sensitive) for c in ex_cols}
    ex_kw_norm = [_normalize_for_match(k, case_sensitive=case_sensitive) for k in ex_kw]
    if (preset_ex_cols or preset_ex_kw) and all(re.fullmatch(r'col_\d+', c) for c in canonical_columns):
        raise ValueError(f'Semantic ablation preset {chosen_preset!r} requires meaningful feature headers. '
                         'Supply the original schema or explicitly select preset none.')
    unmatched_columns = [str(c) for c in parse_csv_list_arg(exclude_columns)
                         if _normalize_for_match(str(c), case_sensitive=case_sensitive) not in canon_norm]
    unmatched_keywords = [str(k) for k in parse_csv_list_arg(exclude_keywords)
                          if not any(_normalize_for_match(str(k), case_sensitive=case_sensitive) in c for c in canon_norm)]
    if unmatched_columns or unmatched_keywords:
        logging.warning('Feature exclusions matched no columns: names=%s, keywords=%s', unmatched_columns, unmatched_keywords)

    included: list[str] = []
    excluded: list[str] = []
    for orig, norm in zip(canonical_columns, canon_norm):
        drop = False
        if norm in ex_cols_norm:
            drop = True
        if not drop and ex_kw_norm:
            for kw in ex_kw_norm:
                if kw and kw in norm:
                    drop = True
                    break
        if drop:
            excluded.append(orig)
        else:
            included.append(orig)

    if not included:
        raise ValueError(
            "Feature plan would include zero columns after applying ablation rules. "
            "Check exclude_columns/exclude_keywords." 
        )

    schema_hash = compute_schema_hash(included)
    spec = {
        "mode": "exclude_rules",
        "preset": chosen_preset,
        "exclude_columns": ex_cols,
        "exclude_keywords": ex_kw,
        "unmatched_exclude_columns": unmatched_columns,
        "unmatched_exclude_keywords": unmatched_keywords,
        "case_sensitive": bool(case_sensitive),
    }
    return FeaturePlan(
        canonical_columns=canonical_columns,
        included_columns=included,
        excluded_columns=excluded,
        schema_hash=schema_hash,
        spec=spec,
    )
