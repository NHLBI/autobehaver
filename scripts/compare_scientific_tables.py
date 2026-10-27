#!/usr/bin/env python3
"""Compare CSV, Excel and HDF5 values, schemas and ordering.

Numeric values use the selected tolerances and matching NaNs compare equal.
Excel sheets are compared individually; HDF5 file/dataset attributes are ignored."""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
import pandas as pd


def _compare_frames(expected: pd.DataFrame, actual: pd.DataFrame, *, rtol: float, atol: float) -> list[str]:
    """Return schema/value differences; matching NaNs compare equal."""
    errors: list[str] = []
    if expected.shape != actual.shape:
        return [f"shape differs: expected {expected.shape}, actual {actual.shape}"]
    if list(expected.columns) != list(actual.columns):
        return ["column names or ordering differ"]

    numeric = expected.select_dtypes(include=np.number).columns
    actual_numeric = actual.select_dtypes(include=np.number).columns
    if list(numeric) != list(actual_numeric):
        errors.append("numeric column types differ")
    else:
        left = expected[numeric].to_numpy(dtype=float)
        right = actual[numeric].to_numpy(dtype=float)
        if not np.allclose(left, right, rtol=rtol, atol=atol, equal_nan=True):
            finite = np.isfinite(left) & np.isfinite(right)
            max_diff = float(np.max(np.abs(left[finite] - right[finite]))) if finite.any() else float("nan")
            errors.append(f"numeric values differ (max absolute difference {max_diff:g})")

    numeric_set = set(numeric)
    for column in expected.columns:
        if column in numeric_set:
            continue
        left = expected[column].fillna("<NA>").astype(str)
        right = actual[column].fillna("<NA>").astype(str)
        if not left.equals(right):
            errors.append(f"non-numeric values differ in column {column!r}")
    return errors


def _hdf_datasets(handle: h5py.File) -> dict[str, np.ndarray]:
    """Read datasets recursively, excluding file and dataset attributes."""
    datasets: dict[str, np.ndarray] = {}

    def collect(name: str, obj: object) -> None:
        if isinstance(obj, h5py.Dataset):
            datasets[name] = obj[...]

    handle.visititems(collect)
    return datasets


def compare(expected_path: Path, actual_path: Path, *, rtol: float, atol: float) -> list[str]:
    """Return content differences for matching CSV, Excel or HDF5 file types."""
    suffix = expected_path.suffix.lower()
    if suffix != actual_path.suffix.lower():
        return ["file extensions differ"]
    if suffix == ".csv":
        return _compare_frames(pd.read_csv(expected_path), pd.read_csv(actual_path), rtol=rtol, atol=atol)
    if suffix in {".xlsx", ".xls"}:
        expected_book = pd.ExcelFile(expected_path)
        actual_book = pd.ExcelFile(actual_path)
        if expected_book.sheet_names != actual_book.sheet_names:
            return ["workbook sheet names or ordering differ"]
        errors: list[str] = []
        for sheet in expected_book.sheet_names:
            sheet_errors = _compare_frames(
                pd.read_excel(expected_path, sheet_name=sheet),
                pd.read_excel(actual_path, sheet_name=sheet),
                rtol=rtol,
                atol=atol,
            )
            errors.extend(f"sheet {sheet!r}: {error}" for error in sheet_errors)
        return errors
    if suffix in {".h5", ".hdf5"}:
        with h5py.File(expected_path, "r") as expected_file, h5py.File(actual_path, "r") as actual_file:
            expected = _hdf_datasets(expected_file)
            actual = _hdf_datasets(actual_file)
        if list(expected) != list(actual):
            return ["HDF5 dataset names or ordering differ"]
        errors = []
        for name, left in expected.items():
            right = actual[name]
            if left.shape != right.shape or left.dtype != right.dtype:
                errors.append(f"dataset {name!r} shape or dtype differs")
            elif left.dtype.kind in "biufc":
                if not np.allclose(left, right, rtol=rtol, atol=atol, equal_nan=True):
                    errors.append(f"dataset {name!r} numeric values differ")
            elif not np.array_equal(left, right):
                errors.append(f"dataset {name!r} values differ")
        return errors
    return [f"unsupported file type: {suffix}"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("expected", type=Path)
    parser.add_argument("actual", type=Path)
    parser.add_argument("--rtol", type=float, default=1e-6)
    parser.add_argument("--atol", type=float, default=1e-7)
    args = parser.parse_args()

    errors = compare(args.expected, args.actual, rtol=args.rtol, atol=args.atol)
    if errors:
        print("FAIL")
        for error in errors:
            print(f"- {error}")
        return 1
    print("PASS: scientific contents match")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
