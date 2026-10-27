"""Load, verify, and align MMGen inference and positional inputs.

Use original-frame indices or declared legacy offsets and retain explicit opt-in
compatibility for heuristic alignment.
"""

from __future__ import annotations

import os
import json
from pathlib import Path

import pandas as pd

from .io import find_matching_poscalc_file
from autobehaver.atomic_io import sha256_file


def prediction_provenance(inference_path):
    """Read native/legacy provenance if supplied; malformed metadata is an error."""
    path = Path(str(inference_path) + '.provenance.json')
    if not path.exists():
        return None
    try:
        info = json.loads(path.read_text(encoding='utf-8'))
    except (ValueError, OSError) as exc:
        raise ValueError(f'Cannot read prediction provenance: {path}') from exc
    if not isinstance(info, dict):
        raise ValueError(f'Prediction provenance is not an object: {path}')
    verify_prediction_output(inference_path, info)
    return info


def verify_prediction_output(inference_path, info):
    """Verify the selected CSV when modern prediction/smoothing output is declared.

    Offset-only historical sidecars remain supported. Native output records
    certify the bytes actually consumed rather than just a neighboring file.
    """
    native_schema = info.get('schema') == 'autobehaver.posepred_inference.provenance.v1'
    resmoothed = info.get('operation') == 'corrected_label_smoothing'
    if not native_schema and not resmoothed and 'outputs' not in info:
        return
    if info.get('status') != 'complete':
        raise ValueError(f'Prediction output has no completed native record: {inference_path}')
    entries = [info.get('output')] if resmoothed else info.get('outputs')
    if not isinstance(entries, list) or not entries:
        raise ValueError(f'Prediction provenance has no declared outputs: {inference_path}')
    source = Path(inference_path).resolve()
    matches = []
    for entry in entries:
        if not isinstance(entry, dict) or not isinstance(entry.get('path'), str):
            raise ValueError(f'Malformed declared prediction output: {inference_path}')
        if (source.parent / entry['path']).resolve() == source:
            matches.append(entry)
    if len(matches) != 1:
        raise ValueError(f'Prediction provenance must declare this CSV exactly once: {inference_path}')
    expected = matches[0].get('sha256')
    if (not isinstance(expected, str) or len(expected) != 64
            or any(c not in '0123456789abcdef' for c in expected.lower())
            or sha256_file(source) != expected.lower()):
        raise ValueError(f'Prediction CSV does not match its completed native output: {inference_path}')


def verify_prediction_positional_source(inference_path, positional_path, info=None):
    """Bind new native predictions to their actual upstream positional bytes.

    Historical/imported predictions without this field remain supported, but
    their unavailable upstream identity is not invented from current files.
    """
    info = prediction_provenance(inference_path) if info is None else info
    if not info:
        return 'legacy_unverified'
    first = info.get('source_positional_sha256')
    nested = info.get('positional_source') or {}
    if not isinstance(nested, dict):
        raise ValueError(f'Malformed positional source in {inference_path}')
    second = nested.get('sha256')
    if first and second and first != second:
        raise ValueError(f'Conflicting positional source fingerprints in {inference_path}')
    expected = first or second
    if expected:
        if not isinstance(expected, str) or len(expected) != 64:
            raise ValueError(f'Invalid positional source fingerprint in {inference_path}')
        if sha256_file(positional_path) != expected:
            raise ValueError(f'Predictions and positional measurements belong to different processing results: {inference_path}; {positional_path}')
        return 'verified'
    return 'legacy_unverified'


def effective_source_frame_offset(info, *, explicit_indices, requested=0):
    """Resolve saved or requested frame offsets without shifting original-frame indices
    twice.
    """
    if explicit_indices:
        if requested:
            raise ValueError('Predictions already use original movie frames; a manual source offset would shift them twice.')
        return 0
    saved = None if info is None else info.get('source_frame_offset', (info.get('settings') or {}).get('source_frame_offset'))
    if saved is not None:
        if isinstance(saved, bool) or int(saved) != saved or saved < 0:
            raise ValueError('Saved original-frame offset must be a nonnegative integer.')
        if requested and int(requested) != int(saved):
            raise ValueError('Manual frame offset disagrees with saved original-frame provenance.')
        return int(saved)
    return requested


def _align_poscalc_to_inference(
    df: pd.DataFrame,
    poscalc_df: pd.DataFrame,
    *,
    edge_skip: int,
    source_name: str,
    allow_unsafe_fallback: bool = False,
    source_frame_offset: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Align positional rows to consecutive inference frame IDs.

    Use zero-based Source Frame Index or one-based Frame Number plus the declared source
    offset. Legacy inputs without frame IDs can use edge_skip; truncation or prefix
    fallback requires explicit opt-in.
    """

    if 'Source Frame Index' in df:
        if source_frame_offset:
            raise ValueError('Predictions already use original movie frames; a manual source offset would shift them twice.')
        original = pd.to_numeric(df['Source Frame Index'], errors='coerce')
        if not original.equals(pd.to_numeric(df['Frame Number'], errors='coerce') - 1):
            raise ValueError('Original-frame index disagrees with Frame Number.')
    if int(source_frame_offset) != source_frame_offset or source_frame_offset < 0:
        raise ValueError("source_frame_offset must be a nonnegative integer")
    if source_frame_offset and "Frame Number" not in df.columns:
        raise ValueError("source_frame_offset requires an explicit Frame Number column")
    if "Frame Number" in df.columns:
        frame_numbers = pd.to_numeric(df["Frame Number"], errors="coerce")
        integer_frames = frame_numbers.notna().all() and bool((frame_numbers % 1 == 0).all())
        if integer_frames:
            idx = frame_numbers.astype(int).to_numpy() - 1 + int(source_frame_offset)
            strictly_increasing = len(idx) < 2 or bool((idx[1:] - idx[:-1] == 1).all())
            if (
                len(idx) == len(df)
                and len(idx) > 0
                and strictly_increasing
                and bool((frame_numbers >= 1).all())
                and int(idx.min()) >= 0
                and int(idx.max()) < len(poscalc_df)
            ):
                return df.reset_index(drop=True), poscalc_df.iloc[idx].reset_index(drop=True)
        if source_frame_offset or not allow_unsafe_fallback:
            raise ValueError(
                f"Invalid Frame Number alignment for {source_name}: values must be unique, "
                "consecutive positive integers within the positional-calcs timeline "
                f"after applying source_frame_offset={source_frame_offset}."
            )

    start = max(int(edge_skip or 0), 0)
    stop = start + len(df)
    aligned = poscalc_df.iloc[start:stop].reset_index(drop=True)
    if len(aligned) == len(df):
        return df.reset_index(drop=True), aligned

    if not allow_unsafe_fallback:
        raise ValueError(
            f"Inference/positional-calcs alignment mismatch for {source_name}: "
            f"inference_rows={len(df)}, positional_rows={len(poscalc_df)}, edge_skip={start}. "
            "No rows were truncated or shifted. Correct the inputs, or explicitly enable "
            "allow_unsafe_alignment_fallback to reproduce the historical heuristic."
        )

    if len(poscalc_df) >= len(df):
        print(
            f"WARNING: Edge-skip alignment produced {len(aligned)} rows for {source_name}; "
            f"falling back to first {len(df)} positional_calcs rows."
        )
        return df.reset_index(drop=True), poscalc_df.iloc[: len(df)].reset_index(drop=True)

    n = min(len(df), len(poscalc_df))
    print(
        f"WARNING: Inference/poscalc row-count mismatch for {source_name}; "
        f"truncating to common prefix length {n} (inference={len(df)}, poscalc={len(poscalc_df)})."
    )
    return df.iloc[:n].reset_index(drop=True), poscalc_df.iloc[:n].reset_index(drop=True)


def load_inference_and_poscalc(
    inf_csv_path: str,
    poscalc_dir: str,
    *,
    edge_skip: int,
    inf_tag: str,
    allow_unsafe_alignment_fallback: bool = False,
    source_frame_offset: int = 0,
):
    """Load inference CSV and matching positional_calcs CSV and align them.

    Returns (df, poscalc_df, aligned_poscalc_df) on success.
    Returns (None, None, None) if the matching positional_calcs file is missing.
    """

    df = pd.read_csv(inf_csv_path)
    info = prediction_provenance(inf_csv_path)
    source_frame_offset = effective_source_frame_offset(info, explicit_indices='Source Frame Index' in df, requested=source_frame_offset)

    poscalc_csv_path = find_matching_poscalc_file(
        inf_csv_path, poscalc_dir, inf_tag
    )
    if poscalc_csv_path is None:
        print(
            f"WARNING: No matching positional_calcs file found for {os.path.basename(inf_csv_path)}. "
            "Skipping this recording."
        )
        return None, None, None

    verify_prediction_positional_source(inf_csv_path, poscalc_csv_path, info)
    poscalc_df = pd.read_csv(poscalc_csv_path)
    df_aligned, aligned_poscalc_df = _align_poscalc_to_inference(
        df,
        poscalc_df,
        edge_skip=edge_skip,
        source_name=os.path.basename(inf_csv_path),
        allow_unsafe_fallback=allow_unsafe_alignment_fallback,
        source_frame_offset=source_frame_offset,
    )

    return df_aligned, poscalc_df, aligned_poscalc_df


def extract_smoothed_labels(df: pd.DataFrame):
    """Extract smoothed pose/orientation labels from an inference dataframe."""

    masked_poses = df["Smoothed Pose Label"].to_numpy()

    has_orient = "Smoothed Orient Label" in df.columns
    masked_orients = None
    if has_orient:
        masked_orients = df["Smoothed Orient Label"].to_numpy()

    return masked_poses, masked_orients, has_orient
