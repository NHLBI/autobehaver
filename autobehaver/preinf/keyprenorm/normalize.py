"""Apply feature normalization and publish frame-aligned HDF5 outputs."""

from __future__ import annotations

import logging
import os
import json
import shutil
from autobehaver.portable_paths import original_name, normalized_path, generated_path
from autobehaver.completion import preserve_previous, verified_run, write_receipt, file_digest, file_signature
import multiprocessing as mp
from typing import Dict, Optional, Sequence

import h5py
import numpy as np
import pandas as pd

from . import __stage_version__
from .logging_utils import setup_worker_logging
from .identity import normalization_identity, label_identity
from .csv_chunks import RetainedCsvChunks, HDF_COMPRESSION_LEVEL
from .inspect_outputs import inspect_label_h5, inspect_normalized_h5
from ..frame_identity import consecutive_frames, h5_frame_timeline, retained_frame_indices, write_h5_frame_timeline

logger = logging.getLogger("keyprenorm")

# Normalization worker

_NORM_CANONICAL_COLUMNS: Optional[list[str]] = None
_NORM_MEANS: Optional[np.ndarray] = None
_NORM_STDS: Optional[np.ndarray] = None
_NORM_EDGE_SKIP: int = 0
_NORM_CHUNK_ROWS: int = 4096
_NORM_OUTPUT_DIR: Optional[str] = None
_NORM_LABELS_INPUT_DIR: Optional[str] = None
_NORM_LABELS_OUTPUT_DIR: Optional[str] = None


def init_normalize_worker(
    canonical_columns: Sequence[str],
    means: np.ndarray,
    stds: np.ndarray,
    edge_skip: int,
    output_dir: str,
    labels_input_dir: Optional[str],
    labels_output_dir: Optional[str],
    log_queue: mp.Queue,
    log_level: str,
    chunk_rows: int = 4096,
) -> None:
    """Initialize canonical columns, float32 parameters, output paths, and worker logging."""
    global _NORM_CANONICAL_COLUMNS, _NORM_MEANS, _NORM_STDS
    global _NORM_EDGE_SKIP, _NORM_CHUNK_ROWS
    global _NORM_OUTPUT_DIR, _NORM_LABELS_INPUT_DIR, _NORM_LABELS_OUTPUT_DIR

    _NORM_CANONICAL_COLUMNS = list(canonical_columns)
    _NORM_MEANS = means.astype(np.float32, copy=False)
    _NORM_STDS = stds.astype(np.float32, copy=False)
    _NORM_EDGE_SKIP = int(edge_skip)
    _NORM_CHUNK_ROWS = int(chunk_rows)
    _NORM_OUTPUT_DIR = output_dir
    _NORM_LABELS_INPUT_DIR = labels_input_dir
    _NORM_LABELS_OUTPUT_DIR = labels_output_dir

    setup_worker_logging(log_queue, log_level)


def _rebind_existing_labels(label_path, old_features, source_csv, identity, source_rows, source_indices, temporary):
    """Retain annotations when only normalization changed, with complete pair proof.

    A changed positional source cannot be identified biologically from its basename
    or frame count. In that case the original annotation CSV must be restored.
    """
    message = ("Cannot safely rebind existing annotations without the source label CSV; "
               "restore the original annotation CSV and rerun normalization")
    try:
        with h5py.File(old_features, 'r') as hf:
            old_identity = hf.attrs['content_identity']
            old_metadata = json.loads(old_identity)
            n_cols = int(hf['data'].shape[1])
            old_rows, old_indices, complete = h5_frame_timeline(hf)
        if (old_metadata.get('source') != file_digest(source_csv)
                or not complete or old_rows != source_rows
                or not np.array_equal(old_indices, source_indices)):
            raise ValueError('positional source or retained frame identity changed')
        valid_features, _, reason = inspect_normalized_h5(old_features, n_cols,
            expected_identity=old_identity, expected_source_rows=source_rows)
        if not valid_features:
            raise ValueError(f'previous features failed integrity validation: {reason}')
        valid_labels, reason = inspect_label_h5(label_path, len(source_indices),
            expected_feature_identity=old_identity, expected_source_rows=source_rows,
            expected_edge_skip=int(old_metadata['edge_skip']))
        if not valid_labels:
            raise ValueError(f'previous annotations failed integrity validation: {reason}')
        with h5py.File(label_path, 'r') as hf:
            binding = json.loads(hf.attrs['content_identity'])
        binding['feature_identity'] = identity
        rebound_identity = json.dumps(binding, sort_keys=True, separators=(',', ':'))
        shutil.copyfile(label_path, temporary)
        with h5py.File(temporary, 'r+') as hf:
            hf.attrs['content_identity'] = rebound_identity
        return rebound_identity
    except (OSError, KeyError, TypeError, ValueError) as exc:
        raise ValueError(f'{message}: {exc}.') from exc


@verified_run
def normalize_one_csv(csv_file: str) -> Dict[str, object]:
    """Normalize retained rows as (x - mean)/std and publish paired HDF5 outputs.

    Use float32; set nonpositive-deviation columns and nonfinite normalized values
    to zero. Preserve original zero-based frame indices after trimming both edges.
    Validate paired label alignment before publishing features.
    """
    temporary_paths = []
    try:
        if _NORM_CANONICAL_COLUMNS is None or _NORM_MEANS is None or _NORM_STDS is None or _NORM_OUTPUT_DIR is None:
            raise RuntimeError("Normalize worker not initialized.")

        cols = _NORM_CANONICAL_COLUMNS

        means = _NORM_MEANS
        stds = _NORM_STDS
        nonzero = stds > 0

        base_name = original_name(csv_file)
        out_path = str(normalized_path(csv_file, _NORM_OUTPUT_DIR))
        tmp_path = out_path + ".tmp"
        temporary_paths.append(tmp_path)
        identity = normalization_identity(csv_file, cols, means, stds,
            _NORM_EDGE_SKIP)

        chunks = RetainedCsvChunks(csv_file, cols, _NORM_EDGE_SKIP, chunk_rows=_NORM_CHUNK_ROWS)
        normalized_rows = 0
        with h5py.File(tmp_path, "w") as hf:
            dataset = hf.create_dataset("data", shape=(0, len(cols)),
                maxshape=(None, len(cols)), dtype=np.float32,
                chunks=(min(256, chunks.chunk_rows), len(cols)),
                compression="gzip", compression_opts=HDF_COMPRESSION_LEVEL)
            for df in chunks:
                x = df.to_numpy(dtype=np.float32, copy=False)
                x_norm = np.empty_like(x, dtype=np.float32)
                np.subtract(x, means, out=x_norm)
                np.divide(x_norm, stds, out=x_norm, where=nonzero)
                if np.any(~nonzero):
                    x_norm[:, ~nonzero] = 0.0
                np.nan_to_num(x_norm, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
                stop = normalized_rows + len(x_norm)
                dataset.resize(stop, axis=0)
                dataset[normalized_rows:stop] = x_norm
                normalized_rows = stop
            source_rows = chunks.source_rows
            source_indices = retained_frame_indices(source_rows, _NORM_EDGE_SKIP)
            if normalized_rows != len(source_indices):
                raise ValueError("Normalized rows do not match the retained source timeline.")
            hf.attrs["columns"] = [c.encode("utf-8") for c in cols]
            hf.attrs["keyprenorm_version"] = __stage_version__.encode("utf-8")
            hf.attrs['content_identity'] = identity
            hf.attrs['source_positional_sha256'] = file_digest(csv_file)
            hf.attrs['source_positional_path'] = os.path.abspath(csv_file)
            hf.attrs['normalization_edge_skip'] = _NORM_EDGE_SKIP
            write_h5_frame_timeline(hf, frame_count=source_rows, indices=source_indices)

        # Validate the complete label timeline before publishing either paired output.
        labels_saved = False
        labels_rebound = False
        label_df = None
        label_out_path = None
        tmp_label_path = None
        if _NORM_LABELS_INPUT_DIR and _NORM_LABELS_OUTPUT_DIR:
            label_name = base_name.replace("_positional_calcs.csv", "") + "_labels.csv"
            label_path = os.path.join(_NORM_LABELS_INPUT_DIR, label_name)
            label_out_path = str(generated_path(_NORM_LABELS_OUTPUT_DIR, label_name.replace('.csv','.h5')))
            if os.path.exists(label_path):
                full_labels = pd.read_csv(label_path, dtype={"Behavior": str, "Orientation": str})
                if 'Frame' not in full_labels:
                    raise ValueError('Label CSV requires an explicit Frame column.')
                frames = consecutive_frames(full_labels['Frame'], source=label_path, require_origin=True)
                if len(full_labels) != source_rows:
                    raise ValueError(f'Label/source timeline length mismatch: {len(full_labels)} != {source_rows}')
                label_df = full_labels.iloc[source_indices].copy()
                label_df['Frame'] = frames[source_indices]
                if len(label_df) != normalized_rows:
                    raise ValueError('Trimmed label/features length mismatch.')
                label_receipt = label_identity(label_path, identity)
            elif os.path.exists(label_out_path):
                tmp_label_path = label_out_path + '.tmp'
                temporary_paths.append(tmp_label_path)
                label_receipt = _rebind_existing_labels(label_out_path, out_path, csv_file,
                    identity, source_rows, source_indices, tmp_label_path)
                labels_rebound = True
                logger.info('Preserved annotations and rebound normalization identity: %s', label_out_path)

        if label_df is not None:
            tmp_label_path = label_out_path + '.tmp'
            temporary_paths.append(tmp_label_path)
            with h5py.File(tmp_label_path, 'w') as hf:
                for col in label_df:
                    series = label_df[col]
                    if pd.api.types.is_string_dtype(series.dtype) or series.dtype == 'object':
                        hf.create_dataset(col, data=series.astype(str).to_numpy(dtype=object),
                            dtype=h5py.string_dtype(encoding='utf-8'), compression='gzip', compression_opts=HDF_COMPRESSION_LEVEL)
                    else:
                        dtype = np.int64 if pd.api.types.is_integer_dtype(series.dtype) else np.float32
                        hf.create_dataset(col, data=series.to_numpy(dtype=dtype), compression='gzip', compression_opts=HDF_COMPRESSION_LEVEL)
                hf.attrs['columns'] = [c.encode('utf-8') for c in label_df.columns]
                hf.attrs['keyprenorm_version'] = __stage_version__.encode('utf-8')
                hf.attrs['content_identity'] = label_receipt
                hf.attrs['normalization_edge_skip'] = _NORM_EDGE_SKIP
                write_h5_frame_timeline(hf, frame_count=source_rows, indices=source_indices)
        if tmp_label_path is not None:
            # Feature output is the final commit. A failure between these renames
            # leaves a mismatched pair whose identities cannot pass reuse checks.
            preserve_previous(label_out_path)
            os.replace(tmp_label_path, label_out_path)
            write_receipt(label_out_path, label_receipt, source_rows=source_rows,
                          edge_skip=_NORM_EDGE_SKIP, rows=normalized_rows)
            labels_saved = True
        if normalization_identity(csv_file, cols, means, stds, _NORM_EDGE_SKIP) != identity:
            raise ValueError("Positional source changed during normalization; outputs are not published as complete.")
        preserve_previous(out_path)
        os.replace(tmp_path, out_path)
        write_receipt(out_path, identity, source_rows=source_rows,
                      edge_skip=_NORM_EDGE_SKIP, rows=normalized_rows,
                      source_positional_sha256=file_digest(csv_file))

        ok, checked_rows, reason = inspect_normalized_h5(out_path, len(cols),
            expected_rows=normalized_rows, expected_columns=cols, expected_identity=identity,
            expected_source_rows=source_rows, expected_edge_skip=_NORM_EDGE_SKIP)
        if not ok:
            raise ValueError('Normalized output failed integrity validation: ' + reason)
        if labels_saved:
            ok, reason = inspect_label_h5(label_out_path, expected_len=normalized_rows,
                expected_identity=label_receipt, expected_feature_identity=identity,
                expected_source_rows=source_rows, expected_edge_skip=_NORM_EDGE_SKIP)
            if not ok:
                raise ValueError('Normalized label output failed integrity validation: ' + reason)
        verification = dict(rows=checked_rows, source_rows=source_rows, identity=identity,
            source_signature=file_signature(csv_file), output_signature=file_signature(out_path),
            output_digest=file_digest(out_path))

        return {
            "status": "success",
            "file": csv_file,
            "rows": normalized_rows,
            "source_rows": source_rows,
            "verification": verification,
            "labels_saved": labels_saved,
            "labels_rebound": labels_rebound,
            "error": None,
        }

    except Exception as e:
        logger.exception("Error processing %s", csv_file)
        return {"status": "failed", "file": csv_file, "labels_saved": False, "error": str(e)}
    finally:
        for path in temporary_paths:
            if os.path.exists(path):
                os.unlink(path)
