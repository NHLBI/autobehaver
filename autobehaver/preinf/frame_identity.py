"""Validate original-frame identities for preprocessing and inference."""
from __future__ import annotations
import numpy as np
import pandas as pd
import json


def retained_frame_indices(frame_count, edge_skip):
    """Return original zero-based indices after trimming both preprocessing edges."""
    frame_count, edge_skip = int(frame_count), int(edge_skip)
    if frame_count < 0 or edge_skip < 0:
        raise ValueError("Frame count and edge skip must be nonnegative.")
    return np.arange(edge_skip, max(edge_skip, frame_count - edge_skip), dtype=np.int64)


def h5_frame_timeline(hf):
    """Read original frame identity from stored metadata without decoding footage.

    Legacy content receipts recover their known leading offset. Their unknown
    omitted tail is recorded as incomplete coverage. Plain feature H5 inputs
    without normalization markers are consecutive untrimmed feature timelines.
    """
    rows = int(hf['data'].shape[0])
    if 'source_frame_indices' in hf:
        raw = np.asarray(hf['source_frame_indices'][:])
        declared = hf.attrs['original_frame_count']
        if (not np.isfinite(raw).all() or not np.equal(raw,np.floor(raw)).all()
                or not np.isfinite(declared) or declared != int(declared)):
            raise ValueError('Original-frame metadata must contain finite integer indices and counts.')
        indices = raw.astype(np.int64)
        total = int(declared)
        complete = True
    else:
        identity = hf.attrs.get('content_identity', '{}')
        if isinstance(identity, bytes):
            identity = identity.decode()
        identity = json.loads(identity)
        if 'keyprenorm_version' in hf.attrs and 'edge_skip' not in identity:
            raise ValueError("Legacy normalized input lacks frame identity; recreate it from its positional-calcs source.")
        offset = int(identity.get('edge_skip', 0))
        indices = np.arange(offset, offset + rows, dtype=np.int64)
        total = offset + rows
        complete = not bool(offset or 'keyprenorm_version' in hf.attrs)
    if (len(indices) != rows or total < 0 or (rows and (indices[0] < 0 or indices[-1] >= total))
            or (rows > 1 and not np.all(np.diff(indices) == 1))):
        raise ValueError("Feature rows and original-frame metadata disagree.")
    return total, indices, complete


def write_h5_frame_timeline(hf, *, frame_count, indices):
    """Store the original movie length and retained zero-based frame indices."""
    hf.attrs['frame_identity_version'] = 'original-movie-v1'
    hf.attrs['original_frame_count'] = int(frame_count)
    hf.create_dataset('source_frame_indices', data=np.asarray(indices, dtype=np.int64), compression='gzip')


def consecutive_frames(values, *, source, require_origin=False):
    """Validate consecutive, finite, nonnegative integer frame indices without reordering.

    When require_origin is true, require the first index to be zero or one.
    """
    numbers = pd.to_numeric(pd.Series(values), errors='coerce').to_numpy(dtype=float)
    if (not np.isfinite(numbers).all() or np.any(numbers % 1 != 0)
            or np.any(numbers < 0) or (len(numbers) > 1 and np.any(np.diff(numbers) != 1))):
        raise ValueError(f'{source}: frames must be finite, nonnegative consecutive integers; no sorting/renumbering performed.')
    if require_origin and len(numbers) and numbers[0] not in (0, 1):
        raise ValueError(f'{source}: source frame timeline must start at0 or1.')
    return numbers.astype(np.int64)
