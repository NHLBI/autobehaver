"""Read and flatten DeepLabCut CSV headers for KeyPre."""

from __future__ import annotations

import pandas as pd
from ..frame_identity import consecutive_frames


def _flatten(df):
    """Validate the DLC triplets and flatten a bounded frame of coordinates."""
    flattened = [f"{col[1]}_{col[2]}" for col in df.columns]
    if len(flattened) != len(set(flattened)):
        raise ValueError('Duplicate flattened DLC bodypart/coordinate columns.')
    triplets = {}
    for scorer, part, coordinate in df.columns:
        if part == 'bodyparts' and coordinate == 'coords':
            continue
        if coordinate not in ('x', 'y', 'likelihood') or not str(part).strip():
            raise ValueError(f'Invalid DLC bodypart/coordinate header: {part}/{coordinate}')
        triplets.setdefault(str(part), set()).add(str(coordinate))
    incomplete = [part for part, coords in triplets.items() if coords != {'x', 'y', 'likelihood'}]
    if incomplete or not triplets:
        raise ValueError(f'DLC requires complete x/y/likelihood triplets: {incomplete}')
    df.columns = flattened
    return df.drop(columns=['bodyparts_coords'], errors='ignore')


def load_dlc_csv_flat(csv_file: str) -> pd.DataFrame:
    """Read a three-row DeepLabCut header and preserve the frame index.

    Flatten coordinates to part_x, part_y, and part_likelihood columns.
    """

    df = pd.read_csv(csv_file, header=[0, 1, 2], index_col=0)
    consecutive_frames(df.index, source=str(csv_file), require_origin=True)
    return _flatten(df)


def dlc_chunks(csv_file, *, chunk_rows, halo=0):
    """Yield (context frame, retained start/stop, first, last) without losing rows.

    Keep one lookahead block and a left halo. Validate the original frame index
    across reads; never reset or infer a missing source frame.
    """
    if chunk_rows < 1 or halo < 0 or halo >= chunk_rows:
        raise ValueError('Require chunk_rows > halo >= 0.')
    previous_index = None
    buffer = None
    left = 0
    first = True
    with pd.read_csv(csv_file, header=[0, 1, 2], index_col=0, chunksize=chunk_rows) as reader:
        for raw in reader:
            indices = consecutive_frames(raw.index, source=str(csv_file), require_origin=previous_index is None)
            if len(indices):
                if previous_index is not None and indices[0] != previous_index + 1:
                    raise ValueError(f'{csv_file}: nonconsecutive source frames at a chunk boundary.')
                previous_index = indices[-1]
            raw = _flatten(raw)
            buffer = raw if buffer is None else pd.concat([buffer, raw])
            # Retain lookahead even when halo=0 to identify the real final block.
            while len(buffer) > left + chunk_rows + halo:
                stop = left + chunk_rows
                yield buffer.iloc[:stop + halo].copy(), left, stop, first, False
                buffer = buffer.iloc[stop - halo:].copy()
                left, first = halo, False
        if buffer is not None:
            yield buffer, left, len(buffer), first, True


def dlc_row_count(csv_file, chunk_rows=4096):
    """Count/validate source frames using only bounded coordinate chunks."""
    return sum(stop-start for _, start, stop, _, _ in dlc_chunks(csv_file, chunk_rows=chunk_rows))
