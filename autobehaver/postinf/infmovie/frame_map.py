"""Explicit mapping of inference rows to original movie frames."""
from __future__ import annotations
import csv
import json
from pathlib import Path
import numpy as np
from dataclasses import dataclass
from contextlib import contextmanager
from itertools import chain
from autobehaver.completion import file_signature


@dataclass
class InferenceCSVSummary:
    """Hold checked original-frame IDs, row offsets, selection columns, and the source
    signature.
    """
    headers: tuple
    frames: np.ndarray
    row_offsets: np.ndarray
    columns: dict
    signature: tuple

    @contextmanager
    def reader(self, csv_path, start_row=0):
        """Seek directly to a checked CSV record, including quoted multiline rows."""
        if file_signature(csv_path) != self.signature:
            raise ValueError('Inference CSV changed after its frame/index validation')
        with open(csv_path, 'rb') as stream:
            if start_row < len(self.row_offsets):
                stream.seek(int(self.row_offsets[start_row]))
                rows = csv.reader(iter(lambda: stream.readline().decode('utf-8'), ''))
            else:
                rows = iter(())
            # Match read_csv_summary: empty physical records are not frame rows.
            yield iter(chain([self.headers], (row for row in rows if row)))
        if file_signature(csv_path) != self.signature:
            raise ValueError('Inference CSV changed while rendering')


def read_csv_summary(csv_path, *, source_frame_offset=None, video_frames=None, include_selection=False):
    """Validate original frame identity and build a row-seek index in one parse."""
    signature = file_signature(csv_path)
    with open(csv_path, 'rb') as handle:
        reader = csv.reader(iter(lambda: handle.readline().decode('utf-8'), ''))
        fields = next(reader, [])
        if fields:
            fields[0] = fields[0].lstrip('\ufeff')
        data_start = handle.tell()
    if len(fields) != len(set(fields)):
        raise ValueError('Inference CSV contains duplicate headers')
    if source_frame_offset is None:
        receipt = Path(str(csv_path) + '.provenance.json')
        metadata = json.loads(receipt.read_text()) if receipt.exists() else {}
        source_frame_offset = metadata.get('source_frame_offset')
        if source_frame_offset is None:
            source_frame_offset = metadata.get('settings', {}).get('source_frame_offset')
        if source_frame_offset is None:
            if 'Source Frame Index' in fields:
                source_frame_offset = 0
        if source_frame_offset is None:
            raise ValueError('InfMovie requires explicit source_frame_offset (0 for untrimmed inputs, 5 when five leading rows were removed)')
    if isinstance(source_frame_offset, bool) or int(source_frame_offset) != source_frame_offset:
        raise ValueError('source_frame_offset must be an integer')
    ids, offsets = [], []
    selected = [name for name in fields if include_selection and
                (name in ('Predicted Pose Label', 'Predicted Orient Label', 'Smoothed Pose Label', 'Smoothed Orient Label')
                 or (name.startswith(('Pose-', 'Orient-')) and name.endswith('-Prob')))]
    columns = {name: [] for name in selected}
    indexes = {name: fields.index(name) for name in fields}
    with open(csv_path, 'rb') as handle:
        handle.seek(data_start)
        reader = csv.reader(iter(lambda: handle.readline().decode('utf-8'), ''))
        if 'Frame Number' not in fields:
            raise ValueError('Inference CSV lacks one-based Frame Number')
        while True:
            position = handle.tell()
            row = next(reader, None)
            if row is None:
                break
            if not row:
                continue
            if len(row) != len(fields):
                raise ValueError('Inference CSV contains malformed rows')
            value = float(row[indexes['Frame Number']])
            if 'Source Frame Index' in indexes:
                original = float(row[indexes['Source Frame Index']])
                if source_frame_offset != 0 or original != value - 1:
                    raise ValueError('Original frame identity cannot be overridden by a manual offset.')
            if not np.isfinite(value) or value != int(value) or value < 1:
                raise ValueError('Frame Number must contain positive integer IDs')
            ids.append(int(value) - 1 + int(source_frame_offset))
            offsets.append(position)
            for name in selected:
                text = row[indexes[name]]
                if name.endswith('-Prob'):
                    try:
                        value = float(text)
                    except ValueError:
                        value = float('nan')
                    columns[name].append(value)
                else:
                    columns[name].append('' if text in ('nan', 'None') else text)
    frames = np.asarray(ids, dtype=np.int64)
    if not len(frames) or frames[0] < 0 or (len(frames) > 1 and not np.all(np.diff(frames) == 1)):
        raise ValueError('Inference frame IDs must be nonempty, ordered, unique and consecutive')
    if video_frames is not None and (video_frames <= 0 or frames[-1] >= int(video_frames)):
        raise ValueError('Inference frame mapping exceeds the source movie')
    if file_signature(csv_path) != signature:
        raise ValueError('Inference CSV changed during validation')
    return InferenceCSVSummary(tuple(fields), frames, np.asarray(offsets, dtype=np.int64),
                               {key: np.asarray(values, dtype=float if key.endswith('-Prob') else object)
                                for key, values in columns.items()}, signature)


def load_source_frames(csv_path, *, source_frame_offset=None, video_frames=None):
    """Return consecutive original zero-based frame IDs, rejecting ambiguity."""
    return read_csv_summary(csv_path, source_frame_offset=source_frame_offset, video_frames=video_frames).frames


def source_aligned(values, frames, *, fill):
    """Pad absent leading source frames, never compress or reinterpret them."""
    values = np.asarray(values)
    if len(values) != len(frames):
        raise ValueError('Frame map and values have different lengths')
    output = np.full(int(frames[-1]) + 1, fill, dtype=values.dtype)
    output[frames] = values
    return output
