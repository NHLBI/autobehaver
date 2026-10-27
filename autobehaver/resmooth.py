"""Regenerate smoothed labels in verified copies of existing PosePred CSVs."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import tempfile
import numpy as np
import pandas as pd
from .inference import smooth_bouts_vectorized


def _hash(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def resmooth_file(source, destination, *, min_pose_bout=5, min_orient_bout=5, source_frame_offset=None):
    """Write a new CSV and provenance receipt with regenerated smoothed labels.

    Preserve all other cells. Use modern source indices or recorded legacy
    alignment; require an explicit verified offset for unidentified legacy
    inputs. Require a new destination and verify readback and source identity.
    """
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if source == destination or destination.exists() or Path(str(destination)+'.provenance.json').exists():
        raise FileExistsError('Resmoothing requires a new, unused destination')
    if min_pose_bout < 1 or min_orient_bout < 1:
        raise ValueError('Minimum bout lengths must be positive')
    if source_frame_offset is not None and (isinstance(source_frame_offset, bool) or int(source_frame_offset) != source_frame_offset or source_frame_offset < 0):
        raise ValueError('source_frame_offset must be a nonnegative integer')
    source_hash = _hash(source)
    with source.open(newline='') as handle:
        reader = csv.DictReader(handle)
        headers = list(reader.fieldnames or [])
        rows = list(reader)
    if len(headers) != len(set(headers)) or not rows or any(None in row or any(v is None for v in row.values()) for row in rows):
        raise ValueError('Inference CSV has missing/duplicate headers or malformed/empty rows')
    frame_ids = [float(row['Frame Number']) for row in rows]
    if any(not np.isfinite(v) or v != int(v) or v < 1 for v in frame_ids) or any(b-a != 1 for a,b in zip(frame_ids,frame_ids[1:])):
        raise ValueError('Frame Number must be positive, ordered, unique and consecutive')
    # Modern files carry original frame indices. Legacy files need a recorded
    # or explicitly supplied offset; never assume that every file lost five frames.
    from .postinf.infmovie.frame_map import read_csv_summary
    summary = read_csv_summary(source, source_frame_offset=source_frame_offset)
    effective_offset = int(summary.frames[0]) - (int(frame_ids[0]) - 1)
    upstream = Path(str(source)+'.provenance.json')
    upstream_metadata = json.loads(upstream.read_text()) if upstream.exists() else None
    originals = [dict(row) for row in rows]
    changed_columns = []
    for prefix, raw, smoothed, minimum in (
        ('Pose', 'Predicted Pose Label', 'Smoothed Pose Label', min_pose_bout),
        ('Orient', 'Predicted Orient Label', 'Smoothed Orient Label', min_orient_bout),
    ):
        if raw not in headers:
            continue
        columns = [h for h in headers if h.startswith(prefix+'-') and h.endswith('-Prob')]
        if not columns:
            raise ValueError(f'Missing {prefix} probabilities needed for smoothing')
        probabilities = pd.DataFrame({h: pd.to_numeric([r[h] for r in rows], errors='raise') for h in columns})
        labels = [r[raw] if r[raw] and r[raw].strip().lower() not in {'nan','none'} else None for r in rows]
        output = smooth_bouts_vectorized(labels, probabilities, minimum, prefix, ['Jump'] if prefix=='Pose' else [])
        if smoothed not in headers:
            headers.append(smoothed)
        changed_columns.append(smoothed)
        for row, label in zip(rows, output):
            row[smoothed] = '' if pd.isna(label) else str(label)
    if not changed_columns:
        raise ValueError('No raw behavior/orientation labels to smooth')
    for old, new in zip(originals, rows):
        if any(new[key] != value for key,value in old.items() if key not in changed_columns):
            raise RuntimeError('Resmoothing altered a nonsmoothed CSV value')
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile('w', newline='', delete=False, dir=destination.parent, suffix='.csv.tmp') as handle:
            temporary = Path(handle.name)
            writer = csv.DictWriter(handle, fieldnames=headers)
            writer.writeheader(); writer.writerows(rows)
        with temporary.open(newline='') as handle:
            reread = list(csv.DictReader(handle))
        if reread != rows or _hash(source) != source_hash:
            raise RuntimeError('Resmoothing readback/input-identity verification failed')
        import autobehaver
        receipt = {
            'status': 'complete', 'operation': 'corrected_label_smoothing',
            'source_frame_offset': effective_offset,
            'frame_identity': 'original-movie-v1' if 'Source Frame Index' in headers else 'legacy-recorded-offset-v1',
            'source_frame_indices_sha256': hashlib.sha256(summary.frames.astype('<i8').tobytes()).hexdigest(),
            'source': {'path': str(source), 'sha256': source_hash},
            'upstream_provenance': upstream_metadata,
            'upstream_provenance_sha256': _hash(upstream) if upstream.exists() else None,
            'smoothing_code_sha256': _hash(Path(__file__).with_name('inference.py')),
            'label_gap_code_sha256': _hash(Path(__file__).with_name('label_gaps.py')),
            'resmoothing_code_sha256': _hash(Path(__file__)),
            'package_version': getattr(autobehaver,'__version__',None),
            'settings': {'min_pose_bout': min_pose_bout, 'min_orient_bout': min_orient_bout,
                         'pose_exempt_labels': ['Jump'], 'label_gap_policy': 'matching-internal-flanks-max4-v1'},
            'row_count': len(rows), 'changed_columns': changed_columns,
            'output': {'path': destination.name, 'sha256': _hash(temporary)},
        }
        if upstream_metadata:
            for key in ('source_positional_sha256', 'positional_source', 'original_frame_count', 'complete_movie_coverage'):
                if key in upstream_metadata:
                    receipt[key] = upstream_metadata[key]
        os.replace(temporary, destination); temporary = None
        receipt_path = Path(str(destination)+'.provenance.json')
        with tempfile.NamedTemporaryFile('w', delete=False, dir=destination.parent, suffix='.json.tmp') as handle:
            json.dump(receipt, handle, indent=2); handle.write('\n')
            temporary = Path(handle.name)
        os.replace(temporary, receipt_path); temporary = None
        return receipt
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--min-pose-bout', type=int, default=5)
    parser.add_argument('--min-orient-bout', type=int, default=5)
    parser.add_argument('--source-frame-offset', type=int, default=None,
                        help='Legacy files only: supply an offset when their processing metadata has none. Modern frame indices are preserved.')
    args = parser.parse_args(argv)
    source, destination = Path(args.input_dir).resolve(), Path(args.output_dir).resolve()
    if source == destination or source in destination.parents:
        raise ValueError('Output must be a separate new inference directory')
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError('Output directory must be empty/new')
    files = sorted(source.glob('*_inference.csv'))
    if not files:
        raise FileNotFoundError('No *_inference.csv inputs')
    receipts = []
    for path in files:
        receipt = resmooth_file(path, destination/path.name, min_pose_bout=args.min_pose_bout,
                               min_orient_bout=args.min_orient_bout, source_frame_offset=args.source_frame_offset)
        print(f'{path.name}: verified {receipt["row_count"]} rows')
        receipts.append({'filename':path.name,'source_sha256':receipt['source']['sha256'],
                         'output_sha256':receipt['output']['sha256'],'rows':receipt['row_count'],
                         'source_frame_offset':receipt['source_frame_offset']})
    # This run-level completion receipt is written only after every selected
    # recording passed. Partial outputs never certify a completed cohort.
    from .atomic_io import atomic_write_json
    atomic_write_json({'status':'complete','operation':'corrected_label_smoothing',
                       'source_directory':str(source),'file_count':len(receipts),
                       'source_frame_offset':args.source_frame_offset,'files':receipts},
                      destination/'resmoothing.complete.json')


if __name__ == '__main__':
    main()
