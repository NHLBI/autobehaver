"""Content identities for normalized features and their paired labels."""
from __future__ import annotations
from autobehaver.completion import code_digest, file_digest
import hashlib
import json
from pathlib import Path
import numpy as np

IDENTITY_VERSION = 'keyprenorm-content-v1'


def file_hash(path):
    return file_digest(path)


def normalization_identity(csv_file, columns, means, stds, edge_skip):
    # Use the exact numerical parameters applied by the frozen float32 worker.
    """Hash source content, ordered features, normalization parameters, trim policy, and code."""
    params = hashlib.sha256(np.asarray(means, dtype='<f4').tobytes()+
                            np.asarray(stds, dtype='<f4').tobytes()).hexdigest()
    obj = dict(version=IDENTITY_VERSION, source=file_hash(csv_file), columns=list(columns),
               parameters=params, edge_skip=int(edge_skip), row_policy='full-movie-both-edges-v1', math='float32-apply-nan-zero-v1')
    folder = Path(__file__).resolve().parent
    obj['application_sources'] = code_digest([folder/name for name in ('normalize.py','identity.py','csv_chunks.py')])
    return json.dumps(obj, sort_keys=True, separators=(',',':'))


def label_identity(label_file, feature_identity):
    """Hash source labels together with their paired normalized-feature identity."""
    return json.dumps(dict(version=IDENTITY_VERSION, label_source=file_hash(label_file),
                           feature_identity=feature_identity), sort_keys=True, separators=(',',':'))
