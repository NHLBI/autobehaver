"""Bounded-memory storage for numerical FeatureImp task archives.

Uncompressed arrays can be memory-mapped; compressed legacy archives remain readable.
"""
from __future__ import annotations

import os
from pathlib import Path
import struct
import tempfile
import zipfile

import numpy as np


def load_npz_arrays(path, required=(), optional=()):
    """Map numeric ZIP_STORED NPY members; support compressed legacy archives."""
    result = {}
    with zipfile.ZipFile(path) as archive:
        members = {item.filename[:-4]: item for item in archive.infolist() if item.filename.endswith('.npy')}
        missing = set(required) - set(members)
        if missing:
            raise ValueError(f'Archive missing required arrays {sorted(missing)}: {path}')
        for key in list(required) + [key for key in optional if key in members]:
            member = members[key]
            if member.compress_type != zipfile.ZIP_STORED or member.flag_bits & 1:
                with archive.open(member) as stream:
                    result[key] = np.load(stream, allow_pickle=False)
                continue
            with open(path, 'rb') as stream:
                stream.seek(member.header_offset)
                header = stream.read(30)
                if len(header) != 30 or header[:4] != b'PK\x03\x04':
                    raise ValueError(f'Invalid local archive header: {path}/{key}')
                name_len, extra_len = struct.unpack_from('<HH', header, 26)
                stream.seek(name_len + extra_len, os.SEEK_CUR)
                npy_start = stream.tell()
                version = np.lib.format.read_magic(stream)
                if version == (1, 0):
                    shape, fortran, dtype = np.lib.format.read_array_header_1_0(stream)
                elif version == (2, 0):
                    shape, fortran, dtype = np.lib.format.read_array_header_2_0(stream)
                else:
                    raise ValueError(f'Unsupported numeric NPY version {version}: {path}/{key}')
                offset = stream.tell()
                if dtype.hasobject:
                    raise ValueError(f'Object array is not allowed: {path}/{key}')
                nbytes = int(np.prod(shape, dtype=np.int64)) * dtype.itemsize
                if offset - npy_start + nbytes != member.file_size:
                    raise ValueError(f'Invalid array size: {path}/{key}')
            result[key] = (np.memmap(path, mode='r', dtype=dtype, offset=offset,
                                    shape=shape, order='F' if fortran else 'C')
                           if nbytes else np.empty(shape, dtype=dtype))
    return result


def publish_npz(path, arrays):
    """Stream each contiguous array to a temporary NPZ, then publish atomically."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=destination.parent, suffix='.npz')
    os.close(fd)
    try:
        np.savez(temporary, **arrays)
        with zipfile.ZipFile(temporary) as archive:
            if set(archive.namelist()) != {f'{key}.npy' for key in arrays}:
                raise ValueError('Saved task archive has an incomplete array inventory')
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.remove(temporary)


class TaskArchiveWriter:
    """Fill disk-backed task arrays in arrival order without retaining results."""

    def __init__(self, directory, task_features, *, n_train, n_all):
        """Allocate disk-backed arrays for the frozen subset tasks and sample
        dimensions.
        """
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.specs = dict(task_features)
        self.seen = set()
        self.n_train, self.n_all = int(n_train), int(n_all)
        self.index = self.feature_offset = self.oof_offset = self.all_offset = 0
        tasks = len(self.specs)
        total_features = sum(len(value) for value in self.specs.values())
        self.shap, self.margin = {}, {}
        def allocate(collection, name, shape, dtype):
            collection[name] = (np.lib.format.open_memmap(directory / f'{len(collection)}_{name}_{id(collection)}.npy',
                                                        mode='w+', dtype=dtype, shape=shape)
                                if np.prod(shape) else np.empty(shape, dtype=dtype))
        for collection in (self.shap, self.margin):
            allocate(collection, 'h_hit', (tasks,), np.int32)
            allocate(collection, 'subset_idx', (tasks,), np.int32)
        allocate(self.shap, 'feature_count', (tasks,), np.int32)
        allocate(self.shap, 'feature_indices_flat', (total_features,), np.int32)
        for key, rows in [('oof_shap_flat', n_train), ('insample_shap_flat', n_train), ('all_groups_shap_flat', n_all)]:
            allocate(self.shap, key, (int(rows) * total_features,), np.float32)
        for key, rows, dtype in [('oof_margin_sum', n_train, np.float64), ('insample_margin_sum', n_train, np.float64),
                                 ('all_groups_margin_sum', n_all, np.float64), ('test_pred_count', n_train, np.int32),
                                 ('train_pred_count', n_train, np.int32)]:
            allocate(self.margin, key, (tasks, int(rows)), dtype)
        allocate(self.margin, 'all_groups_model_count', (tasks,), np.int32)

    def append(self, result, feature_to_idx):
        """Validate a subset result against its frozen task identity and append its
        numerical arrays.
        """
        key = int(result['h_hit']), int(result['subset_idx'])
        features = tuple(result['features'])
        if key in self.seen or key not in self.specs or features != tuple(self.specs[key]):
            raise ValueError(f'Task archive identity/features differ from frozen plan: {key}')
        i, count = self.index, len(features)
        for collection in (self.shap, self.margin):
            collection['h_hit'][i], collection['subset_idx'][i] = key
        self.shap['feature_count'][i] = count
        self.shap['feature_indices_flat'][self.feature_offset:self.feature_offset + count] = [feature_to_idx[f] for f in features]
        for archive_key, result_key, rows, offset in [
            ('oof_shap_flat', 'oof_shap_sum', self.n_train, self.oof_offset),
            ('insample_shap_flat', 'insample_shap_sum', self.n_train, self.oof_offset),
            ('all_groups_shap_flat', 'all_groups_shap_sum', self.n_all, self.all_offset),
        ]:
            array = np.asarray(result[result_key], dtype=np.float32)
            if array.shape != (rows, count) or not np.isfinite(array).all():
                raise ValueError(f'Misaligned/nonfinite task SHAP values: {key}/{result_key}')
            self.shap[archive_key][offset:offset + rows * count] = array.reshape(-1)
        for name in ('oof_margin_sum', 'insample_margin_sum', 'all_groups_margin_sum', 'test_pred_count', 'train_pred_count'):
            array = np.asarray(result[name], dtype=self.margin[name].dtype)
            if array.shape != self.margin[name].shape[1:] or not np.isfinite(array).all():
                raise ValueError(f'Misaligned/nonfinite task margin values: {key}/{name}')
            self.margin[name][i] = array
        self.margin['all_groups_model_count'][i] = int(result['all_groups_model_count'])
        self.feature_offset += count
        self.oof_offset += self.n_train * count
        self.all_offset += self.n_all * count
        self.index += 1
        self.seen.add(key)

    def publish(self, *, margin_path, shap_path, metadata):
        """Publish margin and SHAP archives after every planned task has been written."""
        if self.seen != set(self.specs):
            raise ValueError('Task archive lacks complete frozen task coverage')
        for arrays in (self.margin, self.shap):
            for value in arrays.values():
                if isinstance(value, np.memmap):
                    value.flush()
        shape_metadata = {'n_train': self.n_train, 'n_all': self.n_all, 'include_insample': True, 'include_all_groups': True}
        margin_metadata = {**metadata, 'family': str(metadata['family']).replace('_tasks', '_margin_tasks')}
        shap_metadata = {**metadata, 'family': str(metadata['family']).replace('_tasks', '_shap_tasks')}
        publish_npz(margin_path, {**self.margin, **{f'meta__{k}': np.asarray(v) for k, v in margin_metadata.items()}})
        publish_npz(shap_path, {**self.shap, **{f'meta__{k}': np.asarray(v) for k, v in {**shap_metadata, **shape_metadata}.items()}})

    def close(self):
        """Flush and close task-owned memory maps."""
        for collection in (self.shap, self.margin):
            for value in collection.values():
                if isinstance(value, np.memmap):
                    value.flush()
                    value._mmap.close()
            collection.clear()
