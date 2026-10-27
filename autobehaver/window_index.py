"""Compact temporal-window indexing and process-portable feature storage."""
from collections.abc import Mapping, Sequence
import bisect
import contextvars
import functools
import os
from pathlib import Path
import sys
import numpy as np

from .scratch import scratch_workspace

_LOADER_SCOPE = contextvars.ContextVar('autobehaver_loader_scope', default=None)


def register_dataloader(loader):
    """Track training loaders so error tracebacks cannot keep workers alive."""
    loaders = _LOADER_SCOPE.get()
    if loaders is not None:
        loaders.append(loader)
    return loader


class WindowInstructions(Sequence):
    """Expose window metadata without materializing every instruction.

    Inference center indices remain zero-based original-movie frame indices.
    """
    def __init__(self, owner, segments):
        self.owner = owner
        self.segments = segments
        self.ends = np.cumsum([segment['count'] for segment in segments], dtype=np.int64)

    def __len__(self):
        return int(self.ends[-1]) if len(self.ends) else 0

    def locate(self, index):
        index = int(index)
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        position = bisect.bisect_right(self.ends, index)
        base = int(self.ends[position - 1]) if position else 0
        return self.segments[position], index - base

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self[i] for i in range(*index.indices(len(self)))]
        segment, local = self.locate(index)
        start = local if segment.get('starts') is None else int(segment['starts'][local])
        result = {'input_file': segment['path'], 'sequence_start': start,
                  'sequence_end': start + self.owner.sequence_length}
        if self.owner.inference_mode:
            result['middle_idx'] = segment['source_start'] + start + self.owner.sequence_length // 2
        else:
            for head in ('pose', 'orient'):
                labels = segment.get(head + '_labels')
                label = str(labels[local]) if labels is not None else None
                result[head + '_label'] = label
                result[head + '_index'] = getattr(self.owner, head + '_dict', {}).get(label, -1)
        return result

    def indices_for_files(self, names):
        root = Path(self.owner.directory)
        selected = {os.path.normcase(str((root / str(name)).resolve())) for name in names}
        pieces = []
        base = 0
        for segment in self.segments:
            if os.path.normcase(str(Path(segment['path']).resolve())) in selected:
                pieces.append(np.arange(base, base + segment['count'], dtype=np.int64))
            base += segment['count']
        return np.concatenate(pieces) if pieces else np.empty(0, dtype=np.int64)

    def label_codes(self, head):
        dictionary = getattr(self.owner, head + '_dict', {})
        pieces = []
        for segment in self.segments:
            labels = segment.get(head + '_labels')
            if labels is None:
                pieces.append(np.full(segment['count'], -1, dtype=np.int32))
            else:
                unique, inverse = np.unique(labels, return_inverse=True)
                codes = np.array([dictionary.get(str(value), -1) for value in unique], dtype=np.int32)
                pieces.append(codes[inverse])
        return np.concatenate(pieces) if pieces else np.empty(0, dtype=np.int32)


class DiskFeatureStore(Mapping):
    """Share local .npy arrays with workers through copy-on-write memory maps.

    Only paths cross process boundaries. Writes to a Torch view do not modify
    the backing file; the owning process removes the task workspace.
    """
    def __init__(self, stage='posepred', scratch_dir=None):
        self._context = scratch_workspace(stage, scratch_dir=scratch_dir)
        self.directory = self._context.__enter__()
        self.paths = {}
        self._arrays = {}
        self._owner_pid = os.getpid()

    def add(self, key, values):
        path = self.directory / f'f{len(self.paths):06d}.npy'
        values = np.ascontiguousarray(values, dtype=np.float32)
        free = __import__('shutil').disk_usage(self.directory).free
        if free < values.nbytes + 4096:
            raise OSError(f'Insufficient scratch space for feature array: {self.directory}')
        np.save(path, values, allow_pickle=False)
        self.paths[key] = str(path)

    def __getitem__(self, key):
        if key not in self._arrays:
            self._arrays[key] = np.load(self.paths[key], mmap_mode='c', allow_pickle=False)
        return self._arrays[key]

    def __iter__(self):
        return iter(self.paths)

    def __len__(self):
        return len(self.paths)

    def clear(self):
        self._close_maps()
        for value in self.paths.values():
            Path(value).unlink(missing_ok=True)
        self.paths.clear()

    def _close_maps(self):
        for values in self._arrays.values():
            values._mmap.close()
        self._arrays.clear()

    def close(self, error=None):
        self._close_maps()
        context = getattr(self, '_context', None)
        if context is not None and os.getpid() == self._owner_pid:
            self._context = None
            context.__exit__(*(error or (None, None, None)))

    def __getstate__(self):
        return {'directory': self.directory, 'paths': self.paths,
                '_owner_pid': self._owner_pid, '_arrays': {}, '_context': None}

    def __setstate__(self, state):
        self.__dict__.update(state)

    def __del__(self):
        try:
            self.close()
        except (OSError, AttributeError, RuntimeError):
            pass


def close_feature_stores(function):
    """Release input scratch after the function and its loader locals finish."""
    @functools.wraps(function)
    def wrapped(*args, **kwargs):
        loaders = []
        token = _LOADER_SCOPE.set(loaders)
        stores = []
        for dataset in list(args[:2]) + [kwargs.get('train_dataset'), kwargs.get('val_dataset')]:
            while dataset is not None and hasattr(dataset, 'dataset'):
                dataset = dataset.dataset
            store = getattr(dataset, 'data_in_memory', None)
            if isinstance(store, DiskFeatureStore) and all(store is not other for other in stores):
                stores.append(store)
        try:
            return function(*args, **kwargs)
        finally:
            error = sys.exc_info()
            try:
                for loader in loaders:
                    iterator = getattr(loader, '_iterator', None)
                    if iterator is not None:
                        iterator._shutdown_workers()
            finally:
                _LOADER_SCOPE.reset(token)
                for store in stores:
                    store.close(error if error[0] is not None else None)
    return wrapped
