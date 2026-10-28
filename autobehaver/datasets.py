"""Load HDF5 feature windows, center-frame labels and inference timelines."""

from __future__ import annotations

import multiprocessing
import os
import json
import logging
from pathlib import Path
from typing import Optional, TYPE_CHECKING

from .preinf.frame_identity import h5_frame_timeline
import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from .utils import seed_worker
from .window_index import WindowInstructions, DiskFeatureStore, register_dataloader

if TYPE_CHECKING:  # pragma: no cover
    from .feature_selection import FeaturePlan

__all__ = [
    'KeypointsDataset',
    'build_dataloader',
    'validate_feature_hdf5_file',
    'validate_pretraining_metadata',
    'validate_supervised_training_metadata',
]


def _resolve_training_file(directory: str, declared_name: object, *, field: str) -> Path:
    """Resolve one metadata path and require it to stay inside the dataset directory."""

    if declared_name is None or pd.isna(declared_name) or not str(declared_name).strip():
        raise ValueError(f"Training metadata contains a blank {field!r} value.")
    raw = Path(str(declared_name).strip())
    if raw.is_absolute():
        candidate = raw.resolve()
    else:
        candidate = (Path(directory).resolve() / raw).resolve()
    root = Path(directory).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError(
            f"Training metadata {field!r} path escapes the dataset directory: {declared_name!r}"
        ) from exc
    return candidate


def validate_feature_hdf5_file(
    path: str | os.PathLike[str],
    *,
    sequence_length: int,
    expected_columns: Optional[list[str]] = None,
    scan_values: bool = True,
) -> tuple[int, list[str]]:
    """Validate feature shape and headers without changing the file.

    When ``scan_values`` is enabled, scan the numeric array in bounded chunks
    for NaN/Inf. Missing headers remain supported for legacy files; malformed,
    blank or duplicate supplied headers are rejected.
    """

    feature_path = Path(path)
    if not feature_path.is_file():
        raise FileNotFoundError(f"Training feature file does not exist: {feature_path}")
    with h5py.File(feature_path, 'r') as h5f:
        if 'data' not in h5f:
            raise ValueError(f"Training feature file is missing dataset 'data': {feature_path}")
        dset = h5f['data']
        if getattr(dset, 'ndim', None) != 2:
            raise ValueError(f"Training feature dataset must be 2D: {feature_path}")
        n_rows, n_features = (int(dset.shape[0]), int(dset.shape[1]))
        if n_rows < int(sequence_length):
            raise ValueError(
                f"Training feature file has {n_rows} rows, fewer than sequence_length="
                f"{int(sequence_length)}: {feature_path}"
            )
        if n_features <= 0:
            raise ValueError(f"Training feature file has no feature columns: {feature_path}")
        if not np.issubdtype(dset.dtype, np.number):
            raise ValueError(f"Training feature dataset is not numeric: {feature_path}")

        raw_columns = None
        if 'columns' in h5f.attrs:
            raw_columns = h5f.attrs['columns']
        elif 'columns' in dset.attrs:
            raw_columns = dset.attrs['columns']
        elif 'columns' in h5f:
            raw_columns = h5f['columns'][()]
        if raw_columns is None:
            columns = [f"col_{index}" for index in range(n_features)]
        else:
            columns = _to_str_list(raw_columns)
            if len(columns) != n_features:
                raise ValueError(
                    f"Training feature column metadata length {len(columns)} does not match "
                    f"data width {n_features}: {feature_path}"
                )
            if any(not column.strip() for column in columns):
                raise ValueError(f"Training feature file has blank column names: {feature_path}")
            if len(set(columns)) != len(columns):
                raise ValueError(f"Training feature file has duplicate column names: {feature_path}")

        if expected_columns is not None and columns != expected_columns:
            raise ValueError(
                f"Training feature schema does not match the first declared file: {feature_path}"
            )

        chunk_rows = max(1, min(100_000, n_rows))
        for start in range(0, n_rows if scan_values else 0, chunk_rows):
            values = np.asarray(dset[start : start + chunk_rows])
            if not np.isfinite(values).all():
                raise ValueError(f"Training feature file contains NaN or Inf values: {feature_path}")
    return n_rows, columns


def _canonical_label(value):
    text = value.decode('utf-8', errors='strict') if isinstance(value, bytes) else str(value)
    text = text.strip()
    return '' if text.lower() == 'nan' else text


def _validate_label_pair(feature_path, label_h5f):
    """Do not accept a declared label/feature binding from an earlier normalization."""
    raw_identity = label_h5f.attrs.get('content_identity')
    if raw_identity is None:
        return  # Historical/imported labels retain the explicit timeline/length checks.
    try:
        identity = json.loads(raw_identity)
    except (TypeError, ValueError) as exc:
        raise ValueError(f'Invalid label content identity: {label_h5f.filename}') from exc
    if 'feature_identity' not in identity:
        return
    with h5py.File(feature_path, 'r') as features:
        if identity['feature_identity'] != features.attrs.get('content_identity'):
            raise ValueError('Paired label provenance refers to different normalized features: '
                             f'{label_h5f.filename}. Reapply existing source annotations, or rebind '
                             'them only after verifying recording identity and original frame alignment.')


def _validate_label_dataset(
    h5f: h5py.File,
    *,
    dataset_name: str,
    expected_rows: int,
    path: Path,
) -> None:
    if dataset_name not in h5f:
        raise ValueError(f"Training label file is missing dataset {dataset_name!r}: {path}")
    dset = h5f[dataset_name]
    if getattr(dset, 'ndim', None) != 1:
        raise ValueError(f"Training label dataset {dataset_name!r} must be 1D: {path}")
    if h5py.check_string_dtype(dset.dtype) is None:
        raise ValueError(
            f"Training label dataset {dataset_name!r} must contain strings: {path}"
        )
    if int(dset.shape[0]) != int(expected_rows):
        raise ValueError(
            f"Training label length for {dataset_name!r} does not match feature rows "
            f"({int(dset.shape[0])} != {int(expected_rows)}): {path}"
        )
    usable = 0
    chunk_rows = max(1, min(100_000, int(expected_rows)))
    for start in range(0, int(expected_rows), chunk_rows):
        for value in np.asarray(dset[start : start + chunk_rows]).reshape(-1):
            if _canonical_label(value):
                usable += 1
    if usable == 0:
        raise ValueError(f"Training label dataset {dataset_name!r} has no usable labels: {path}")


def validate_supervised_training_metadata(
    directory: str,
    metadata: pd.DataFrame,
    *,
    sequence_length: int,
    pose_pred: bool,
    orient_pred: bool,
    defined_split: bool,
    noval: bool,
    scan_values: bool = True,
) -> pd.DataFrame:
    """Validate the exact file inventory declared for supervised training."""

    if not isinstance(metadata, pd.DataFrame):
        raise ValueError("Supervised training requires a metadata CSV.")
    if metadata.empty:
        raise ValueError("Supervised training metadata contains no rows.")
    required = {'filename', 'labels'}
    if defined_split and not noval:
        required.add('set')
    missing_columns = sorted(required - set(metadata.columns))
    if missing_columns:
        raise ValueError(f"Training metadata is missing required columns: {missing_columns}")

    normalized_names = metadata['filename'].map(lambda value: '' if pd.isna(value) else str(value).strip())
    if (normalized_names == '').any():
        raise ValueError("Training metadata contains blank filenames.")
    duplicates = sorted(normalized_names[normalized_names.duplicated(keep=False)].unique().tolist())
    if duplicates:
        raise ValueError(f"Training metadata contains duplicate filenames: {duplicates[:20]}")

    if defined_split and not noval:
        split_values = metadata['set'].map(lambda value: '' if pd.isna(value) else str(value).strip())
        invalid = sorted(set(split_values) - {'train', 'val'})
        if invalid:
            raise ValueError(
                "Defined training split values must be exactly 'train' or 'val'; "
                f"found: {invalid}"
            )
        if set(split_values) != {'train', 'val'}:
            raise ValueError("Defined training metadata must contain both train and val rows.")

    expected_columns: Optional[list[str]] = None
    for _, row in metadata.iterrows():
        feature_path = _resolve_training_file(directory, row['filename'], field='filename')
        label_path = _resolve_training_file(directory, row['labels'], field='labels')
        n_rows, columns = validate_feature_hdf5_file(
            feature_path,
            sequence_length=sequence_length,
            expected_columns=expected_columns,
            scan_values=scan_values,
        )
        if expected_columns is None:
            expected_columns = columns
        if not label_path.is_file():
            raise FileNotFoundError(f"Training label file does not exist: {label_path}")
        with h5py.File(label_path, 'r') as label_h5f:
            _validate_label_pair(feature_path, label_h5f)
            if pose_pred:
                _validate_label_dataset(
                    label_h5f,
                    dataset_name='Behavior',
                    expected_rows=n_rows,
                    path=label_path,
                )
            if orient_pred:
                _validate_label_dataset(
                    label_h5f,
                    dataset_name='Orientation',
                    expected_rows=n_rows,
                    path=label_path,
                )
    return metadata.copy()


def validate_pretraining_metadata(
    directory: str,
    metadata: pd.DataFrame,
    *,
    sequence_length: int,
    scan_values: bool = True,
) -> pd.DataFrame:
    """Validate every feature file declared for self-supervised pretraining."""

    if not isinstance(metadata, pd.DataFrame):
        raise ValueError("Pretraining requires a metadata CSV.")
    if metadata.empty:
        raise ValueError("Pretraining metadata contains no rows.")
    if 'filename' not in metadata.columns:
        raise ValueError("Pretraining metadata is missing required column 'filename'.")
    names = metadata['filename'].map(lambda value: '' if pd.isna(value) else str(value).strip())
    if (names == '').any():
        raise ValueError("Pretraining metadata contains blank filenames.")
    duplicates = sorted(names[names.duplicated(keep=False)].unique().tolist())
    if duplicates:
        raise ValueError(f"Pretraining metadata contains duplicate filenames: {duplicates[:20]}")

    expected_columns: Optional[list[str]] = None
    for name in names:
        feature_path = _resolve_training_file(directory, name, field='filename')
        _n_rows, columns = validate_feature_hdf5_file(
            feature_path,
            sequence_length=sequence_length,
            expected_columns=expected_columns,
            scan_values=scan_values,
        )
        if expected_columns is None:
            expected_columns = columns
    return metadata.copy()


def _to_str_list(x) -> list[str]:
    """Coerce HDF5 attribute/dataset arrays into a Python list[str]."""
    arr = np.array(x)
    out: list[str] = []
    for item in arr.flatten().tolist():
        if isinstance(item, bytes):
            out.append(item.decode('utf-8', errors='ignore'))
        else:
            out.append(str(item))
    return out


def _read_headers_from_h5file(h5f: h5py.File, *, dataset_key: str = 'data') -> list[str]:
    """Read feature/column names from an already-open HDF5 file.

    Column metadata is read from file attrs['columns'], dataset attrs['columns'],
    or the root dataset 'columns'. If missing, generic col_0..col_{F-1} names
    are returned.
    """
    if dataset_key not in h5f:
        raise KeyError(f"HDF5 file is missing dataset '{dataset_key}'")
    dset = h5f[dataset_key]
    if len(dset.shape) != 2:
        raise ValueError(f"Expected '{dataset_key}' to be 2D, got shape={dset.shape}")
    num_features = int(dset.shape[1])

    cols = None
    if 'columns' in h5f.attrs:
        cols = h5f.attrs['columns']
    elif 'columns' in dset.attrs:
        cols = dset.attrs['columns']
    elif 'columns' in h5f:
        cols = h5f['columns'][()]
    if cols is None:
        return [f"col_{i}" for i in range(num_features)]

    headers = _to_str_list(cols)
    if len(headers) != num_features:
        # Corrupt metadata; fall back to generic names.
        return [f"col_{i}" for i in range(num_features)]
    return headers


def _build_keep_indices(file_headers: list[str], included_columns: list[str]) -> list[int]:
    """Build the column indices to keep, mapping by column name."""
    # Guard against duplicates (ambiguous mapping)
    if len(set(file_headers)) != len(file_headers):
        raise ValueError("Duplicate column names found in HDF5 header metadata; cannot map features by name.")
    col_to_idx = {c: i for i, c in enumerate(file_headers)}
    missing = [c for c in included_columns if c not in col_to_idx]
    if missing:
        raise KeyError(
            "Missing required columns in HDF5 file: "
            + (", ".join(missing[:20]) + (" ..." if len(missing) > 20 else ""))
        )
    return [int(col_to_idx[c]) for c in included_columns]

class KeypointsDataset(Dataset):
    """Load feature windows with center-frame supervision or inference indices.

    Use current class dictionaries for training labels and preserve original
    movie-frame indices during inference. Worker-enabled runs share disk-backed
    arrays; runs without workers hold features in RAM.
    """
    def __init__(
        self,
        directory,
        metadata,
        sequence_length,
        num_workers,
        pose_pred,
        orient_pred,
        recon,
        inference_mode,
        feature_plan: Optional["FeaturePlan"] = None,
        ablation_strict: bool = True,
    ):
        self.sequence_length = sequence_length
        self.metadata = metadata
        self.directory = directory
        self.pose_pred = pose_pred
        self.orient_pred = orient_pred
        self.recon = recon
        self.inference_mode = inference_mode
        self.num_workers = multiprocessing.cpu_count() if num_workers is None else int(num_workers)
        self._file_timelines = {}

        def store_array(path, values):
            values = np.ascontiguousarray(values, dtype=np.float32)
            if isinstance(self.data_in_memory, DiskFeatureStore):
                self.data_in_memory.add(path, values)
            else:
                self.data_in_memory[path] = values

        self.feature_plan = feature_plan
        self.ablation_strict = bool(ablation_strict)
        self.included_columns: Optional[list[str]] = None
        self.schema_hash: Optional[str] = None
        if self.feature_plan is not None:
            self.included_columns = list(self.feature_plan.included_columns)
            self.schema_hash = str(self.feature_plan.schema_hash)

        if self.inference_mode:
            print("Loading HDF5 files into RAM and creating inference instructions...")
            self.h5_files = metadata
            self.data_in_memory = DiskFeatureStore() if self.num_workers > 0 else {}
            segments = []
            
            for h5_file in self.h5_files:
                h5_file_path = os.path.join(self.directory, h5_file)
                try:
                    with h5py.File(h5_file_path, 'r') as h5f:
                        data_np = h5f['data'][:]
                        total, source_frames, complete = h5_frame_timeline(h5f)
                        self._file_timelines[h5_file_path] = (total, source_frames, complete)

                        if self.included_columns is not None:
                            file_headers = _read_headers_from_h5file(h5f, dataset_key='data')
                            keep_idx = _build_keep_indices(file_headers, self.included_columns)
                            data_np = data_np[:, keep_idx]

                        if not np.isfinite(data_np).all():
                            raise ValueError(f'Nonfinite selected inference features in {h5_file_path}')
                        store_array(h5_file_path, data_np)
                    segments.append(self._create_instructions_for_inference(h5_file_path, self.sequence_length))
                except Exception as e:
                    msg = f"Warning: Could not process {h5_file} for inference. Error: {e}. Skipping."
                    if self.inference_mode or self.ablation_strict:
                        raise
                    print(msg)
            self.dataset_instructions = WindowInstructions(self, segments)
            print("Feature storage and compact inference index prepared.")

        else:
            print("Initializing data_in_memory...")
            self.data_in_memory = DiskFeatureStore() if self.num_workers > 0 else {}
            print("Loading HDF5 files into RAM...")

            bad_filenames: set[str] = set()
            for _, row in metadata.iterrows():
                input_file_path = os.path.join(self.directory, row['filename'])
                if row['filename'] in bad_filenames:
                    continue
                if input_file_path not in self.data_in_memory:
                    try:
                        with h5py.File(input_file_path, 'r') as h5f:
                            data_np = h5f['data'][:]
                            if not np.isfinite(data_np).all():
                                raise ValueError(f'Nonfinite training features in {input_file_path}')
                            self._file_timelines[input_file_path] = h5_frame_timeline(h5f)

                            if self.included_columns is not None:
                                file_headers = _read_headers_from_h5file(h5f, dataset_key='data')
                                keep_idx = _build_keep_indices(file_headers, self.included_columns)
                                data_np = data_np[:, keep_idx]

                            store_array(input_file_path, data_np)
                    except Exception as e:
                        if self.ablation_strict:
                            raise
                        bad_filenames.add(row['filename'])
                        if input_file_path in self.data_in_memory:
                            if isinstance(self.data_in_memory, dict):
                                del self.data_in_memory[input_file_path]
                        print(f"Warning: Skipping file due to feature-selection load error: {row['filename']} ({e})")
            print("Data loaded into RAM.")

            if bad_filenames:
                # Filter metadata so we never create instructions for skipped files.
                kept = metadata[~metadata['filename'].isin(bad_filenames)].copy()
                kept = kept.reset_index(drop=True)
                self.metadata = kept
                print(f"Non-strict mode: skipped {len(bad_filenames)} files due to missing/invalid columns.")
            else:
                self.metadata = metadata

            # Create dataset instructions sequentially to avoid multiprocessing memory spikes
            print("Creating dataset instructions sequentially...")
            rows_as_tuples = [(row.filename, row.labels) for row in self.metadata.itertuples(index=False)]
            self.dataset_instructions = WindowInstructions(
                self, [self._create_instructions(row_tuple) for row_tuple in rows_as_tuples])
            print("Instructions created.")

            self.pose_dict = {}
            self.orient_dict = {}
            if self.pose_pred or self.orient_pred:
                all_behaviors = set()
                all_orientations = set()
                
                for segment in self.dataset_instructions.segments:
                    if self.pose_pred:
                        all_behaviors.update(map(str, np.unique(segment['pose_labels'])))
                    if self.orient_pred:
                        all_orientations.update(map(str, np.unique(segment['orient_labels'])))
                
                if self.pose_pred:
                    for label in sorted(all_behaviors):
                        if label == 'nan' or label.lower() == 'nan':
                            print(f"Warning: 'nan' label found in behaviors: {label}")
                        self.pose_dict[label] = len(self.pose_dict)
                        print(f"Added to pose_dict: {label} -> {self.pose_dict[label]}")
                
                if self.orient_pred:
                    for label in sorted(all_orientations):
                        if label == 'nan' or label.lower() == 'nan':
                            print(f"Warning: 'nan' label found in orientations: {label}")
                        self.orient_dict[label] = len(self.orient_dict)
                        print(f"Added to orient_dict: {label} -> {self.orient_dict[label]}")

                self._add_indices_to_instructions()

            print(f"Number of dataset instructions: {len(self.dataset_instructions)}")

        if num_workers is None:
            num_workers = multiprocessing.cpu_count()
        self.num_workers = num_workers

    def _create_instructions(self, row):
        """Build valid training-window segments from one file's center-frame labels."""
        filename, labels = row  
        input_file_path = os.path.join(self.directory, filename)
        label_file_path = os.path.join(self.directory, labels)

        _, source_frames, _ = self._file_timelines[input_file_path]
        num_rows = len(source_frames)
        num_sequences = max(0, num_rows - (self.sequence_length - 1))

        with h5py.File(label_file_path, 'r') as label_h5f:
            _validate_label_pair(input_file_path, label_h5f)
            all_pose_labels = label_h5f['Behavior'][:] if self.pose_pred else None
            all_orient_labels = label_h5f['Orientation'][:] if self.orient_pred else None
            for values in (all_pose_labels,all_orient_labels):
                if values is not None and len(values) != num_rows:
                    raise ValueError(f'Paired training labels and feature rows differ: {input_file_path}')
            if 'source_frame_indices' in label_h5f and not np.array_equal(label_h5f['source_frame_indices'][:],source_frames):
                raise ValueError(f'Paired training labels have different original-frame indices: {input_file_path}')

        middle = (self.sequence_length - 1) // 2
        valid = np.ones(num_sequences, dtype=bool)
        selected = {}
        for head, labels in (('pose', all_pose_labels), ('orient', all_orient_labels)):
            if labels is None:
                selected[head + '_labels'] = None
                continue
            raw_values = [value.decode('utf-8') if isinstance(value, bytes) else str(value)
                          for value in labels[middle:middle + num_sequences]]
            values = np.array([_canonical_label(value) for value in raw_values], dtype='U')
            changed = sum(raw != clean for raw, clean in zip(raw_values, values))
            if changed:
                logging.warning('Canonicalized whitespace/missingness in %d %s labels: %s', changed, head, label_file_path)
            valid &= (np.char.lower(values) != 'nan') & (values != '')
            selected[head + '_labels'] = values
        starts = np.flatnonzero(valid).astype(np.int64)
        for key, values in selected.items():
            if values is not None:
                selected[key] = values[valid]
        return {'path': input_file_path, 'count': len(starts),
                'starts': None if valid.all() else starts, **selected}

    def _add_indices_to_instructions(self):
        """Map instruction labels through the current class dictionaries."""
        if isinstance(self.dataset_instructions, WindowInstructions):
            return  # Views resolve through the current validated dictionaries.
        for instruction in self.dataset_instructions:
            pose_label = instruction['pose_label']
            orient_label = instruction['orient_label']
            instruction['pose_index'] = self.pose_dict.get(pose_label, -1)  # Use -1 to indicate missing
            instruction['orient_index'] = self.orient_dict.get(orient_label, -1)
            
    def _create_instructions_for_inference(self, h5_file_path, sequence_length):
        """Build all complete sliding windows with original source-frame alignment."""
        if h5_file_path in self._file_timelines:
            _, source_frames, _ = self._file_timelines[h5_file_path]
        else:
            with h5py.File(h5_file_path, 'r') as h5f:
                _, source_frames, _ = h5_frame_timeline(h5f)
        num_rows = len(source_frames)

        num_sequences = max(0, num_rows - (sequence_length - 1))
        return {'path': h5_file_path, 'count': num_sequences, 'starts': None,
                'source_start': int(source_frames[0]) if num_rows else 0}

    def __len__(self):
        """Return the number of available sequence windows."""
        return len(self.dataset_instructions)

    def __getitem__(self, idx):
        """Return a float32 (T, F) feature window and its supervision or identity.

        Training returns (window, pose_index, orient_index, global_instruction_index).
        Inference returns (window, filename, zero_based_source_center_index).
        """
        if self.inference_mode:
            instruction = self.dataset_instructions[idx]

            sequence_data = self.data_in_memory[instruction['input_file']]

            seq_np = sequence_data[instruction['sequence_start']:instruction['sequence_end']]
            sequence_tensor = torch.from_numpy(seq_np).float()

            return sequence_tensor, instruction['input_file'], instruction['middle_idx']
        
        else:
            instruction = self.dataset_instructions[idx]

            sequence_data = self.data_in_memory[instruction['input_file']][instruction['sequence_start']:instruction['sequence_end']]
            sequence_tensor = torch.from_numpy(sequence_data).float()

            # Return labels as Python ints (DataLoader will collate to int64 tensors).
            pose_index = int(instruction['pose_index'])
            orient_index = int(instruction['orient_index'])

            return sequence_tensor, pose_index, orient_index, idx

    def one_hot_encode_label(self, label, label_dict):
        """Encode a known label as one-hot; return zeros for an unknown label."""
        one_hot = np.zeros(len(label_dict))
        if label in label_dict:
            one_hot[label_dict[label]] = 1
        return one_hot


def build_dataloader(
    dataset: torch.utils.data.Dataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    *,
    pin_memory: bool = True,
    drop_last: bool = False,
    seed: Optional[int] = None,
    prefetch_factor: int = 4
) -> DataLoader:
    """Build a DataLoader with worker settings valid for the requested worker count.

    Enable prefetching and persistent workers only when workers are present.
    Use the supplied seed for deterministic shuffled order.
    """
    dl_kwargs = dict(
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )

    if num_workers > 0:
        dl_kwargs.update(
            prefetch_factor=prefetch_factor,
            persistent_workers=True,
            worker_init_fn=seed_worker,
        )

    # Deterministic shuffling when requested.
    if seed is not None and shuffle:
        g = torch.Generator()
        g.manual_seed(int(seed))
        dl_kwargs["generator"] = g

    return register_dataloader(DataLoader(dataset, **dl_kwargs))
