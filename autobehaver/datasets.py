"""Dataset and DataLoader helpers for Autobehaver.

Includes dataset classes for loading per-frame engineered features from HDF5, building temporal
windows, and returning inputs/labels for training, validation, and inference."""

from __future__ import annotations

import multiprocessing
import os
from typing import Optional, TYPE_CHECKING

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .utils import seed_worker

if TYPE_CHECKING:  # pragma: no cover
    from .feature_selection import FeaturePlan

__all__ = ['KeypointsDataset', 'build_dataloader']


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
    """
    PyTorch Dataset for loading sequential keypoint data with pose/orientation labels.
    
    Supports two modes:
    - Training: Loads data + labels from metadata, builds class dictionaries  
    - Inference: Loads all HDF5 files with sliding window coverage
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

        # Feature selection / ablation
        self.feature_plan = feature_plan
        self.ablation_strict = bool(ablation_strict)
        self.included_columns: Optional[list[str]] = None
        self.schema_hash: Optional[str] = None
        if self.feature_plan is not None:
            self.included_columns = list(self.feature_plan.included_columns)
            self.schema_hash = str(self.feature_plan.schema_hash)

        if self.inference_mode:
            print("Loading HDF5 files into RAM and creating inference instructions...")
            # During inference mode, metadata will be a single file that gets passed in one at a time from a master list
            self.h5_files = metadata
            self.data_in_memory = {}
            self.dataset_instructions = []
            
            # Load all HDF5 files into RAM and create instructions
            for h5_file in self.h5_files:
                h5_file_path = os.path.join(self.directory, h5_file)
                try:
                    with h5py.File(h5_file_path, 'r') as h5f:
                        data_np = h5f['data'][:]

                        # Apply feature selection (if configured)
                        if self.included_columns is not None:
                            file_headers = _read_headers_from_h5file(h5f, dataset_key='data')
                            keep_idx = _build_keep_indices(file_headers, self.included_columns)
                            data_np = data_np[:, keep_idx]

                        self.data_in_memory[h5_file_path] = data_np
                    self.dataset_instructions.extend(self._create_instructions_for_inference(h5_file_path, self.sequence_length))
                except Exception as e:
                    msg = f"Warning: Could not process {h5_file} for inference. Error: {e}. Skipping."
                    if self.ablation_strict:
                        # In strict mode, propagate the error to fail fast.
                        raise
                    print(msg)
            print("HDF5 files loaded into RAM and instructions created for inference.")

        else:
            # Load all HDF5 files into RAM for training
            print("Initializing data_in_memory...")
            self.data_in_memory = {}
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

                            # Apply feature selection (if configured)
                            if self.included_columns is not None:
                                file_headers = _read_headers_from_h5file(h5f, dataset_key='data')
                                keep_idx = _build_keep_indices(file_headers, self.included_columns)
                                data_np = data_np[:, keep_idx]

                            self.data_in_memory[input_file_path] = data_np
                    except Exception as e:
                        if self.ablation_strict:
                            raise
                        # Non-strict: skip this file entirely
                        bad_filenames.add(row['filename'])
                        if input_file_path in self.data_in_memory:
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
            self.dataset_instructions = [
                instruction for row_tuple in rows_as_tuples
                for instruction in self._create_instructions(row_tuple)
            ]
            print("Instructions created.")

            # Initialize pose and orientation dictionaries if needed
            self.pose_dict = {}
            self.orient_dict = {}
            if self.pose_pred or self.orient_pred:
                all_behaviors = set()
                all_orientations = set()
                
                for instruction in self.dataset_instructions:
                    if self.pose_pred:
                        all_behaviors.add(instruction['pose_label'])
                    if self.orient_pred:
                        all_orientations.add(instruction['orient_label'])
                
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

                # Add indices to instructions
                self._add_indices_to_instructions()

            # Print dataset-instruction counts.
            print(f"Number of dataset instructions: {len(self.dataset_instructions)}")

        # Set up multiprocessing workers
        if num_workers is None:
            num_workers = multiprocessing.cpu_count()
        self.num_workers = num_workers

    def _create_instructions(self, row):
        """Creates sequence instructions for a single file during training setup."""
        filename, labels = row  # Unpack simple tuple
        input_file_path = os.path.join(self.directory, filename)
        label_file_path = os.path.join(self.directory, labels)

        with h5py.File(input_file_path, 'r') as h5f:
            num_rows = h5f['data'].shape[0]
            num_sequences = num_rows - (self.sequence_length - 1)

        # Load all labels for the current file ONCE for efficiency
        with h5py.File(label_file_path, 'r') as label_h5f:
            all_pose_labels = label_h5f['Behavior'][:] if self.pose_pred else None
            all_orient_labels = label_h5f['Orientation'][:] if self.orient_pred else None

        instructions = []
        for start_idx in range(num_sequences):
            end_idx = start_idx + self.sequence_length
            label_idx = start_idx + (self.sequence_length - 1) // 2

            # Read from the in-memory arrays instead of opening the file repeatedly
            pose_label = str(all_pose_labels[label_idx].decode('utf-8')) if self.pose_pred else None
            orient_label = str(all_orient_labels[label_idx].decode('utf-8')) if self.orient_pred else None

            pose_is_invalid = self.pose_pred and (pose_label.lower() == 'nan' or pose_label == '')
            orient_is_invalid = self.orient_pred and (orient_label.lower() == 'nan' or orient_label == '')

            # Skip sequences with missing or 'nan' labels for active predictions
            if pose_is_invalid or orient_is_invalid:
                continue

            instruction = {
                'input_file': input_file_path,
                'sequence_start': start_idx,
                'sequence_end': end_idx,
                'pose_label': pose_label,
                'orient_label': orient_label
            }
            instructions.append(instruction)

        return instructions

    def _add_indices_to_instructions(self):
        """Converts string labels to integer indices using built class dictionaries."""
        for instruction in self.dataset_instructions:
            pose_label = instruction['pose_label']
            orient_label = instruction['orient_label']
            instruction['pose_index'] = self.pose_dict.get(pose_label, -1)  # Use -1 to indicate missing
            instruction['orient_index'] = self.orient_dict.get(orient_label, -1)
            
    def _create_instructions_for_inference(self, h5_file_path, sequence_length):
        """Creates sliding window sequence instructions for inference mode."""
        instructions = []
        
        # Read the HDF5 file to determine the number of rows
        with h5py.File(h5_file_path, 'r') as h5f:
            num_rows = h5f['data'].shape[0]

        # Calculate the number of sequences
        num_sequences = num_rows - (sequence_length - 1)

        # Generate instructions for each sequence
        for start_idx in range(num_sequences):
            end_idx = start_idx + sequence_length
            middle_idx = start_idx + (sequence_length // 2)
            instruction = {
                'input_file': h5_file_path,
                'sequence_start': start_idx,
                'sequence_end': end_idx,
                'middle_idx': middle_idx
            }
            instructions.append(instruction)

        return instructions

    def __len__(self):
        """Returns total number of sequence instructions available."""
        return len(self.dataset_instructions)

    def __getitem__(self, idx):
        """
        Returns sequence data for given index.
        
        Training mode: (sequence_tensor, pose_tensor, orient_tensor, idx)
        Inference mode: (sequence_tensor, filename, middle_frame_idx)
        """
        if self.inference_mode:
            # Retrieve precomputed instructions for the given index
            instruction = self.dataset_instructions[idx]

            # Access the sequence data from RAM using the full file path
            sequence_data = self.data_in_memory[instruction['input_file']]

            # Extract the sequence defined by the start and end indices
            seq_np = sequence_data[instruction['sequence_start']:instruction['sequence_end']]
            sequence_tensor = torch.from_numpy(seq_np).float()

            # Return the sequence tensor, filename, and middle index
            return sequence_tensor, instruction['input_file'], instruction['middle_idx']
        
        else:
            # Retrieve precomputed instructions for the given index
            instruction = self.dataset_instructions[idx]

            # Access the sequence data from RAM
            sequence_data = self.data_in_memory[instruction['input_file']][instruction['sequence_start']:instruction['sequence_end']]
            sequence_tensor = torch.from_numpy(sequence_data).float()

            # Return labels as Python ints (DataLoader will collate to int64 tensors).
            pose_index = int(instruction['pose_index'])
            orient_index = int(instruction['orient_index'])

            return sequence_tensor, pose_index, orient_index, idx

    def one_hot_encode_label(self, label, label_dict):
        """Converts string label to one-hot encoded vector using provided dictionary."""
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
    """
    Build a PyTorch DataLoader with safe defaults.

    Fixes the common PyTorch gotcha:
      - `prefetch_factor` and `persistent_workers` are only valid when `num_workers > 0`.

    Also optionally attaches a deterministic `torch.Generator` for reproducible shuffling
    when a global random seed is provided.

    Parameters
    ----------
    dataset : Dataset
        Dataset instance.
    batch_size : int
        Batch size.
    shuffle : bool
        Whether to shuffle.
    num_workers : int
        Number of DataLoader workers.
    pin_memory : bool
        Whether to pin memory.
    drop_last : bool
        Whether to drop last incomplete batch.
    seed : Optional[int]
        If provided and shuffle=True, use this seed for deterministic shuffling.
    prefetch_factor : int
        Number of batches prefetched by each worker when `num_workers > 0`.

    Returns
    -------
    DataLoader
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

    return DataLoader(dataset, **dl_kwargs)


