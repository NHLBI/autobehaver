"""I/O helpers for Autobehaver.

Includes utilities for discovering input files, reading HDF5 feature arrays, and extracting basic
metadata required to configure the model and data pipeline."""

from __future__ import annotations

import os

import h5py
import numpy as np


__all__ = ['get_headers_from_first_hdf5', 'get_num_features_from_first_hdf5']


def get_headers_from_first_hdf5(input_folder: str) -> list[str]:
    """
    Retrieves feature/column names from the first suitable HDF5 file in the specified folder.

    IMPORTANT:
    - The behavior model expects the HDF5 file to contain a 2D dataset named 'data' with shape (T, F).
    - The feature (column) names are expected to be stored under the key 'columns'.

    This function is used to determine feature/column names so downstream code can apply augmentation
    policies based on feature semantics.

    If no column-name metadata is found, this function will fall back to returning
    generic names ["col_0", ..., "col_{F-1}"] and will print a warning. In that case,
    any name-based feature-selection policy may be less accurate.

    Parameters:
    - input_folder: Directory containing HDF5 files.

    Returns:
    - list[str]: Feature names of length F.
    """
    files = os.listdir(input_folder)
    hdf5_files = sorted(
        f for f in files
        if (f.endswith('.h5') or f.endswith('.hdf5')) and ('metadata' not in f)
    )
    if not hdf5_files:
        raise FileNotFoundError("No suitable HDF5 files found in the input folder.")

    def _to_str_list(x) -> list[str]:
        arr = np.array(x)
        out: list[str] = []
        for item in arr.flatten().tolist():
            if isinstance(item, bytes):
                out.append(item.decode('utf-8', errors='ignore'))
            else:
                out.append(str(item))
        return out

    for fname in hdf5_files:
        fpath = os.path.join(input_folder, fname)
        try:
            with h5py.File(fpath, 'r') as h5f:
                if 'data' not in h5f:
                    continue
                dset = h5f['data']
                if len(dset.shape) != 2:
                    continue
                num_features = int(dset.shape[1])

                cols = None
                if 'columns' in h5f.attrs:
                    cols = h5f.attrs['columns']
                elif 'columns' in dset.attrs:
                    cols = dset.attrs['columns']
                elif 'columns' in h5f:
                    cols = h5f['columns'][()]

                if cols is not None:
                    headers = _to_str_list(cols)
                    if len(headers) != num_features:
                        print(
                            f"WARNING: Found column metadata in {fname} but length {len(headers)} "
                            f"does not match data.shape[1]={num_features}. Falling back to generic names."
                        )
                        return [f"col_{i}" for i in range(num_features)]
                    return headers
        except Exception:
            continue

    for fname in hdf5_files:
        fpath = os.path.join(input_folder, fname)
        try:
            with h5py.File(fpath, 'r') as h5f:
                if 'data' not in h5f:
                    continue
                num_features = int(h5f['data'].shape[1])
                print(
                    "WARNING: No feature-name metadata found in HDF5 files. "
                    "Falling back to generic column names col_0..col_{F-1}. "
                    "Scaling selection by name will likely be empty."
                )
                return [f"col_{i}" for i in range(num_features)]
        except Exception:
            continue

    raise FileNotFoundError("Could not find any suitable HDF5 file containing a 2D 'data' dataset.")


def get_num_features_from_first_hdf5(input_folder: str) -> int:
    """
    Determines the number of features (columns) from the first suitable HDF5 file in the specified folder.

    This function scans the folder for HDF5 files and returns the second dimension (F) of the 2D
    dataset named 'data' with shape (T, F).

    Parameters:
    - input_folder: str
        Path to directory containing HDF5 files.

    Returns:
    - int: number of features (F).

    Raises:
    - FileNotFoundError: If no suitable HDF5 file with a 2D 'data' dataset is found.
    """
    files = os.listdir(input_folder)
    hdf5_files = sorted(
        f for f in files
        if (f.endswith('.h5') or f.endswith('.hdf5')) and ('metadata' not in f)
    )
    if not hdf5_files:
        raise FileNotFoundError("No suitable HDF5 files found in the input folder.")

    for fname in hdf5_files:
        fpath = os.path.join(input_folder, fname)
        try:
            with h5py.File(fpath, 'r') as h5f:
                if 'data' not in h5f:
                    continue
                dset = h5f['data']
                if len(dset.shape) != 2:
                    continue
                return int(dset.shape[1])
        except Exception:
            continue

    raise FileNotFoundError("Could not find any suitable HDF5 file containing a 2D 'data' dataset.")
