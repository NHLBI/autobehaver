"""Inference pipeline for pose/orientation prediction.

Loads one or more trained checkpoints, runs prediction over all files in a directory, applies
optional smoothing, and writes per-frame inference CSV outputs. Supports ensembles and test-time
augmentation."""

from __future__ import annotations

import os
import shutil
import sys
import time
from datetime import datetime
from typing import List, Optional, Tuple

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .model import PosePredTransformer
from .datasets import KeypointsDataset
from .checkpoint import load_checkpoint_model
from .augmentation import augment_features, select_columns_for_augmentation
from .feature_selection import FeaturePlan, compute_schema_hash

__all__ = ['_validate_class_dict', '_labels_by_index_from_class_dict', '_build_reorder_indices', 'check_ensemble_compatibility_and_get_permutations', 'run_inference', 'prepare_data_inference', 'load_model_for_inference', 'perform_inference', 'smooth_bouts_vectorized']

def _validate_class_dict(class_dict: dict, dict_name: str, model_name: str) -> None:
    """Validate that a class dictionary is well-formed (string->int, contiguous indices)."""
    if not isinstance(class_dict, dict) or len(class_dict) == 0:
        raise ValueError(f"{model_name}: Missing or empty {dict_name}_dict in checkpoint. Cannot ensemble safely.")

    # Basic type checks
    for k, v in class_dict.items():
        if not isinstance(k, str):
            raise ValueError(f"{model_name}: {dict_name}_dict has a non-string key: {repr(k)}")
        if not isinstance(v, int):
            raise ValueError(f"{model_name}: {dict_name}_dict has a non-int value for key '{k}': {repr(v)}")

    # Index integrity checks (unique + contiguous 0..C-1)
    values = list(class_dict.values())
    if len(set(values)) != len(values):
        raise ValueError(f"{model_name}: {dict_name}_dict indices are not unique. Found duplicates.")

    expected = set(range(len(class_dict)))
    actual = set(values)
    if actual != expected:
        raise ValueError(
            f"{model_name}: {dict_name}_dict indices must be exactly {sorted(expected)} but found {sorted(actual)}. "
            "This usually indicates an invalid checkpoint."
        )


def _labels_by_index_from_class_dict(class_dict: dict[str, int], dict_name: str, model_name: str) -> List[str]:
    """Return a list of labels ordered by their class index."""
    _validate_class_dict(class_dict, dict_name=dict_name, model_name=model_name)

    labels_by_index: List[Optional[str]] = [None] * len(class_dict)
    for lbl, idx in class_dict.items():
        if idx < 0 or idx >= len(class_dict):
            raise ValueError(f"{model_name}: {dict_name}_dict has out-of-range index {idx} for label '{lbl}'.")
        if labels_by_index[idx] is not None:
            raise ValueError(f"{model_name}: {dict_name}_dict has duplicate index {idx} (label '{lbl}').")
        labels_by_index[idx] = lbl

    if any(x is None for x in labels_by_index):
        raise ValueError(f"{model_name}: {dict_name}_dict is missing one or more indices after validation.")

    return [x for x in labels_by_index if x is not None]


def _build_reorder_indices(
    reference_dict: dict[str, int],
    model_dict: dict[str, int],
    dict_name: str,
    model_name: str
) -> List[int]:
    """
    Build a column-reorder index list so that:
        logits_aligned = logits_model[:, reorder]
    will be aligned to the reference dictionary's index ordering.
    """
    ref_labels_by_index = _labels_by_index_from_class_dict(reference_dict, dict_name=dict_name, model_name="Reference")
    _validate_class_dict(model_dict, dict_name=dict_name, model_name=model_name)

    # Ensure label sets match
    ref_labels = set(reference_dict.keys())
    model_labels = set(model_dict.keys())
    if ref_labels != model_labels:
        missing = sorted(ref_labels - model_labels)
        extra = sorted(model_labels - ref_labels)
        raise ValueError(
            f"{model_name}: {dict_name}_dict label set mismatch vs reference. "
            f"Missing={missing[:10]}{'...' if len(missing) > 10 else ''} "
            f"Extra={extra[:10]}{'...' if len(extra) > 10 else ''}"
        )

    reorder = [model_dict[lbl] for lbl in ref_labels_by_index]

    # Validate it's a true permutation
    expected = set(range(len(reference_dict)))
    if set(reorder) != expected or len(reorder) != len(reference_dict):
        raise ValueError(
            f"{model_name}: Computed {dict_name} reorder indices are not a valid permutation. Got={reorder}"
        )

    return reorder


def check_ensemble_compatibility_and_get_permutations(
    checkpoint_paths: List[str],
    reference_pose_dict: dict[str, int],
    reference_orient_dict: dict[str, int],
) -> Tuple[dict, List[Optional[List[int]]], List[Optional[List[int]]]]:
    """
    Loads all checkpoints (CPU) and validates that they are ensemble compatible.

    Why this exists:
      - Ensembling assumes 'logit column i' means the SAME class for every model.
      - If two checkpoints use different label->index mappings, averaging logits silently corrupts results.

    This function enforces compatibility and returns per-model permutations that align each model's
    logits into the *reference* class-index ordering.

    Returns:
      reference_global_vars,
      pose_permutations: list aligned with checkpoint_paths, each is None (identity) or a list[int]
      orient_permutations: same as above
    """
    if not checkpoint_paths:
        raise ValueError("No model checkpoints were provided for inference.")

    # Load all checkpoints to CPU and collect minimal metadata
    metas = []
    for p in checkpoint_paths:
        if not os.path.isfile(p):
            raise FileNotFoundError(f"Inference checkpoint not found: {p}")
        ckpt = torch.load(p, map_location='cpu')
        gv = ckpt.get('global_vars', {})
        if not gv:
            raise ValueError(f"{os.path.basename(p)}: Missing 'global_vars' in checkpoint. Cannot validate ensemble compatibility.")
        metas.append({
            'path': p,
            'name': os.path.basename(p),
            'global_vars': gv,
            'pose_dict': ckpt.get('pose_dict', {}),
            'orient_dict': ckpt.get('orient_dict', {}),
        })

    ref_gv = metas[0]['global_vars']
    ref_name = metas[0]['name']

    # Reference flags / shapes that MUST match for this script's inference pipeline
    ref_pose_pred = bool(ref_gv.get('pose_pred', False))
    ref_orient_pred = bool(ref_gv.get('orient_pred', False))
    ref_sequence_length = ref_gv.get('sequence_length', None)
    ref_num_features = ref_gv.get('num_features', None)

    # Feature schema validation (prevents silent semantic corruption when num_features matches but
    # the *meaning/order* of columns differs).
    ref_schema_hash = ref_gv.get('feature_schema_hash', None)
    if ref_schema_hash is None:
        ref_inc = ref_gv.get('feature_columns_included', None)
        if isinstance(ref_inc, (list, tuple)) and len(ref_inc) > 0:
            ref_schema_hash = compute_schema_hash([str(x) for x in ref_inc])

    # Sanity: if reference says it predicts a head, we require a non-empty dict
    if ref_pose_pred and len(reference_pose_dict) == 0:
        raise ValueError(f"Reference model '{ref_name}' has pose_pred=True but pose_dict is empty.")
    if ref_orient_pred and len(reference_orient_dict) == 0:
        raise ValueError(f"Reference model '{ref_name}' has orient_pred=True but orient_dict is empty.")

    # Validate the reference dicts themselves
    if ref_pose_pred:
        _validate_class_dict(reference_pose_dict, dict_name='pose', model_name='Reference')
    if ref_orient_pred:
        _validate_class_dict(reference_orient_dict, dict_name='orient', model_name='Reference')

    pose_perms: List[Optional[List[int]]] = []
    orient_perms: List[Optional[List[int]]] = []

    print("\n--- Ensemble compatibility check ---")
    print(f"Reference checkpoint: {ref_name}")
    print(
        f"  pose_pred={ref_pose_pred}, orient_pred={ref_orient_pred}, sequence_length={ref_sequence_length}, "
        f"num_features={ref_num_features}, feature_schema_hash={ref_schema_hash}"
    )

    for meta in metas:
        name = meta['name']
        gv = meta['global_vars']

        # Hard requirements (shape + enabled heads)
        if bool(gv.get('pose_pred', False)) != ref_pose_pred:
            raise ValueError(
                f"Ensemble mismatch: '{name}' pose_pred={gv.get('pose_pred', False)} but reference '{ref_name}' pose_pred={ref_pose_pred}."
            )
        if bool(gv.get('orient_pred', False)) != ref_orient_pred:
            raise ValueError(
                f"Ensemble mismatch: '{name}' orient_pred={gv.get('orient_pred', False)} but reference '{ref_name}' orient_pred={ref_orient_pred}."
            )
        if gv.get('sequence_length', None) != ref_sequence_length:
            raise ValueError(
                f"Ensemble mismatch: '{name}' sequence_length={gv.get('sequence_length', None)} but reference '{ref_name}' sequence_length={ref_sequence_length}. "
                "(All ensemble members must use the same sequence_length because they all receive the same input windows.)"
            )
        if gv.get('num_features', None) != ref_num_features:
            raise ValueError(
                f"Ensemble mismatch: '{name}' num_features={gv.get('num_features', None)} but reference '{ref_name}' num_features={ref_num_features}. "
                "(All ensemble members must ingest the same feature dimension.)"
            )

        # Feature schema hash (optional but recommended; enforce if reference has it)
        gv_schema_hash = gv.get('feature_schema_hash', None)
        if gv_schema_hash is None:
            inc = gv.get('feature_columns_included', None)
            if isinstance(inc, (list, tuple)) and len(inc) > 0:
                gv_schema_hash = compute_schema_hash([str(x) for x in inc])

        if ref_schema_hash is not None:
            if gv_schema_hash is None:
                raise ValueError(
                    f"Ensemble mismatch: reference '{ref_name}' includes feature_schema_hash but '{name}' does not. "
                    "Cannot safely ensemble checkpoints with unknown feature semantics."
                )
            if str(gv_schema_hash) != str(ref_schema_hash):
                raise ValueError(
                    f"Ensemble mismatch: '{name}' feature_schema_hash={gv_schema_hash} but reference '{ref_name}' feature_schema_hash={ref_schema_hash}."
                )
        else:
            # If the reference lacks schema info but another checkpoint has it, refuse to proceed.
            if gv_schema_hash is not None:
                raise ValueError(
                    f"Ensemble mismatch: reference '{ref_name}' lacks feature schema metadata, but '{name}' includes it. "
                    "Choose a reference checkpoint with schema metadata or retrain for consistent checkpoint formats."
                )

        # Build permutations (identity -> None)
        if ref_pose_pred:
            model_pose_dict = meta['pose_dict']
            reorder = _build_reorder_indices(reference_pose_dict, model_pose_dict, dict_name='pose', model_name=name)
            if reorder == list(range(len(reference_pose_dict))):
                pose_perms.append(None)
            else:
                print(f"  -> '{name}' pose_dict index order differs from reference. Will reorder pose logits to align classes.")
                pose_perms.append(reorder)
        else:
            pose_perms.append(None)

        if ref_orient_pred:
            model_orient_dict = meta['orient_dict']
            reorder = _build_reorder_indices(reference_orient_dict, model_orient_dict, dict_name='orient', model_name=name)
            if reorder == list(range(len(reference_orient_dict))):
                orient_perms.append(None)
            else:
                print(f"  -> '{name}' orient_dict index order differs from reference. Will reorder orient logits to align classes.")
                orient_perms.append(reorder)
        else:
            orient_perms.append(None)

    print("Ensemble compatibility check passed.\n")
    return ref_gv, pose_perms, orient_perms


def _read_headers_from_hdf5_path(hdf5_path: str) -> list[str]:
    """Read feature headers from a specific HDF5 file used for inference."""
    def _to_str_list(x) -> list[str]:
        arr = np.array(x)
        out: list[str] = []
        for item in arr.flatten().tolist():
            if isinstance(item, bytes):
                out.append(item.decode('utf-8', errors='ignore'))
            else:
                out.append(str(item))
        return out

    with h5py.File(hdf5_path, 'r') as h5f:
        if 'data' not in h5f:
            raise FileNotFoundError(f"HDF5 file is missing required 'data' dataset: {hdf5_path}")
        dset = h5f['data']
        if len(dset.shape) != 2:
            raise ValueError(f"HDF5 'data' dataset must be 2D for inference: {hdf5_path}")
        num_features = int(dset.shape[1])

        cols = None
        if 'columns' in h5f.attrs:
            cols = h5f.attrs['columns']
        elif 'columns' in dset.attrs:
            cols = dset.attrs['columns']
        elif 'columns' in h5f:
            cols = h5f['columns'][()]
        if cols is None:
            print(
                "WARNING: No feature-name metadata found in selected inference HDF5. "
                "Falling back to generic column names col_0..col_{F-1}. "
                "Scaling selection by name will likely be empty."
            )
            return [f"col_{i}" for i in range(num_features)]

        headers = _to_str_list(cols)
        if len(headers) != num_features:
            print(
                f"WARNING: Column metadata length {len(headers)} does not match data.shape[1]={num_features} "
                f"for {os.path.basename(hdf5_path)}. Falling back to generic names."
            )
            return [f"col_{i}" for i in range(num_features)]
        return headers



def _discover_inference_input_files(
    directory: str,
    *,
    explicit_files: Optional[List[str]] = None,
    file_filter: Optional[str] = None,
) -> List[str]:
    """Resolve the exact HDF5 files to process and guard against output-name collisions."""
    if explicit_files is not None:
        files: List[str] = []
        for p in explicit_files:
            fp = os.path.abspath(str(p))
            if not os.path.isfile(fp):
                raise FileNotFoundError(f"Explicit inference input not found: {fp}")
            lf = os.path.basename(fp).lower()
            if not ((lf.endswith('.h5') or lf.endswith('.hdf5')) and ('metadata' not in lf)):
                raise ValueError(f"Explicit inference input is not a supported normalized HDF5 file: {fp}")
            files.append(fp)
        file_list = sorted(set(files))
    else:
        file_list = sorted(
            os.path.join(directory, f)
            for f in os.listdir(directory)
            if ((f.endswith('.h5') or f.endswith('.hdf5')) and ('metadata' not in f.lower()))
        )

    if file_filter:
        needle = str(file_filter).lower()
        before = len(file_list)
        file_list = [p for p in file_list if needle in os.path.basename(p).lower()]
        print(f"Applied file_filter='{file_filter}': {len(file_list)}/{before} files selected.")

    stem_to_paths: dict[str, list[str]] = {}
    for p in file_list:
        stem = os.path.splitext(os.path.basename(p))[0]
        stem_to_paths.setdefault(stem, []).append(p)
    collisions = {stem: paths for stem, paths in stem_to_paths.items() if len(paths) > 1}
    if collisions:
        preview = []
        for stem, paths in sorted(collisions.items()):
            preview.append(f"{stem}: {paths[:3]}")
        raise ValueError(
            "Selected inference inputs would collide on output filenames because they share the same stem(s). "
            "Rename or process separately. Collisions: " + "; ".join(preview)
        )

    return file_list


def run_inference(
    directory: str,
    inference_model_paths: List[str],
    num_workers: int,
    batch: int,
    inf_average: int,
    sequence_length: int,
    min_pose_bout: int,
    min_orient_bout: int,
    use_tta: bool = False,
    tta_noise_level: float = 0.01,
    *,
    output_dir: Optional[str] = None,
    file_filter: Optional[str] = None,
    clobber: bool = True,
    archive_existing: bool = True,
    use_inferences_subdir: bool = True,
    explicit_files: Optional[List[str]] = None,
) -> None:
    """
    Orchestrates the complete inference pipeline for a trained transformer model.
    
    Execution Flow:
    
    1. **Dataset Preparation**
       - Load checkpoint to extract training configuration (sequence_length, prediction flags)
       - Create KeypointsDataset with inference mode enabled for sliding window coverage
       - Extract class dictionaries (pose_dict, orient_dict) for label mapping
    
    2. **Model Loading & Setup**
       - Load PosePredTransformer architecture from checkpoint parameters
       - Restore trained weights with proper class count configuration
       - Extract training parameters (pose_pred, orient_pred, recon) for consistency
       - Set model to evaluation mode and move to GPU if available
    
    3. **Data Processing Setup**
       - Create DataLoader with performance optimizations (pin_memory, persistent_workers)
       - Configure batch processing for efficient GPU utilization
       - Print diagnostic information about loaded class dictionaries
    
    4. **Inference Execution**
       - Run perform_inference() with multi-pass averaging for stability
       - Process all HDF5 files in directory with sliding window sequences
       - Generate predictions, probabilities, and context vectors per frame
       - Handle both pose classification and orientation prediction as configured
    
    5. **Results Export**
       - Save detailed CSV files to 'inferences' subdirectory with timestamps
       - Include frame numbers, logits, probabilities, predicted labels for each sequence
       - Archive previous inference results to prevent overwriting
    
    Key Features:
    - **Consistency Guarantee**: Uses exact training configuration from checkpoint
    - **Multi-pass Averaging**: Performs multiple forward passes per input for prediction averaging
    - **Complete Coverage**: Sliding window approach ensures all frames are processed
    - **Detailed Output**: Comprehensive CSV files with all prediction metrics
    - **Automatic Archiving**: Preserves previous inference results with timestamps
    """
    max_inference_cvs = len(inference_model_paths)

    # ----------------------------------------------------------------------------------
    # Output routing
    # ----------------------------------------------------------------------------------
    output_root = output_dir if output_dir else directory
    os.makedirs(output_root, exist_ok=True)

    if use_inferences_subdir:
        inference_dir = os.path.join(output_root, 'inferences')
        past_inferences_dir = os.path.join(inference_dir, 'past_inferences')
        details_dir_for_archive = os.path.join(inference_dir, 'inference_details')
    else:
        inference_dir = output_root
        past_inferences_dir = os.path.join(output_root, 'past_inferences')
        details_dir_for_archive = os.path.join(output_root, 'inference_details')

    # Archiving moves existing outputs away, so non-clobbering runs disable archive rotation.
    if (not clobber) and archive_existing:
        print("[WARN] clobber=False is incompatible with archive_existing=True. Disabling archive_existing.")
        archive_existing = False

    # Archive existing inference outputs before starting the run, including detailed
    # sidecar outputs that could otherwise become stale across model sets.
    if archive_existing:
        os.makedirs(past_inferences_dir, exist_ok=True)
        if use_inferences_subdir:
            os.makedirs(inference_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        for filename in os.listdir(inference_dir):
            if not filename.endswith('_inference.csv'):
                continue
            file_path = os.path.join(inference_dir, filename)
            if os.path.isfile(file_path):
                new_filename = f"{timestamp}_{filename}"
                shutil.move(file_path, os.path.join(past_inferences_dir, new_filename))

        if os.path.isdir(details_dir_for_archive):
            archived_details_dir = os.path.join(past_inferences_dir, f"{timestamp}_inference_details")
            if os.path.exists(archived_details_dir):
                shutil.rmtree(archived_details_dir, ignore_errors=True)
            shutil.move(details_dir_for_archive, archived_details_dir)

    # Resolve the exact file set before loading checkpoint metadata.
    # This matters for pipeline registry usage where explicit input files may live outside `directory`,
    # and for safety because outputs are written by file stem.
    file_list = _discover_inference_input_files(
        directory,
        explicit_files=explicit_files,
        file_filter=file_filter,
    )
    if not file_list:
        print("No files to process (after filtering / selection).")
        return

    # --- Use the first model as the reference for configuration ---
    main_model_path = inference_model_paths[0]
    print(f"Using '{os.path.basename(main_model_path)}' as the reference for model configuration.")

    # Prepare for inference by getting the selected file list and model configuration from the main model
    file_list, pose_dict, orient_dict, columns_to_noise, feature_plan = prepare_data_inference(
        directory=directory,
        pretrained_model_path=main_model_path, # Use the main model path here
        num_workers=num_workers,
        inference_mode=True,
        candidate_files=file_list,
    )

    # Keep existing outputs when clobbering is disabled.
    if not clobber:
        kept: List[str] = []
        skipped = 0
        for fp in file_list:
            stem = os.path.splitext(os.path.basename(fp))[0]
            expected = os.path.join(inference_dir, f"{stem}_inference.csv")
            if os.path.exists(expected):
                skipped += 1
            else:
                kept.append(fp)
        if skipped:
            print(f"Skipping {skipped} files with existing inference CSVs in: {inference_dir}")
        file_list = kept

    if not file_list:
        print("No files to process (after filtering / existing-output check).")
        return

    # --- Validate ensemble compatibility and build class-alignment permutations ---
    # This prevents silent corruption when different checkpoints use different label->index mappings.
    _, pose_permutations, orient_permutations = check_ensemble_compatibility_and_get_permutations(
        checkpoint_paths=inference_model_paths,
        reference_pose_dict=pose_dict,
        reference_orient_dict=orient_dict,
    )

    # --- Load ALL models for the ensemble ---
    print(f"\nLoading {len(inference_model_paths)} models for ensembling...")
    models = []
    checkpoint = None # To store the checkpoint of the main model
    for i, model_path in enumerate(inference_model_paths):
        print(f" -> Loading model {i+1}/{len(inference_model_paths)} from {os.path.basename(model_path)}")
        model, loaded_checkpoint = load_model_for_inference(
            checkpoint_path=model_path,
            pose_dict=pose_dict,
            orient_dict=orient_dict
        )
        models.append(model)
        if i == 0: # Save the checkpoint of the first model
            checkpoint = loaded_checkpoint
    print("All models loaded successfully.")


    # Extract the actual run parameters from the main model's checkpoint to ensure consistency
    inference_params = checkpoint.get('global_vars', {})
    inf_pose_pred = inference_params.get('pose_pred', False)
    inf_orient_pred = inference_params.get('orient_pred', False)
    inf_recon = inference_params.get('recon', False)
    inf_sequence_length = inference_params.get('sequence_length', sequence_length)

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Print label dictionaries before inference.
    print("\nPose Dict before inference:", pose_dict)
    print("Orient Dict before inference:", orient_dict)

    # Process each file individually to conserve RAM
    for i, file_path in enumerate(file_list):
        print(f"\n--- Processing file {i+1}/{len(file_list)}: {os.path.basename(file_path)} ---")
        
        file_start_time = time.time()

        # Create a temporary dataset and dataloader for the current file
        single_file_dataset = KeypointsDataset(
            directory=directory,
            metadata=[file_path], 
            sequence_length=inf_sequence_length,
            num_workers=num_workers,
            pose_pred=inf_pose_pred,
            orient_pred=inf_orient_pred,
            recon=inf_recon,
            inference_mode=True,
            feature_plan=feature_plan,
            ablation_strict=True,
        )
        
        if len(single_file_dataset) == 0:
            print("No valid sequences found in this file. Skipping.")
            continue
            
        dataloader = DataLoader(single_file_dataset, batch_size=batch, shuffle=False, num_workers=num_workers, pin_memory=True)

        # --- Pass the list of models to perform_inference ---
        perform_inference(
            models, dataloader, device, directory, inf_average, inf_pose_pred, inf_orient_pred, 
            inf_recon, inf_sequence_length, pose_dict, orient_dict, use_tta, 
            tta_noise_level, columns_to_noise, 
            min_pose_bout, min_orient_bout, max_inference_cvs,
            model_checkpoint_paths=inference_model_paths,
            pose_permutations=pose_permutations,
            orient_permutations=orient_permutations,
            output_dir=output_root,
            use_inferences_subdir=use_inferences_subdir,
        )
        
        # Clean up memory before processing the next file
        del single_file_dataset, dataloader

        file_end_time = time.time()
        duration = file_end_time - file_start_time
        print(f"--- File processing completed in {duration:.2f} seconds ---")

    print("\n--- All files processed. Inference complete. ---")


def prepare_data_inference(directory: str, pretrained_model_path: str, num_workers: int, 
                           inference_mode: bool = True, candidate_files: Optional[List[str]] = None) -> tuple[list[str], dict[str, int], dict[str, int], list[int], Optional[FeaturePlan]]:
    """
    Prepares for inference by gathering file paths and loading configuration from a trained model checkpoint.
    
    This function scans the target directory for all HDF5 files and loads critical configuration
    details (class mappings, augmentation columns, etc.) from the specified model checkpoint
    to ensure consistency during inference. It does NOT create the dataset itself.
    
    Parameters:
    - directory: Path to directory containing HDF5 data files for inference.
    - pretrained_model_path: Path to the checkpoint file (.pth) containing model and config.
    - num_workers: (Not used in this version but kept for signature consistency).
    - inference_mode: (Not used in this version but kept for signature consistency).
    
    Returns:
    - all_files: A list of full file paths to all HDF5 files found in the directory.
    - pose_dict: Dictionary mapping pose class names to integer indices.
    - orient_dict: Dictionary mapping orientation class names to integer indices.
    - columns_to_noise: List of column indices for noise augmentation.
    - feature_plan: Optional FeaturePlan describing which columns to include (for models trained with ablation).
    """
    # Determine the exact HDF5 files participating in this inference call.
    if candidate_files is not None:
        all_files = [str(p) for p in candidate_files]
        print(f"Using {len(all_files)} preselected HDF5 files for inference.")
    else:
        all_files = [os.path.join(directory, f) for f in os.listdir(directory) if (f.endswith('.h5') or f.endswith('.hdf5')) and 'metadata' not in f]
        print(f"Found {len(all_files)} HDF5 files to process for inference.")

    # Load the checkpoint to retrieve saved parameters safely on the CPU
    checkpoint = torch.load(pretrained_model_path, map_location='cpu')
    
    # Load pose_dict and orient_dict from the checkpoint
    pose_dict = checkpoint.get('pose_dict', {})
    orient_dict = checkpoint.get('orient_dict', {})

    # Extract augmentation column info
    global_vars = checkpoint.get('global_vars', {})
    if not isinstance(global_vars, dict):
        global_vars = {}
    columns_to_noise = global_vars.get('columns_to_noise')

    # Inference requires at least one supervised prediction head.
    if bool(global_vars.get('pretrain_mode', False)):
        raise ValueError(
            "Checkpoint is a pretraining checkpoint and does not contain supervised pose/orientation heads. "
            "Use a supervised posepred training checkpoint for inference."
        )
    if (not bool(global_vars.get('pose_pred', False))) and (not bool(global_vars.get('orient_pred', False))):
        raise ValueError(
            "Checkpoint has neither pose_pred nor orient_pred enabled. "
            "posepred_inference requires at least one prediction head."
        )

    ckpt_num_features = global_vars.get('num_features', None)
    if not all_files:
        raise FileNotFoundError("No suitable HDF5 files found for inference.")
    current_headers = _read_headers_from_hdf5_path(str(all_files[0]))

    # Extract feature schema info (if present)
    feature_plan: Optional[FeaturePlan] = None
    included = global_vars.get('feature_columns_included', None)
    schema_hash = global_vars.get('feature_schema_hash', None)
    if included is not None:
        if not isinstance(included, (list, tuple)) or len(included) == 0:
            raise ValueError("Checkpoint has feature_columns_included but it is not a non-empty list.")
        included_cols = [str(x) for x in included]

        if ckpt_num_features is not None and int(ckpt_num_features) != len(included_cols):
            raise ValueError(
                f"Checkpoint is internally inconsistent: num_features={int(ckpt_num_features)} but "
                f"feature_columns_included has length {len(included_cols)}."
            )

        missing = [c for c in included_cols if c not in set(current_headers)]
        if missing:
            raise ValueError(
                "Current inference directory is missing one or more columns required by the checkpoint. "
                f"Missing: {missing[:20]}" + (" ..." if len(missing) > 20 else "")
            )

        computed_hash = compute_schema_hash(included_cols)
        if schema_hash is not None and str(schema_hash) != computed_hash:
            raise ValueError(
                "Checkpoint is internally inconsistent: feature_schema_hash does not match "
                "feature_columns_included."
            )

        excluded = [c for c in current_headers if c not in set(included_cols)]
        effective_hash = str(schema_hash) if schema_hash is not None else computed_hash
        feature_plan = FeaturePlan(
            canonical_columns=list(current_headers),
            included_columns=list(included_cols),
            excluded_columns=list(excluded),
            schema_hash=effective_hash,
            spec={"mode": "checkpoint"},
        )
    elif ckpt_num_features is not None and len(current_headers) != int(ckpt_num_features):
        raise ValueError(
            "Checkpoint does not include feature_columns_included, and the current inference directory "
            f"has {len(current_headers)} columns while the checkpoint expects {int(ckpt_num_features)}. "
            "Cannot safely infer the feature mapping."
        )

    if feature_plan is not None:
        headers_for_aug = list(feature_plan.included_columns)
    else:
        headers_for_aug = list(current_headers)

    # Validate augmentation columns against the active feature plan.
    if columns_to_noise is None:
        print("Augmentation column lists not found in checkpoint. Generating them from HDF5 headers...")
        columns_to_noise = select_columns_for_augmentation(headers_for_aug)
    else:
        try:
            columns_to_noise = [int(x) for x in columns_to_noise]
        except Exception:
            columns_to_noise = []
        active_dim = len(headers_for_aug)
        if any((idx < 0 or idx >= active_dim) for idx in columns_to_noise):
            print(
                "Warning: Checkpoint columns_to_noise is invalid for the active feature layout. "
                "Regenerating augmentation columns from the active headers."
            )
            columns_to_noise = select_columns_for_augmentation(headers_for_aug)

    return all_files, pose_dict, orient_dict, columns_to_noise, feature_plan


def load_model_for_inference(checkpoint_path: str, pose_dict: dict[str, int], 
                            orient_dict: dict[str, int]) -> tuple[PosePredTransformer, dict]:
    """
    Prepares and loads a PosePredTransformer model for inference from a checkpoint.
    
    This function performs complete model restoration for inference by reading the checkpoint,
    extracting hyperparameters, building the model architecture, and loading trained weights.
    The model is always built in inference mode (pretrain_mode=False) regardless of the
    original training mode to ensure proper prediction head initialization.
    
    Model Loading Process:
    1. Checkpoint file validation and loading
    2. Hyperparameter extraction from stored checkpoint metadata
    3. Class count determination from provided dictionaries
    4. Model architecture instantiation with inference configuration
    5. Weight loading with graceful handling of pretraining artifacts
    6. Model verification through layer shape inspection
    7. Parameter count reporting for validation
    
    Parameters:
    - checkpoint_path: Path to the saved model checkpoint file (.pth format).
    - pose_dict: Dictionary mapping pose class names to integer indices for prediction.
    - orient_dict: Dictionary mapping orientation class names to integer indices for prediction.
    
    Returns:
    - model: Fully initialized PosePredTransformer in evaluation mode with loaded weights.
    - checkpoint: Complete checkpoint dictionary containing all saved state and metadata.
    
    Raises:
    - ValueError: If 'global_vars' are missing from the checkpoint.
    - SystemExit: If the checkpoint file is not found at the specified path.
    
    Notes:
    - Uses strict=False for weight loading to handle pretraining-to-finetuning transitions
    - Always builds model without projection head regardless of checkpoint origin
    - Prints detailed model architecture information for verification
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if not os.path.isfile(checkpoint_path):
        print(f"Error: Checkpoint file not found at {checkpoint_path}. Exiting inference mode.")
        sys.exit(1)
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    global_vars = checkpoint.get('global_vars', {})
    if not isinstance(global_vars, dict) or not global_vars:
        raise ValueError("Could not find 'global_vars' in the checkpoint. Cannot build model for inference.")

    if bool(global_vars.get('pretrain_mode', False)):
        raise ValueError(
            "Checkpoint is a pretraining checkpoint and cannot be used for posepred inference. "
            "Use a supervised training checkpoint instead."
        )
    if (not bool(global_vars.get('pose_pred', False))) and (not bool(global_vars.get('orient_pred', False))):
        raise ValueError(
            "Checkpoint has neither pose_pred nor orient_pred enabled and cannot be used for posepred inference."
        )

    def _format_global_var_for_log(value):
        if isinstance(value, (list, tuple)) and len(value) > 20:
            head = [str(x) for x in value[:5]]
            tail = [str(x) for x in value[-3:]]
            return f"{type(value).__name__}(len={len(value)}, first={head}, last={tail})"
        if isinstance(value, dict) and len(value) > 20:
            keys = list(value.keys())
            return f"dict(len={len(value)}, keys_first={keys[:8]})"
        return value

    print("\nModel parameters loaded from checkpoint for inference:")
    for key, value in global_vars.items():
        print(f"  {key}: {_format_global_var_for_log(value)}")

    # Determine num_classes from the provided dictionaries
    num_pose_classes = len(pose_dict) if global_vars.get('pose_pred') else 0
    num_orient_classes = len(orient_dict) if global_vars.get('orient_pred') else 0
    
    print(f"Length of pose_dict: {num_pose_classes}")
    print(f"Length of orient_dict: {num_orient_classes}")
    
    # Instantiate the model with parameters from the checkpoint, but ALWAYS in inference mode.
    model = PosePredTransformer(
        num_features=global_vars['num_features'],
        sequence_length=global_vars['sequence_length'],
        d_model=global_vars['d_model'],
        nhead=global_vars['nhead'],
        num_encoder_layers=global_vars['num_encoder_layers'],
        num_decoder_layers=global_vars['num_decoder_layers'],
        dim_feedforward=global_vars['dim_feedforward'],
        dropout_rate=global_vars['dropout_rate'],
        frame_representation=global_vars['frame_representation'],
        num_cls=global_vars['num_cls'],
        prediction_head=global_vars['prediction_head'],
        pose_pred=global_vars.get('pose_pred', False),
        orient_pred=global_vars.get('orient_pred', False),
        recon=global_vars.get('recon', False),
        num_pose_classes=num_pose_classes,
        num_orient_classes=num_orient_classes,
        positional_encoding_type=global_vars.get('positional_encoding_type', 'learned'),
        pretrain_mode=False,  # CRITICAL: Always build the model without the projection head for inference
        projection_dim=global_vars.get('projection_dim', 128)
    )
    
    model.to(device)
    
    print("\nLoading model weights for inference...")

    load_checkpoint_model(model, optimizer=None, scheduler=None, checkpoint_path=checkpoint_path, scaler=None, load_optimizer_state=False, pretrain_mode=False)
    
    # For printing, extract the underlying module if wrapped.
    model_module = model.module if hasattr(model, 'module') else model
    
    print("\nModel parameters after weight loading:")
    print(f"Initial Linear Input Shape: {model_module.initial_linear.weight.shape}")
    
    if hasattr(model_module, 'positional_encoding'):
        print(f"Positional Encoding Shape: {model_module.positional_encoding.shape}")
    print(f"Number of Encoder Layers: {len(model_module.transformer_encoder.layers)}")
    if len(model_module.transformer_encoder.layers) > 0:
        print(f"Encoder Layer Shape: {model_module.transformer_encoder.layers[0].self_attn.in_proj_weight.shape}")
    
    if model_module.pose_pred:
        if model_module.prediction_head == 'fc':
            print(f"Pose FC Layer Weight Shape: {model_module.pose_fc.weight.shape}")
        elif hasattr(model_module, 'pose_mlp') and len(model_module.pose_mlp) > 0:
            print(f"Pose MLP First Layer Weight Shape: {model_module.pose_mlp[0].weight.shape}")

    if model_module.orient_pred:
        if model_module.prediction_head == 'fc':
            print(f"Orientation FC Layer Weight Shape: {model_module.orient_fc.weight.shape}")
        elif hasattr(model_module, 'orient_mlp') and len(model_module.orient_mlp) > 0:
            print(f"Orientation MLP First Layer Weight Shape: {model_module.orient_mlp[0].weight.shape}")
    
    if model_module.recon:
        print(f"Number of Decoder Layers: {len(model_module.transformer_decoder.layers)}")
        if len(model_module.transformer_decoder.layers) > 0:
            print(f"Decoder Layer Shape: {model_module.transformer_decoder.layers[0].self_attn.in_proj_weight.shape}")
        print(f"Output Linear Weight Shape: {model_module.output_linear.weight.shape}")
    
    total_params = sum(p.numel() for p in model_module.parameters())
    print("\nTotal number of parameters:", total_params)

    
    # Return the initialized model and the checkpoint data
    return model, checkpoint


def perform_inference(
    models: List[nn.Module],
    dataloader: DataLoader,
    device: torch.device,
    main_directory: str,
    inf_average: int,
    pose_pred: bool,
    orient_pred: bool,
    recon: bool,
    sequence_length: int,
    pose_dict: dict[str, int],
    orient_dict: dict[str, int],
    use_tta: bool,
    tta_noise_level: float,
    columns_to_noise: list[int],
    min_pose_bout: int,
    min_orient_bout: int,
    max_inference_cvs: int,
    model_checkpoint_paths: Optional[List[str]] = None,
    pose_permutations: Optional[List[Optional[List[int]]]] = None,
    orient_permutations: Optional[List[Optional[List[int]]]] = None,
    *,
    output_dir: Optional[str] = None,
    use_inferences_subdir: bool = True,
) -> None:
    """
    Executes batched inference and writes:
      1) Main inference CSV (probabilities + predicted labels) to: inferences/<movie>_inference.csv
      2) Logits CSV (raw per-model logits; aligned to reference label indexing) to:
         inferences/inference_details/<movie>_logits.csv
      3) Context vectors HDF5 (one file per *model* per movie) to:
         inferences/inference_details/context_vectors/model_XXX_<ckptstem>/<movie>_context_vectors.h5

    Context vectors are saved as the TTA-mean (inf_average) embedding from each model independently
    (NO cross-model averaging in embedding space). The HDF5 file also stores raw (unsmoothed) ensemble
    predicted pose/orient indices so you can filter embeddings by the "best" label estimate.
    """
    for model in models:
        model.eval()

    # Validate that any provided permutation lists match the number of models
    if pose_permutations is not None and len(pose_permutations) != len(models):
        raise ValueError(
            f"pose_permutations length ({len(pose_permutations)}) does not match number of models ({len(models)})."
        )
    if orient_permutations is not None and len(orient_permutations) != len(models):
        raise ValueError(
            f"orient_permutations length ({len(orient_permutations)}) does not match number of models ({len(models)})."
        )
    if model_checkpoint_paths is not None and len(model_checkpoint_paths) != len(models):
        raise ValueError(
            f"model_checkpoint_paths length ({len(model_checkpoint_paths)}) does not match number of models ({len(models)})."
        )

    # --- Output folders ---
    output_root = output_dir if output_dir else main_directory
    os.makedirs(output_root, exist_ok=True)

    if use_inferences_subdir:
        inference_dir = os.path.join(output_root, 'inferences')
        details_dir = os.path.join(inference_dir, 'inference_details')
    else:
        inference_dir = output_root
        details_dir = os.path.join(output_root, 'inference_details')
    os.makedirs(details_dir, exist_ok=True)

    context_vectors_root = os.path.join(details_dir, 'context_vectors')
    os.makedirs(context_vectors_root, exist_ok=True)

    # --- Label reverse maps (index -> name) ---
    reverse_pose_dict = {v: k for k, v in pose_dict.items()}
    reverse_orient_dict = {v: k for k, v in orient_dict.items()}

    pose_labels_in_index_order: Optional[List[str]] = None
    orient_labels_in_index_order: Optional[List[str]] = None
    if pose_pred and pose_dict:
        pose_labels_in_index_order = [reverse_pose_dict[i] for i in range(len(pose_dict))]
    if orient_pred and orient_dict:
        orient_labels_in_index_order = [reverse_orient_dict[i] for i in range(len(orient_dict))]

    # --- How many model embeddings to save? (default: all models; can be limited by max_inference_cvs) ---
    n_models = len(models)
    n_save_models = min(max_inference_cvs, n_models)

    # Per-model embedding dimensions (can differ across ensemble members; each gets its own file)
    model_d_models: List[int] = []
    model_frame_reps: List[str] = []
    for m in range(n_save_models):
        mm = models[m].module if hasattr(models[m], 'module') else models[m]
        d_model = getattr(mm, 'd_model', None)
        if d_model is None:
            raise RuntimeError(f"Could not determine d_model for model index {m}.")
        model_d_models.append(int(d_model))
        model_frame_reps.append(str(getattr(mm, 'frame_representation', 'unknown')))

    # Dataset file map (best-effort). Used to get total_frames for padding/writing.
    dataset_h5_files = getattr(dataloader.dataset, 'h5_files', [])
    filename_to_path = {os.path.basename(str(f)): str(f) for f in dataset_h5_files} if dataset_h5_files else {}

    # --- Results for CSV outputs (still written as before) ---
    all_main_results: dict[str, list[list]] = {}
    all_logits_results: dict[str, list[list]] = {}

    # --- Buffers for context vectors (one per movie/file) ---
    # Stored as full-length arrays so downstream can slice by frame number without loading CSV.
    cv_buffers: dict[str, dict] = {}

    def _sanitize_for_path(name: str, max_len: int = 80) -> str:
        safe = ''.join(c if (c.isalnum() or c in '._-') else '_' for c in str(name))
        safe = safe.strip('_')
        if not safe:
            safe = 'checkpoint'
        if max_len and len(safe) > max_len:
            safe = safe[:max_len].rstrip('_')
        return safe

    def _ensure_cv_buffer(base_filename: str, source_path: str) -> dict:
        if base_filename in cv_buffers:
            return cv_buffers[base_filename]

        if not os.path.isfile(source_path):
            # Try dataset mapping as a fallback
            source_path = filename_to_path.get(base_filename, source_path)

        if not os.path.isfile(source_path):
            raise FileNotFoundError(f"Could not locate source HDF5 for '{base_filename}'. Tried: {source_path}")

        with h5py.File(source_path, 'r') as h5f:
            total_frames = int(h5f['data'].shape[0])
            num_features_local = int(h5f['data'].shape[1])

        buf = {
            'source_path': source_path,
            'total_frames': total_frames,
            'num_features': num_features_local,
            'frame_number': np.arange(1, total_frames + 1, dtype=np.int32),
            'valid': np.zeros(total_frames, dtype=np.uint8),
            # Ensemble (raw, unsmoothed) predictions
            'pred_pose_idx_ensemble': (np.full(total_frames, -1, dtype=np.int16) if pose_pred else None),
            'pred_orient_idx_ensemble': (np.full(total_frames, -1, dtype=np.int16) if orient_pred else None),
            # Model-specific predictions (aligned to reference indexing) for agreement analysis
            'pred_pose_idx_model': ([np.full(total_frames, -1, dtype=np.int16) for _ in range(n_save_models)] if pose_pred else None),
            'pred_orient_idx_model': ([np.full(total_frames, -1, dtype=np.int16) for _ in range(n_save_models)] if orient_pred else None),
            # Context vectors (one per model)
            'context_vectors': [
                np.full((total_frames, model_d_models[m]), np.nan, dtype=np.float16)
                for m in range(n_save_models)
            ],
        }
        cv_buffers[base_filename] = buf
        return buf

    total_sequences_processed = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            sequences, filenames, middle_indices = batch[0].to(device), batch[1], batch[2]
            print(f"Processing batch {batch_idx + 1}/{len(dataloader)}...")

            batch_logits_pose_all_models: List[torch.Tensor] = []
            batch_logits_orient_all_models: List[torch.Tensor] = []
            batch_cls_tokens_all_models: List[torch.Tensor] = []

            # --- Forward passes for each model (with optional TTA + averaging) ---
            for model_idx, model in enumerate(models):
                accumulated_logits_pose = None
                accumulated_logits_orient = None
                accumulated_cls_token = None

                for pass_idx in range(inf_average):
                    sequences_to_process = sequences.clone()
                    if use_tta:
                        sequences_to_process = augment_features(
                            sequences_to_process, 0, tta_noise_level, columns_to_noise
                        )
                    with torch.amp.autocast(device_type='cuda', enabled=device.type == 'cuda'):
                        logits_pose, logits_orient, _, cls_token = model(sequences_to_process)

                    if pose_pred:
                        accumulated_logits_pose = logits_pose.clone() if accumulated_logits_pose is None else accumulated_logits_pose + logits_pose
                    if orient_pred:
                        accumulated_logits_orient = logits_orient.clone() if accumulated_logits_orient is None else accumulated_logits_orient + logits_orient
                    accumulated_cls_token = cls_token.clone() if accumulated_cls_token is None else accumulated_cls_token + cls_token

                # Average over passes
                if pose_pred:
                    if accumulated_logits_pose is None:
                        raise RuntimeError(f"Model {model_idx+1} produced no pose logits, but pose_pred=True.")
                    pose_logits_avg = accumulated_logits_pose / inf_average
                    if pose_permutations is not None and pose_permutations[model_idx] is not None:
                        # Reorder columns so every model is aligned to the reference pose_dict indexing
                        pose_logits_avg = pose_logits_avg[:, pose_permutations[model_idx]]
                    batch_logits_pose_all_models.append(pose_logits_avg)

                if orient_pred:
                    if accumulated_logits_orient is None:
                        raise RuntimeError(f"Model {model_idx+1} produced no orient logits, but orient_pred=True.")
                    orient_logits_avg = accumulated_logits_orient / inf_average
                    if orient_permutations is not None and orient_permutations[model_idx] is not None:
                        # Reorder columns so every model is aligned to the reference orient_dict indexing
                        orient_logits_avg = orient_logits_avg[:, orient_permutations[model_idx]]
                    batch_logits_orient_all_models.append(orient_logits_avg)

                batch_cls_tokens_all_models.append(accumulated_cls_token / inf_average)

            # --- Ensemble predictions (aligned class indexing) ---
            predicted_pose_indices = None
            predicted_pose_labels = None
            ensemble_probs_pose = None

            predicted_orient_indices = None
            predicted_orient_labels = None
            ensemble_probs_orient = None

            if pose_pred:
                ensemble_logits_pose = torch.stack(batch_logits_pose_all_models, dim=0).mean(dim=0)
                ensemble_probs_pose = F.softmax(ensemble_logits_pose, dim=1).cpu().numpy()
                predicted_pose_indices = np.argmax(ensemble_probs_pose, axis=1)
                predicted_pose_labels = [reverse_pose_dict[idx] for idx in predicted_pose_indices]

            if orient_pred:
                ensemble_logits_orient = torch.stack(batch_logits_orient_all_models, dim=0).mean(dim=0)
                ensemble_probs_orient = F.softmax(ensemble_logits_orient, dim=1).cpu().numpy()
                predicted_orient_indices = np.argmax(ensemble_probs_orient, axis=1)
                predicted_orient_labels = [reverse_orient_dict[idx] for idx in predicted_orient_indices]

            total_sequences_processed += sequences.size(0)

            # --- Convert embeddings / per-model argmax once per batch (avoid per-row GPU->CPU conversions) ---
            cls_np_all_models: List[np.ndarray] = []
            model_pose_idx_all_models: Optional[List[np.ndarray]] = None
            model_orient_idx_all_models: Optional[List[np.ndarray]] = None

            for m in range(n_save_models):
                cls_np_all_models.append(batch_cls_tokens_all_models[m].detach().cpu().numpy().astype(np.float16, copy=False))

            if pose_pred:
                model_pose_idx_all_models = []
                for m in range(n_save_models):
                    model_pose_idx_all_models.append(
                        torch.argmax(batch_logits_pose_all_models[m], dim=1).detach().cpu().numpy().astype(np.int16, copy=False)
                    )

            if orient_pred:
                model_orient_idx_all_models = []
                for m in range(n_save_models):
                    model_orient_idx_all_models.append(
                        torch.argmax(batch_logits_orient_all_models[m], dim=1).detach().cpu().numpy().astype(np.int16, copy=False)
                    )

            # --- Update per-file context vector buffers (frame-aligned) ---
            basenames = [os.path.basename(str(fn)) for fn in filenames]
            unique_basenames = sorted(set(basenames))
            middle_idx_np = middle_indices.detach().cpu().numpy().astype(np.int64, copy=False)

            if len(unique_basenames) == 1:
                bn = unique_basenames[0]
                src_path = filename_to_path.get(bn, str(filenames[0]))
                buf = _ensure_cv_buffer(bn, src_path)

                # Mark valid frames and store ensemble predictions
                buf['valid'][middle_idx_np] = 1
                if pose_pred and predicted_pose_indices is not None:
                    buf['pred_pose_idx_ensemble'][middle_idx_np] = predicted_pose_indices.astype(np.int16, copy=False)
                if orient_pred and predicted_orient_indices is not None:
                    buf['pred_orient_idx_ensemble'][middle_idx_np] = predicted_orient_indices.astype(np.int16, copy=False)

                # Store model-specific predictions + embeddings
                for m in range(n_save_models):
                    buf['context_vectors'][m][middle_idx_np, :] = cls_np_all_models[m]
                    if pose_pred and model_pose_idx_all_models is not None:
                        buf['pred_pose_idx_model'][m][middle_idx_np] = model_pose_idx_all_models[m]
                    if orient_pred and model_orient_idx_all_models is not None:
                        buf['pred_orient_idx_model'][m][middle_idx_np] = model_orient_idx_all_models[m]
            else:
                # Multi-file batches: group by base filename
                for bn in unique_basenames:
                    positions = [j for j, b in enumerate(basenames) if b == bn]
                    if not positions:
                        continue
                    src_path = filename_to_path.get(bn, str(filenames[positions[0]]))
                    buf = _ensure_cv_buffer(bn, src_path)

                    idxs = middle_idx_np[positions]
                    buf['valid'][idxs] = 1
                    if pose_pred and predicted_pose_indices is not None:
                        buf['pred_pose_idx_ensemble'][idxs] = predicted_pose_indices[positions].astype(np.int16, copy=False)
                    if orient_pred and predicted_orient_indices is not None:
                        buf['pred_orient_idx_ensemble'][idxs] = predicted_orient_indices[positions].astype(np.int16, copy=False)

                    for m in range(n_save_models):
                        buf['context_vectors'][m][idxs, :] = cls_np_all_models[m][positions, :]
                        if pose_pred and model_pose_idx_all_models is not None:
                            buf['pred_pose_idx_model'][m][idxs] = model_pose_idx_all_models[m][positions]
                        if orient_pred and model_orient_idx_all_models is not None:
                            buf['pred_orient_idx_model'][m][idxs] = model_orient_idx_all_models[m][positions]

            # --- Build rows for CSV outputs (per frame prediction rows only) ---
            for i, filename in enumerate(filenames):
                base_filename = os.path.basename(str(filename))
                if base_filename not in all_main_results:
                    all_main_results[base_filename] = []
                    all_logits_results[base_filename] = []

                frame_number = middle_indices[i].item() + 1

                main_row = [frame_number]
                logits_row = [frame_number]

                # Logits row
                for model_idx in range(len(models)):
                    if pose_pred:
                        logits_row.append(batch_logits_pose_all_models[model_idx][i].detach().cpu().numpy().tolist())
                    if orient_pred:
                        logits_row.append(batch_logits_orient_all_models[model_idx][i].detach().cpu().numpy().tolist())

                # Main row
                if pose_pred and ensemble_probs_pose is not None:
                    main_row.extend([round(ensemble_probs_pose[i][pose_dict[label]], 4) for label in pose_dict.keys()])
                if orient_pred and ensemble_probs_orient is not None:
                    main_row.extend([round(ensemble_probs_orient[i][orient_dict[label]], 4) for label in orient_dict.keys()])

                if pose_pred and predicted_pose_labels is not None:
                    main_row.append(predicted_pose_labels[i])
                if orient_pred and predicted_orient_labels is not None:
                    main_row.append(predicted_orient_labels[i])
                if pose_pred and orient_pred and predicted_pose_labels is not None and predicted_orient_labels is not None:
                    main_row.append(f"{predicted_pose_labels[i]}-{predicted_orient_labels[i]}")

                all_main_results[base_filename].append(main_row)
                all_logits_results[base_filename].append(logits_row)

    print("\nInference loop complete. Processing and writing results to separate files...")

    # --- Write outputs per file ---
    for base_filename, main_results in all_main_results.items():
        logits_results = all_logits_results[base_filename]

        # Headers
        main_headers = ['Frame Number']
        if pose_pred:
            main_headers.extend([f"Pose-{label}-Prob" for label in pose_dict.keys()])
        if orient_pred:
            main_headers.extend([f"Orient-{label}-Prob" for label in orient_dict.keys()])
        if pose_pred:
            main_headers.append('Predicted Pose Label')
        if orient_pred:
            main_headers.append('Predicted Orient Label')
        if pose_pred and orient_pred:
            main_headers.append('Predicted Behavior')

        logits_headers = ['Frame Number']
        for model_idx in range(len(models)):
            if pose_pred:
                logits_headers.append(f'Pose_Logits_M{model_idx+1}')
            if orient_pred:
                logits_headers.append(f'Orient_Logits_M{model_idx+1}')

        # DataFrames for CSV outputs
        main_df = pd.DataFrame(main_results, columns=main_headers)
        logits_df = pd.DataFrame(logits_results, columns=logits_headers)

        # Full timeline (for padding)
        buf = cv_buffers.get(base_filename)
        if buf is not None:
            total_frames = int(buf['total_frames'])
        else:
            # Fallback: use observed max frame
            total_frames = int(main_df['Frame Number'].max()) if not main_df.empty else 0

        full_df = pd.DataFrame({'Frame Number': range(1, total_frames + 1)})

        # 1) LOGITS CSV
        final_logits_df = pd.merge(full_df, logits_df, on='Frame Number', how='left')
        logits_csv_path = os.path.join(details_dir, f"{os.path.splitext(base_filename)[0]}_logits.csv")
        print(f"  Writing {len(final_logits_df)} rows to {logits_csv_path}...")
        final_logits_df.to_csv(logits_csv_path, index=False)

        # 2) CONTEXT VECTORS HDF5 (per model)
        if buf is None:
            print(f"  WARNING: No context-vector buffer found for {base_filename}; skipping HDF5 context vector write.")
        else:
            # Precompute agreement flags per model (optional but useful)
            valid = buf['valid']
            pred_pose_ens = buf['pred_pose_idx_ensemble']
            pred_orient_ens = buf['pred_orient_idx_ensemble']

            # Determine base output name once
            base_out_stem = os.path.splitext(base_filename)[0]
            timestamp = datetime.now().isoformat()

            for m in range(n_save_models):
                ckpt_stem = os.path.splitext(os.path.basename(model_checkpoint_paths[m]))[0] if model_checkpoint_paths else f"model_{m+1:03d}"
                ckpt_stem = _sanitize_for_path(ckpt_stem)

                model_dir_name = f"model_{m+1:03d}_{ckpt_stem}"
                model_out_dir = os.path.join(context_vectors_root, model_dir_name)
                os.makedirs(model_out_dir, exist_ok=True)

                cv_h5_path = os.path.join(model_out_dir, f"{base_out_stem}_context_vectors.h5")
                tmp_path = cv_h5_path + ".tmp"

                # Datasets to write
                context_vec = buf['context_vectors'][m]
                pred_pose_model = buf['pred_pose_idx_model'][m] if pose_pred and buf['pred_pose_idx_model'] is not None else None
                pred_orient_model = buf['pred_orient_idx_model'][m] if orient_pred and buf['pred_orient_idx_model'] is not None else None

                try:
                    with h5py.File(tmp_path, 'w') as hf:
                        # Root attributes (metadata)
                        hf.attrs['schema_version'] = "context_vectors_h5_v1"
                        hf.attrs['created_at'] = timestamp
                        hf.attrs['script_name'] = os.path.basename(__file__) if '__file__' in globals() else 'autobehaver_posepred'
                        hf.attrs['source_hdf5'] = buf.get('source_path', '')
                        hf.attrs['checkpoint_path'] = model_checkpoint_paths[m] if model_checkpoint_paths else ''
                        hf.attrs['model_index'] = int(m + 1)
                        hf.attrs['num_models_in_ensemble'] = int(n_models)
                        hf.attrs['num_models_saved'] = int(n_save_models)
                        hf.attrs['sequence_length'] = int(sequence_length)
                        hf.attrs['inf_average'] = int(inf_average)
                        hf.attrs['use_tta'] = int(bool(use_tta))
                        hf.attrs['tta_noise_level'] = float(tta_noise_level)
                        hf.attrs['pose_pred'] = int(bool(pose_pred))
                        hf.attrs['orient_pred'] = int(bool(orient_pred))
                        hf.attrs['recon'] = int(bool(recon))
                        hf.attrs['num_features'] = int(buf.get('num_features', -1))
                        hf.attrs['d_model'] = int(context_vec.shape[1])
                        hf.attrs['frame_representation'] = model_frame_reps[m] if m < len(model_frame_reps) else 'unknown'

                        N = context_vec.shape[0]
                        chunk_1d = (min(N, 65536),)

                        # Core indexing datasets
                        hf.create_dataset(
                            'frame_number',
                            data=buf['frame_number'],
                            dtype=np.int32,
                            compression='gzip',
                            compression_opts=4,
                            shuffle=True,
                            chunks=chunk_1d
                        )
                        hf.create_dataset(
                            'valid',
                            data=valid,
                            dtype=np.uint8,
                            compression='gzip',
                            compression_opts=4,
                            shuffle=True,
                            chunks=chunk_1d
                        )

                        # Context vector dataset
                        chunk_frames = min(N, 2048)
                        hf.create_dataset(
                            'context_vector',
                            data=context_vec,
                            dtype=np.float16,
                            compression='gzip',
                            compression_opts=4,
                            shuffle=True,
                            chunks=(chunk_frames, context_vec.shape[1])
                        )

                        # Label vocabularies (index -> string)
                        str_dt = h5py.string_dtype(encoding='utf-8')
                        if pose_pred and pose_labels_in_index_order is not None:
                            hf.create_dataset('pose_labels', data=np.array(pose_labels_in_index_order, dtype=object), dtype=str_dt)
                        if orient_pred and orient_labels_in_index_order is not None:
                            hf.create_dataset('orient_labels', data=np.array(orient_labels_in_index_order, dtype=object), dtype=str_dt)

                        # Raw ensemble predictions (unsmoothed)
                        if pose_pred and pred_pose_ens is not None:
                            hf.create_dataset(
                                'pred_pose_idx_ensemble',
                                data=pred_pose_ens,
                                dtype=np.int16,
                                compression='gzip',
                                compression_opts=4,
                                shuffle=True,
                                chunks=chunk_1d
                            )
                        if orient_pred and pred_orient_ens is not None:
                            hf.create_dataset(
                                'pred_orient_idx_ensemble',
                                data=pred_orient_ens,
                                dtype=np.int16,
                                compression='gzip',
                                compression_opts=4,
                                shuffle=True,
                                chunks=chunk_1d
                            )

                        # Model-specific predictions + agreement flags
                        if pose_pred and pred_pose_model is not None and pred_pose_ens is not None:
                            hf.create_dataset(
                                'pred_pose_idx_model',
                                data=pred_pose_model,
                                dtype=np.int16,
                                compression='gzip',
                                compression_opts=4,
                                shuffle=True,
                                chunks=chunk_1d
                            )
                            agree_pose = ((pred_pose_model == pred_pose_ens) & (valid.astype(bool))).astype(np.uint8)
                            hf.create_dataset(
                                'agree_pose',
                                data=agree_pose,
                                dtype=np.uint8,
                                compression='gzip',
                                compression_opts=4,
                                shuffle=True,
                                chunks=chunk_1d
                            )

                        if orient_pred and pred_orient_model is not None and pred_orient_ens is not None:
                            hf.create_dataset(
                                'pred_orient_idx_model',
                                data=pred_orient_model,
                                dtype=np.int16,
                                compression='gzip',
                                compression_opts=4,
                                shuffle=True,
                                chunks=chunk_1d
                            )
                            agree_orient = ((pred_orient_model == pred_orient_ens) & (valid.astype(bool))).astype(np.uint8)
                            hf.create_dataset(
                                'agree_orient',
                                data=agree_orient,
                                dtype=np.uint8,
                                compression='gzip',
                                compression_opts=4,
                                shuffle=True,
                                chunks=chunk_1d
                            )

                    os.replace(tmp_path, cv_h5_path)
                    print(f"  Wrote context vectors (model {m+1}/{n_save_models}) to {cv_h5_path}")
                except Exception as e:
                    try:
                        if os.path.exists(tmp_path):
                            os.remove(tmp_path)
                    except Exception:
                        pass
                    raise

        # 3) MAIN INFERENCE CSV (with optional smoothing)
        final_main_df = pd.merge(full_df, main_df, on='Frame Number', how='left')

        if pose_pred and min_pose_bout > 1:
            print(f"    -> Applying pose label smoothing (min_bout={min_pose_bout})...")
            final_main_df['Smoothed Pose Label'] = smooth_bouts_vectorized(
                labels=final_main_df['Predicted Pose Label'],
                probs_df=final_main_df,
                min_bout=min_pose_bout,
                prefix='Pose',
                exempt_labels=['Jump']
            )
        if orient_pred and min_orient_bout > 1:
            print(f"    -> Applying orient label smoothing (min_bout={min_orient_bout})...")
            final_main_df['Smoothed Orient Label'] = smooth_bouts_vectorized(
                labels=final_main_df['Predicted Orient Label'],
                probs_df=final_main_df,
                min_bout=min_orient_bout,
                prefix='Orient'
            )

        main_csv_path = os.path.join(inference_dir, f"{os.path.splitext(base_filename)[0]}_inference.csv")
        print(f"  Writing {len(final_main_df)} rows to {main_csv_path}...")
        final_main_df.to_csv(main_csv_path, index=False)

    print(f"\nTotal sequences processed: {total_sequences_processed}")


def smooth_bouts_vectorized(labels, probs_df, min_bout, prefix, exempt_labels=None):
    """
    Smooths a sequence of labels by re-evaluating short bouts based on the
    cumulative probability of flanking behaviors. (VECTORIZED IMPLEMENTATION)

    Exempts any labels in `exempt_labels` from being considered 'short'
    (e.g., 'Jump'), so they are never rewritten by smoothing.
    """
    exempt_labels = set(exempt_labels or [])
    s = pd.Series(labels, dtype="object")
    if s.empty or s.isna().all():
        return s.to_numpy()

    is_blank = s.isna() | (s == "")
    # Work only on non-blank runs
    work = s.mask(is_blank, other="__BLANK__")
    df = pd.DataFrame({'label': work})

    df['bout_id'] = (df['label'].ne(df['label'].shift())).cumsum()
    df['is_blank'] = df['label'] == "__BLANK__"
    df['bout_length'] = df.groupby('bout_id')['label'].transform('size')

    # Never mark exempt labels as "short"
    df['is_exempt'] = df['label'].isin(exempt_labels)
    df['is_short'] = (~df['is_blank']) & (~df['is_exempt']) & (df['bout_length'] < min_bout)

    bout_info = df.loc[~df['is_blank']].drop_duplicates('bout_id').set_index('bout_id')
    bout_info['prev_label'] = bout_info['label'].shift(1)
    bout_info['next_label'] = bout_info['label'].shift(-1)
    df = df.join(bout_info[['prev_label','next_label']], on='bout_id')

    df['smoothed_label'] = df['label']

    easy = df['is_short'] & (df['prev_label'] == df['next_label']) & df['prev_label'].notna()
    df.loc[easy, 'smoothed_label'] = df.loc[easy, 'prev_label']

    df['bout_id_smoothed'] = (df['smoothed_label'].ne(df['smoothed_label'].shift())).cumsum()
    df['bout_length_smoothed'] = df.groupby('bout_id_smoothed')['smoothed_label'].transform('size')

    complex_ids = df.loc[(~df['is_blank']) & (df['bout_length_smoothed'] < min_bout) & (~df['is_exempt']), 'bout_id_smoothed'].unique()

    for bid in complex_ids:
        idx = df.index[df['bout_id_smoothed'] == bid]
        start_idx, end_idx = idx.min(), idx.max()

        prev_label = df['smoothed_label'].iloc[start_idx - 1] if start_idx > 0 else None
        next_label = df['smoothed_label'].iloc[end_idx + 1] if end_idx < len(df) - 1 else None
        if prev_label in [None, "__BLANK__"] and next_label in [None, "__BLANK__"]:
            continue

        def prob_sum(lbl):
            if lbl in [None, "__BLANK__"]:
                return 0.0
            col = f"{prefix}-{lbl}-Prob"
            return probs_df.iloc[start_idx:end_idx+1][col].sum() if col in probs_df.columns else 0.0

        prev_sum, next_sum = prob_sum(prev_label), prob_sum(next_label)
        winner = prev_label if prev_sum > next_sum else (next_label if next_sum > prev_sum else (prev_label or next_label))
        df.loc[start_idx:end_idx, 'smoothed_label'] = winner

    out = df['smoothed_label'].replace({"__BLANK__": np.nan}).to_numpy()

    return out


