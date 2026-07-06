"""Self-supervised pretraining routines.

Implements encoder pretraining with either a contrastive objective (InfoNCE) or a
redundancy-reduction objective (Barlow Twins) prior to supervised fine-tuning."""

from __future__ import annotations

import math
import multiprocessing
import os
import random
import time
from typing import Optional, TYPE_CHECKING

import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from .constants import MEMORY_CLEANUP_EVERY_EPOCHS, STANDARD_WEIGHT_INIT_POLICY
from .augmentation import augment_features, select_columns_for_augmentation
from .losses import info_nce_loss, barlow_twins_loss
from ._training_core import model_setup
from .checkpoint import load_checkpoint_model, save_checkpoint, set_checkpoint_globals
from .io import get_num_features_from_first_hdf5, get_headers_from_first_hdf5
from .records import setup_records_file
from .utils import cleanup_memory

__all__ = ['run_pretraining', 'PretrainingDataset']

if TYPE_CHECKING:  # pragma: no cover
    from .feature_selection import FeaturePlan


def _pretrain_epoch_seed(random_seed: Optional[int], epoch: int) -> int:
    base = int(random_seed if random_seed is not None else 0)
    return (base + 1000003 * int(epoch)) % (2**32)


def _pretrain_chunk_seed(random_seed: Optional[int], epoch: int, chunk_index: int) -> int:
    base = int(random_seed if random_seed is not None else 0)
    return (base + 1000003 * int(epoch) + 9176 * int(chunk_index) + 53) % (2**32)


def _seed_pretrain_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def _extract_pretraining_runtime_state(
    *,
    resume_epoch: int,
    next_chunk_index: int,
    next_batch_index_in_chunk: int,
    epoch_total_loss: float,
    epoch_num_batches: int,
    interval_loss: float,
    interval_steps: int,
    best_interval_loss: float,
    interval_elapsed_s: float,
) -> dict:
    return {
        'resume_epoch': int(resume_epoch),
        'next_chunk_index': int(next_chunk_index),
        'next_batch_index_in_chunk': int(next_batch_index_in_chunk),
        'epoch_total_loss': float(epoch_total_loss),
        'epoch_num_batches': int(epoch_num_batches),
        'interval_loss': float(interval_loss),
        'interval_steps': int(interval_steps),
        'best_interval_loss': float(best_interval_loss),
        'interval_elapsed_s': float(interval_elapsed_s),
    }


def _apply_resume_pretraining_runtime_state(state: Optional[dict]) -> dict:
    if not isinstance(state, dict):
        return {
            'resume_epoch': None,
            'next_chunk_index': 0,
            'next_batch_index_in_chunk': 0,
            'epoch_total_loss': 0.0,
            'epoch_num_batches': 0,
            'interval_loss': 0.0,
            'interval_steps': 0,
            'best_interval_loss': float('inf'),
            'interval_elapsed_s': 0.0,
        }
    return {
        'resume_epoch': int(state.get('resume_epoch')) if state.get('resume_epoch') is not None else None,
        'next_chunk_index': int(state.get('next_chunk_index', 0)),
        'next_batch_index_in_chunk': int(state.get('next_batch_index_in_chunk', 0)),
        'epoch_total_loss': float(state.get('epoch_total_loss', 0.0)),
        'epoch_num_batches': int(state.get('epoch_num_batches', 0)),
        'interval_loss': float(state.get('interval_loss', 0.0)),
        'interval_steps': int(state.get('interval_steps', 0)),
        'best_interval_loss': float(state.get('best_interval_loss', float('inf'))),
        'interval_elapsed_s': float(state.get('interval_elapsed_s', 0.0)),
    }

def run_pretraining(
    directory: str, metadata: pd.DataFrame, sequence_length: int, batch: int, num_workers: int, num_epochs: int, 
    checkpoint_interval: int, log_interval: int, checkpoint_dir: str, run_name: str, projection_dim: int, temperature: float, 
    pretrain_files_per_group: int, mask_p: float, noise_level: float, 
    records_path: str, d_model: int, nhead: int, num_encoder_layers: int, num_decoder_layers: int, 
    dim_feedforward: int, dropout_rate: float, frame_representation: str, num_cls: int, prediction_head: str, 
    positional_encoding_type: str, base_lr: float, warmup_steps: int, 
    patience: int, plateau_factor: float, pretrain_loss: str, barlow_lambda: float,
    scheduler_type: str, weight_decay: float, t_0: int, t_mult: int, eta_min: float,
    grad_clip_norm: float, random_seed: Optional[int], metadata_tag: str,
    feature_plan: Optional["FeaturePlan"] = None,
    ablation_strict: bool = True,
    checkpoint_path: Optional[str] = None,
    restart_optimizer: bool = False,
) -> None:
    """
    Run self-supervised pretraining to learn general sequence representations.

    The current implementation supports two self-supervised objectives:
    - ``info_nce`` for contrastive pretraining with positive pairs
    - ``barlow_twins`` for redundancy-reduction pretraining

    Execution Flow:

    1. **Dataset Preparation**
       - Auto-detect feature dimensions from HDF5 files and extract column headers
       - Select feature columns for noise/masking augmentation
       - Create ``PretrainingDataset`` with dual-view augmentation (masking + noise)

    2. **Model & Training Setup**
       - Initialize ``PosePredTransformer`` in pretraining mode with a projection head
       - Setup AdamW, the requested LR scheduler, and mixed precision scaler
       - Configure the model for self-supervised training (projection head only;
         supervised heads and reconstruction remain disabled)
       - Create CSV logging infrastructure for interval and epoch metrics

    3. **Chunked Training Loop**
       - Process files in chunks to limit memory use
       - Shuffle file order deterministically per epoch
       - For each batch, generate two augmented views of the same window,
         compute the selected self-supervised loss, and update the model
       - Save periodic step checkpoints and interval-best checkpoints

    4. **Epoch Completion & Logging**
       - Calculate average epoch loss across all processed batches
       - Log epoch metrics to CSV with the current learning rate
       - Perform periodic memory cleanup when configured

    5. **Resume Robustness**
       - Supports weights-only restore or true resume
       - True resume restores optimizer/scheduler/scaler state and mid-epoch
         chunk/batch progress when available

    Outputs:
        - Step checkpoints and best-interval checkpoints under ``checkpoint_dir``
        - A final pretraining checkpoint at the end of training
        - CSV training logs with interval and epoch loss progression
    """

    print("Setting up pretraining...")

    # --- File Scanning with Multiprocessing ---
    print("Scanning pretraining data based on metadata file...")
    scan_args = [(directory, filename, sequence_length) for filename in metadata['filename'].tolist()]
    with multiprocessing.Pool() as pool:
        file_results = pool.map(PretrainingDataset._scan_file_info, scan_args)
    
    # Filter out None results from failed scans
    valid_file_info = [res for res in file_results if res is not None]

    # Get necessary parameters from data (respect feature_plan if provided)
    if feature_plan is not None:
        num_features = len(feature_plan.included_columns)
        print(
            f"[PRETRAIN] Features: raw={len(feature_plan.canonical_columns)} included={num_features} "
            f"hash={feature_plan.schema_hash}"
        )
    else:
        num_features = get_num_features_from_first_hdf5(directory)
        print(f"Auto-detected num_features for pretraining: {num_features}")

    # Ensure checkpoints produced during pretraining contain full hyperparameter metadata
    extra_globals = {}
    if feature_plan is not None:
        extra_globals = {
            'feature_schema_hash': feature_plan.schema_hash,
            'feature_columns_included': list(feature_plan.included_columns),
            'feature_columns_excluded': list(feature_plan.excluded_columns),
            'feature_columns_canonical': list(feature_plan.canonical_columns),
            'ablation_spec': dict(feature_plan.spec),
        }

    set_checkpoint_globals(
        run_name=run_name,
        metadata_tag=metadata_tag,
        sequence_length=sequence_length,
        num_features=num_features,
        d_model=d_model,
        nhead=nhead,
        num_encoder_layers=num_encoder_layers,
        num_decoder_layers=num_decoder_layers,
        dim_feedforward=dim_feedforward,
        dropout_rate=dropout_rate,
        frame_representation=frame_representation,
        num_cls=num_cls,
        prediction_head=prediction_head,
        positional_encoding_type=positional_encoding_type,
        # Pretraining only trains the self-supervised projection head.
        # Supervised heads and reconstruction remain disabled in pretraining mode.
        pose_pred=False,
        orient_pred=False,
        recon=False,
        batch=batch,
        scheduler_type=scheduler_type,
        base_lr=base_lr,
        weight_decay=weight_decay,
        warmup_steps=warmup_steps,
        t_0=t_0,
        t_mult=t_mult,
        eta_min=eta_min,
        mask_p=mask_p,
        noise_level=noise_level,
        grad_clip_norm=grad_clip_norm,
        random_seed=random_seed,
        projection_dim=projection_dim,
        temperature=temperature,
        pretrain_files_per_group=pretrain_files_per_group,
        log_interval=log_interval,
        pretrain_loss=pretrain_loss,
        barlow_lambda=barlow_lambda,
        patience=patience,
        plateau_factor=plateau_factor,
        weight_initialization=STANDARD_WEIGHT_INIT_POLICY,
        pretrain_mode=True,
        **extra_globals,
    )
    # Augmentation columns must align with the feature tensor the model sees.
    if feature_plan is not None:
        headers_for_aug = list(feature_plan.included_columns)
    else:
        headers_for_aug = get_headers_from_first_hdf5(directory)
    columns_to_noise = select_columns_for_augmentation(headers_for_aug)
    # Persist in checkpoint globals so downstream stages can reuse it.
    set_checkpoint_globals(columns_to_noise=columns_to_noise)
    print(f"Augmentation columns (pretrain): noise={len(columns_to_noise)} of {len(headers_for_aug)} features")

    # Initialize the dataset
    pretrain_dataset = PretrainingDataset(
        directory,
        metadata,
        sequence_length,
        columns_to_noise=columns_to_noise,
        mask_p=mask_p,
        noise_level=noise_level,
        feature_plan=feature_plan,
        ablation_strict=bool(ablation_strict),
    )
    pretrain_dataset.set_file_info(valid_file_info)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Calculate the actual number of batches per epoch for scheduler
    total_windows = pretrain_dataset.total_windows
    batches_per_epoch = total_windows // batch
    print(f"Total windows: {total_windows}, Batch size: {batch}, Batches per epoch: {batches_per_epoch}")

    # Initialize model, optimizer, scheduler, and scaler
    model, optimizer, scheduler, scaler = model_setup(
        sequence_length, num_features, d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, 
        dropout_rate, frame_representation, num_cls, prediction_head, device, base_lr,
        pose_pred=False, orient_pred=False, pose_dict=None, orient_dict=None, recon=False,
        positional_encoding_type=positional_encoding_type,
        patience=patience, plateau_factor=plateau_factor, pretrain_mode=True, projection_dim=projection_dim,
        batches_per_epoch=batches_per_epoch, # Use actual batches per epoch for scheduler
        scheduler_type=scheduler_type,
        weight_decay=weight_decay,
        warmup_steps=warmup_steps,
        t_0=t_0,
        t_mult=t_mult,
        eta_min=eta_min,
    )

    print("Model and peripherals initialized for pretraining.")

    # Optional pretraining resume / weights-only restore
    load_optimizer_state = bool(checkpoint_path and os.path.isfile(checkpoint_path) and (not restart_optimizer))
    if checkpoint_path and os.path.isfile(checkpoint_path) and not load_optimizer_state:
        print("[CKPT] Pretraining: loading weights only (no optimizer/scheduler/scaler state).")

    resume_pretraining_runtime = None
    if checkpoint_path and os.path.isfile(checkpoint_path):
        try:
            resume_checkpoint = torch.load(checkpoint_path, map_location='cpu')
            if isinstance(resume_checkpoint, dict):
                resume_pretraining_runtime = resume_checkpoint.get('pretraining_runtime_state', None)
        except Exception as e:
            print(f"[CKPT] Warning: could not inspect pretraining checkpoint for runtime state: {e}")

    start_epoch_from_checkpoint, global_step = load_checkpoint_model(
        model, optimizer, scheduler, checkpoint_path, scaler,
        load_optimizer_state=load_optimizer_state, pretrain_mode=True
    )

    restored_pretraining_runtime = _apply_resume_pretraining_runtime_state(resume_pretraining_runtime)
    resuming_mid_epoch = bool(
        load_optimizer_state and isinstance(resume_pretraining_runtime, dict) and
        restored_pretraining_runtime['resume_epoch'] is not None
    )
    start_epoch = int(restored_pretraining_runtime['resume_epoch']) if resuming_mid_epoch else int(start_epoch_from_checkpoint)
    interval_loss = float(restored_pretraining_runtime['interval_loss']) if load_optimizer_state else 0.0
    interval_steps = int(restored_pretraining_runtime['interval_steps']) if load_optimizer_state else 0
    best_interval_loss = float(restored_pretraining_runtime['best_interval_loss']) if load_optimizer_state else float('inf')
    interval_elapsed_s = float(restored_pretraining_runtime['interval_elapsed_s']) if load_optimizer_state else 0.0
    log_start_time = time.time() - interval_elapsed_s if interval_elapsed_s > 0 else time.time()

    if load_optimizer_state and start_epoch > 0 and not isinstance(resume_pretraining_runtime, dict):
        print(
            "[CKPT] Warning: no persisted pretraining_runtime_state was found. "
            "Resume will continue from the next epoch, but mid-epoch progress and interval logging state cannot be recovered exactly."
        )

    # Setup the records file
    records_file, records_writer, records_headers = setup_records_file(
        records_path, start_epoch, False, False, False, pretrain_mode=True, start_global_step=global_step
    )
    if records_file is None or records_writer is None or records_headers is None:
        raise RuntimeError(f"Failed to initialize pretraining records file at {records_path}. See earlier errors above.")

    # --- Pretraining Loop ---
    print("\nStarting pretraining loop...")

    for epoch in range(start_epoch, num_epochs):
        print(f"--- Epoch {epoch + 1}/{num_epochs} ---")
        epoch_start_time = time.time()
        
        # Deterministic file order for this epoch (important for robust resume).
        file_list = list(pretrain_dataset.all_files_info)
        epoch_rng = random.Random(_pretrain_epoch_seed(random_seed, epoch))
        epoch_rng.shuffle(file_list)

        resume_chunk_index = 0
        resume_batch_index = 0
        if resuming_mid_epoch and epoch == start_epoch:
            total_loss = float(restored_pretraining_runtime['epoch_total_loss'])
            num_batches = int(restored_pretraining_runtime['epoch_num_batches'])
            resume_chunk_index = int(restored_pretraining_runtime['next_chunk_index'])
            resume_batch_index = int(restored_pretraining_runtime['next_batch_index_in_chunk'])
            print(
                f"[PRETRAIN RESUME] Resuming epoch {epoch + 1} at chunk {resume_chunk_index + 1}, "
                f"batch {resume_batch_index + 1}."
            )
        else:
            total_loss = 0.0
            num_batches = 0

        num_chunks = math.ceil(len(file_list) / pretrain_files_per_group) if pretrain_files_per_group > 0 else 0

        # Iterate through the files in groups (chunks)
        for chunk_index, i in enumerate(range(0, len(file_list), pretrain_files_per_group)):
            if chunk_index < resume_chunk_index:
                continue
            file_chunk = file_list[i : i + pretrain_files_per_group]
            
            # Time the loading of the file chunk into memory
            print(f"\n--- Loading Chunk {chunk_index + 1}/{num_chunks} ---")
            chunk_load_start_time = time.time()
            pretrain_dataset.load_data_for_files(file_chunk)
            chunk_load_end_time = time.time()
            
            print(f"  Loaded {len(file_chunk)} files with {len(pretrain_dataset)} windows in {chunk_load_end_time - chunk_load_start_time:.2f}s.")

            # Create a DataLoader directly from the dataset for the current chunk.
            # The DataLoader will handle shuffling and batching of windows within this chunk.
            chunk_generator = torch.Generator()
            chunk_generator.manual_seed(_pretrain_chunk_seed(random_seed, epoch, chunk_index))
            chunk_dataloader = DataLoader(
                pretrain_dataset,
                batch_size=batch,
                shuffle=True, # Shuffle windows within the chunk
                num_workers=num_workers,
                pin_memory=True,
                drop_last=True,
                generator=chunk_generator,
                worker_init_fn=_seed_pretrain_worker if num_workers > 0 else None,
            )
            
            print(f"  Processing {len(chunk_dataloader)} batches...")

            batch_start_index = resume_batch_index if (resuming_mid_epoch and epoch == start_epoch and chunk_index == resume_chunk_index) else 0
            if batch_start_index >= len(chunk_dataloader):
                print(f"  Resume point is beyond this chunk; skipping chunk {chunk_index + 1}.")
                continue
            
            windows_processed_in_chunk = 0
            # Inner loop to train on all batches from the current file chunk
            for batch_idx, (view1, view2) in enumerate(chunk_dataloader):
                if batch_idx < batch_start_index:
                    continue

                # Move data to device
                view1 = view1.to(device, non_blocking=True)
                view2 = view2.to(device, non_blocking=True)
                
                # Increment the windows counter for this chunk
                windows_processed_in_chunk += view1.size(0)
                
                optimizer.zero_grad()

                # --- Forward Pass ---
                with torch.amp.autocast(device_type='cuda', enabled=device.type == 'cuda'):
                    # Get the projected embeddings from the model
                    _, _, _, z1 = model(view1)
                    _, _, _, z2 = model(view2)
                    
                    # Calculate contrastive loss based on selection
                    if pretrain_loss == 'info_nce':
                        loss = info_nce_loss(z1, z2, temperature)
                    elif pretrain_loss == 'barlow_twins':
                        loss = barlow_twins_loss(z1, z2, barlow_lambda)
                    else:
                        raise ValueError(f"Unknown pretrain_loss type: {pretrain_loss}")

                # --- Gradient Pass & Optimizer Step ---
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                # Step the scheduler after each batch for proper LR annealing
                scheduler.step()

                interval_loss += loss.item()  # Accumulate loss for interval logging
                interval_steps += 1
                total_loss += loss.item()
                num_batches += 1
                global_step += 1
                
                # Log progress to console
                if batch_idx == batch_start_index or (batch_idx + 1) % 100 == 0 or (batch_idx + 1) == len(chunk_dataloader):
                    current_lr = scheduler.get_last_lr()[0]
                    print(f"      Batch {batch_idx+1}/{len(chunk_dataloader)} | Global Step: {global_step} | Loss: {loss.item():.4f} | LR: {current_lr:.8f}")

                # Log to CSV file every log_interval steps
                if log_interval > 0 and global_step % log_interval == 0 and interval_steps > 0:
                    avg_interval_loss = interval_loss / max(1, interval_steps)
                    current_lr = scheduler.get_last_lr()[0]
                    log_end_time = time.time()
                    time_elapsed = log_end_time - log_start_time
                    save_best_checkpoint = False

                    # Check if this is the best model so far based on interval loss
                    if avg_interval_loss < best_interval_loss:
                        best_interval_loss = avg_interval_loss
                        save_best_checkpoint = True

                    log_data = {
                        'Global Step': global_step,
                        'Epoch': epoch + 1,
                        'Avg Loss (Interval)': avg_interval_loss,
                        'Learning Rate': current_lr,
                        'Time Elapsed (s)': time_elapsed
                    }
                    records_writer.writerow([log_data.get(header, '') for header in records_headers])
                    records_file.flush()

                    # Reset for next interval
                    interval_loss = 0.0
                    interval_steps = 0
                    log_start_time = time.time()

                    if save_best_checkpoint:
                        print(f"  ** New best pretrain loss: {best_interval_loss:.4f}. Saving best model. **")
                        save_checkpoint(
                            epoch=epoch,
                            global_step=global_step,
                            model=model,
                            optimizer=optimizer,
                            scheduler=scheduler,
                            pose_dict=None,
                            orient_dict=None,
                            scaler=scaler,
                            checkpoint_dir=checkpoint_dir,
                            run_name=run_name,
                            pose_pred=False,
                            orient_pred=False,
                            recon=False,
                            split_filenames=None,
                            pretrain_mode=True,
                            filename_override=f"{run_name}_best_pretrain.pth",
                            pretraining_runtime_state=_extract_pretraining_runtime_state(
                                resume_epoch=epoch,
                                next_chunk_index=chunk_index,
                                next_batch_index_in_chunk=batch_idx + 1,
                                epoch_total_loss=total_loss,
                                epoch_num_batches=num_batches,
                                interval_loss=interval_loss,
                                interval_steps=interval_steps,
                                best_interval_loss=best_interval_loss,
                                interval_elapsed_s=0.0,
                            ),
                        )

                # Save checkpoint every checkpoint_interval steps
                if checkpoint_interval > 0 and global_step % checkpoint_interval == 0:
                    save_checkpoint(
                        epoch=epoch,
                        global_step=global_step,
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        pose_dict=None,
                        orient_dict=None,
                        scaler=scaler,
                        checkpoint_dir=checkpoint_dir,
                        run_name=f"{run_name}_step_{global_step}",
                        pose_pred=False,
                        orient_pred=False,
                        recon=False,
                        split_filenames=None,
                        pretrain_mode=True,
                        pretraining_runtime_state=_extract_pretraining_runtime_state(
                            resume_epoch=epoch,
                            next_chunk_index=chunk_index,
                            next_batch_index_in_chunk=batch_idx + 1,
                            epoch_total_loss=total_loss,
                            epoch_num_batches=num_batches,
                            interval_loss=interval_loss,
                            interval_steps=interval_steps,
                            best_interval_loss=best_interval_loss,
                            interval_elapsed_s=max(0.0, time.time() - log_start_time),
                        ),
                    )

            if resuming_mid_epoch and epoch == start_epoch and chunk_index >= resume_chunk_index:
                resume_batch_index = 0

            print(f"  Chunk complete. Total windows processed: {windows_processed_in_chunk}")

        if resuming_mid_epoch and epoch == start_epoch:
            resuming_mid_epoch = False

        # --- End of Epoch ---
        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        current_lr = scheduler.get_last_lr()[0]
        epoch_end_time = time.time()
        epoch_time = epoch_end_time - epoch_start_time
        
        print(f"--- Epoch {epoch + 1} Summary | Avg Loss: {avg_loss:.4f} | LR: {current_lr:.6f} ---")

        # Periodic memory cleanup.
        if MEMORY_CLEANUP_EVERY_EPOCHS and ((epoch + 1) % int(MEMORY_CLEANUP_EVERY_EPOCHS) == 0):
            print(f"[Memory cleanup] pretrain epoch {epoch + 1} (every {MEMORY_CLEANUP_EVERY_EPOCHS} epochs)")
            cleanup_memory()


    # --- End of Training ---
    save_checkpoint(
        epoch=max(0, num_epochs - 1),
        global_step=global_step,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        pose_dict=None,
        orient_dict=None,
        scaler=scaler,
        checkpoint_dir=checkpoint_dir,
        run_name=run_name,
        pose_pred=False,
        orient_pred=False,
        recon=False,
        split_filenames=None,
        pretrain_mode=True,
        filename_override=f"{run_name}_final_pretrain.pth",
        pretraining_runtime_state=_extract_pretraining_runtime_state(
            resume_epoch=max(0, num_epochs - 1),
            next_chunk_index=0,
            next_batch_index_in_chunk=0,
            epoch_total_loss=0.0,
            epoch_num_batches=0,
            interval_loss=interval_loss,
            interval_steps=interval_steps,
            best_interval_loss=best_interval_loss,
            interval_elapsed_s=max(0.0, time.time() - log_start_time),
        ),
    )

    records_file.close()
    print("Pretraining complete.")


class PretrainingDataset(Dataset):
    """
    Memory-efficient dataset for self-supervised pretraining with on-the-fly augmentation.

    Loads HDF5 files into a managed memory buffer for training and generates
    two augmented views of each sequence for the selected self-supervised loss.
    """
    @staticmethod
    def _scan_file_info(args):
        """Static method for parallel file scanning - can be pickled for multiprocessing."""
        directory, filename, sequence_length = args
        file_path = os.path.join(directory, filename)
        
        # Check if file exists before trying to open
        if not os.path.exists(file_path):
            print(f"Warning: File {filename} from metadata not found. Skipping.")
            return None

        try:
            with h5py.File(file_path, 'r') as h5f:
                if 'data' not in h5f:
                    print(f"Warning: 'data' key not found in {filename}. Skipping.")
                    return None
                
                num_rows = h5f['data'].shape[0]
                # Only include files that can produce at least one window
                if num_rows >= sequence_length:
                    num_windows = num_rows - (sequence_length - 1)
                    return {
                        'path': file_path,
                        'num_windows': num_windows,
                        'filename': filename
                    }
        except Exception as e:
            print(f"Warning: Could not read {filename}. Error: {e}. Skipping.")
        return None

    def __init__(
        self,
        directory: str,
        metadata: pd.DataFrame,
        sequence_length: int,
        columns_to_noise: list[int],
        mask_p: float,
        noise_level: float,
        feature_plan: Optional["FeaturePlan"] = None,
        ablation_strict: bool = True,
    ) -> None:
        self.directory = directory
        self.sequence_length = sequence_length
        self.metadata = metadata
        self.columns_to_noise = columns_to_noise
        self.mask_p = mask_p
        self.noise_level = noise_level

        # Feature selection / ablation
        self.feature_plan = feature_plan
        self.ablation_strict = bool(ablation_strict)
        self.included_columns: Optional[list[str]] = None
        if self.feature_plan is not None:
            self.included_columns = list(self.feature_plan.included_columns)
        
        # Data for the files currently loaded into memory.
        self.data_in_memory: dict[str, np.ndarray] = {}
        # This will map a global window index to a file and a local index within that file.
        self.window_map: list[tuple[str, int]] = []
        self.all_files_info: list[dict] = []
        self.total_windows: int = 0

        print("Scanning pretraining data based on metadata file...")
        
        # Use multiprocessing for parallel file scanning
        scan_args = [(self.directory, filename, self.sequence_length) for filename in self.metadata['filename'].tolist()]
        with multiprocessing.Pool() as pool:
            file_results = pool.map(PretrainingDataset._scan_file_info, scan_args)
        
        # Store the file info for all valid files. This doesn't load data, just metadata.
        self.all_files_info = [res for res in file_results if res is not None]
        
        self.total_windows = sum(info['num_windows'] for info in self.all_files_info)
        print(f"Found {len(self.all_files_info)} valid files with {self.total_windows} total windows.")

    def set_file_info(self, file_info_list: list[dict]) -> None:
        """
        Sets the file information for the dataset and calculates total windows.
        This is called after the file info has been scanned externally.
        """
        self.all_files_info = file_info_list
        self.total_windows = sum(info['num_windows'] for info in self.all_files_info)
        print(f"Dataset configured with {len(self.all_files_info)} valid files and {self.total_windows} total windows.")

    def load_data_for_files(self, file_info_list: list[dict]) -> None:
        """
        Loads the data for a specific list of files into RAM.
        This is called by the main training loop to manage chunks.
        """
        # Clear loaded data
        self.data_in_memory.clear()
        self.window_map.clear()

        print(f"Loading {len(file_info_list)} files into memory for the current chunk...")
        
        for file_info in file_info_list:
            file_path = file_info['path']
            try:
                with h5py.File(file_path, 'r') as h5f:
                    # Load the file data into the shared dictionary
                    data_np = h5f['data'][:]

                    # Apply feature selection (if configured)
                    if self.included_columns is not None:
                        # Reuse HDF5 column parsing and name->index mapping from datasets.py
                        from .datasets import _read_headers_from_h5file, _build_keep_indices

                        file_headers = _read_headers_from_h5file(h5f, dataset_key='data')
                        keep_idx = _build_keep_indices(file_headers, self.included_columns)
                        data_np = data_np[:, keep_idx]

                    self.data_in_memory[file_path] = data_np
                
                # Create the window map for the newly loaded file
                for i in range(file_info['num_windows']):
                    self.window_map.append((file_path, i))

            except Exception as e:
                if self.ablation_strict:
                    raise
                print(f"  Warning: Could not load {file_info['filename']}. Error: {e}")
        
        print(f"  Successfully loaded {len(self.data_in_memory)} files.")
        
    def __len__(self) -> int:
        """Returns total number of sequence windows in the currently loaded chunk."""
        return len(self.window_map)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns two differently augmented views of the same sequence for self-supervised learning.
        Assumes that the required data is already in the `data_in_memory` cache.
        """
        if idx >= len(self.window_map):
            raise IndexError(f"Index {idx} is out of bounds for the current chunk of size {len(self.window_map)}.")

        # 1. Get the file path and local start index from the pre-computed map.
        file_path, local_start_idx = self.window_map[idx]
        
        # 2. Get the sequence data directly from the RAM cache.
        sequence_data = self.data_in_memory[file_path]
        window_data = sequence_data[local_start_idx : local_start_idx + self.sequence_length]
        
        # Convert to tensor
        sequence_tensor = torch.from_numpy(window_data).float()

        # 3. Create two different augmentations of the window.
        view1 = augment_features(sequence_tensor, self.mask_p, self.noise_level, self.columns_to_noise)
        view2 = augment_features(sequence_tensor, self.mask_p, self.noise_level, self.columns_to_noise)
        
        return view1, view2


