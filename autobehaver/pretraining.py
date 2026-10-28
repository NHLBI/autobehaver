"""Pretrain the PosePred encoder with paired-view InfoNCE or Barlow Twins loss."""

from __future__ import annotations

from autobehaver.progress import progress

import math
import multiprocessing
import os
import random
import sys
import time
from typing import Optional, TYPE_CHECKING

import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from .constants import MEMORY_CLEANUP_EVERY_EPOCHS, STANDARD_WEIGHT_INIT_POLICY
from .augmentation import augment_features, select_columns_for_augmentation
from .datasets import validate_pretraining_metadata
from .losses import info_nce_loss, barlow_twins_loss
from ._training_core import model_setup
from .checkpoint import load_checkpoint_model, save_checkpoint, set_checkpoint_globals
from .io import get_num_features_from_first_hdf5, get_headers_from_first_hdf5
from .records import setup_records_file
from .resource_monitor import record_work
from .utils import cleanup_memory
from .window_index import WindowInstructions, DiskFeatureStore
from .training_identity import build_training_data_identity, validate_training_data_identity

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
    """Train the encoder and projection head on two augmented views per window.

    Disable supervised and reconstruction heads. Process validated files in
    chunks, preserving deterministic file/window order for resume. A true
    resume restores training state and recorded chunk/batch progress; a
    weights-only restart begins fresh optimization. Write CSV metrics and
    periodic, interval-best and final checkpoints.
    """

    print("Setting up pretraining...")

    if checkpoint_path and not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f'Requested checkpoint does not exist: {checkpoint_path}')
    metadata = validate_pretraining_metadata(
        directory,
        metadata,
        sequence_length=sequence_length,
        scan_values=not ablation_strict,
    )

    training_identity = build_training_data_identity(directory, metadata,
        sequence_length=sequence_length, pose_pred=False, orient_pred=False)
    training_identity['ordered_filenames'] = [str(name).replace('\\', '/') for name in metadata['filename']]
    split_filenames = {'train_filenames': training_identity['ordered_filenames'],
                      'val_filenames': [], 'training_data_identity': training_identity}
    if checkpoint_path and not restart_optimizer:
        saved_checkpoint = torch.load(checkpoint_path, map_location='cpu')
        saved_identity = saved_checkpoint.get('training_data_identity')
        validate_training_data_identity(saved_identity, training_identity)
        if saved_identity is not None and saved_identity.get('ordered_filenames') != training_identity['ordered_filenames']:
            raise ValueError('Pretraining metadata order changed since the checkpoint; chunk resume requires the same ordered inventory.')

    print("Scanning pretraining data based on metadata file...")
    scan_args = [(directory, filename, sequence_length) for filename in metadata['filename'].tolist()]
    # Scan metadata sequentially; data-loader workers handle training batches.
    file_results = [PretrainingDataset._scan_file_info(item) for item in scan_args]
    
    valid_file_info = [res for res in file_results if res is not None]
    if len(valid_file_info) != len(metadata):
        raise RuntimeError(
            "Pretraining input inventory changed or became unreadable after validation; "
            f"validated {len(metadata)} file(s), scan retained {len(valid_file_info)}."
        )

    if feature_plan is not None:
        num_features = len(feature_plan.included_columns)
        print(
            f"[PRETRAIN] Features: raw={len(feature_plan.canonical_columns)} included={num_features} "
            f"hash={feature_plan.schema_hash}"
        )
    else:
        num_features = get_num_features_from_first_hdf5(directory)
        print(f"Auto-detected num_features for pretraining: {num_features}")

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
        reset=True,
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
        # Train the encoder through its projection head; supervised heads are disabled.
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
        num_epochs=num_epochs,
        checkpoint_interval=checkpoint_interval,
        **extra_globals,
    )
    # Augmentation columns must align with the feature tensor the model sees.
    if feature_plan is not None:
        headers_for_aug = list(feature_plan.included_columns)
    else:
        headers_for_aug = get_headers_from_first_hdf5(directory)
    columns_to_noise = select_columns_for_augmentation(headers_for_aug)
    set_checkpoint_globals(columns_to_noise=columns_to_noise)
    print(f"Augmentation columns (pretrain): noise={len(columns_to_noise)} of {len(headers_for_aug)} features")

    pretrain_dataset = PretrainingDataset(
        directory,
        metadata,
        sequence_length,
        columns_to_noise=columns_to_noise,
        mask_p=mask_p,
        noise_level=noise_level,
        feature_plan=feature_plan,
        ablation_strict=bool(ablation_strict),
        num_workers=num_workers,
        file_info=valid_file_info,
    )
    
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")

        # Calculate the actual number of batches per epoch for scheduler
        total_windows = pretrain_dataset.total_windows
        batches_per_epoch = total_windows // batch
        print(f"Total windows: {total_windows}, Batch size: {batch}, Batches per epoch: {batches_per_epoch}")

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

        records_file, records_writer, records_headers = setup_records_file(
            records_path, start_epoch, False, False, False, pretrain_mode=True, start_global_step=global_step
        )
        if records_file is None or records_writer is None or records_headers is None:
            raise RuntimeError(f"Failed to initialize pretraining records file at {records_path}. See earlier errors above.")

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

            for chunk_index, i in enumerate(range(0, len(file_list), pretrain_files_per_group)):
                if chunk_index < resume_chunk_index:
                    continue
                file_chunk = file_list[i : i + pretrain_files_per_group]
            
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
                    shuffle=True, 
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
                chunk_iterator = iter(chunk_dataloader)
                for batch_idx, (view1, view2) in enumerate(progress(chunk_iterator, desc='Pretraining batches')):
                    if batch_idx < batch_start_index:
                        continue

                    view1 = view1.to(device, non_blocking=True)
                    view2 = view2.to(device, non_blocking=True)
                
                    windows_processed_in_chunk += view1.size(0)
                
                    optimizer.zero_grad()

                    with torch.amp.autocast(device_type='cuda', enabled=device.type == 'cuda'):
                        _, _, _, z1 = model(view1)
                        _, _, _, z2 = model(view2)
                    
                        if pretrain_loss == 'info_nce':
                            loss = info_nce_loss(z1, z2, temperature)
                        elif pretrain_loss == 'barlow_twins':
                            loss = barlow_twins_loss(z1, z2, barlow_lambda)
                        else:
                            raise ValueError(f"Unknown pretrain_loss type: {pretrain_loss}")

                    scaler.scale(loss).backward()
                    if grad_clip_norm > 0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                    scaler.step(optimizer)
                    scaler.update()

                    # Step the scheduler after each batch for proper LR annealing
                    if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                        scheduler.step(float(loss.detach().item()))
                    else:
                        scheduler.step()

                    interval_loss += loss.item()  
                    interval_steps += 1
                    total_loss += loss.item()
                    num_batches += 1
                    global_step += 1
                
                    if batch_idx == batch_start_index or (batch_idx + 1) % 100 == 0 or (batch_idx + 1) == len(chunk_dataloader):
                        current_lr = scheduler.get_last_lr()[0]
                        print(f"      Batch {batch_idx+1}/{len(chunk_dataloader)} | Global Step: {global_step} | Loss: {loss.item():.4f} | LR: {current_lr:.8f}")

                    if log_interval > 0 and global_step % log_interval == 0 and interval_steps > 0:
                        avg_interval_loss = interval_loss / max(1, interval_steps)
                        current_lr = scheduler.get_last_lr()[0]
                        log_end_time = time.time()
                        time_elapsed = log_end_time - log_start_time
                        save_best_checkpoint = False

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
                                split_filenames=split_filenames,
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
                            split_filenames=split_filenames,
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

            avg_loss = total_loss / num_batches if num_batches > 0 else 0
            current_lr = scheduler.get_last_lr()[0]
            epoch_end_time = time.time()
            epoch_time = epoch_end_time - epoch_start_time
            record_work("pretraining_epochs", completed=1)
        
            print(f"--- Epoch {epoch + 1} Summary | Avg Loss: {avg_loss:.4f} | LR: {current_lr:.6f} ---")

            if MEMORY_CLEANUP_EVERY_EPOCHS and ((epoch + 1) % int(MEMORY_CLEANUP_EVERY_EPOCHS) == 0):
                print(f"[Memory cleanup] pretrain epoch {epoch + 1} (every {MEMORY_CLEANUP_EVERY_EPOCHS} epochs)")
                cleanup_memory()


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
            split_filenames=split_filenames,
            pretrain_mode=True,
            filename_override=f"{run_name}_final_pretrain.pth",
            pretraining_runtime_state=_extract_pretraining_runtime_state(
                resume_epoch=num_epochs,
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

    finally:
        if 'chunk_iterator' in locals() and hasattr(chunk_iterator, '_shutdown_workers'):
            chunk_iterator._shutdown_workers()
        if 'records_file' in locals():
            records_file.close()
        if 'chunk_dataloader' in locals():
            del chunk_dataloader
        store = pretrain_dataset.data_in_memory
        if isinstance(store, DiskFeatureStore):
            error = sys.exc_info()
            store.close(error if error[0] is not None else None)


class PretrainingDataset(Dataset):
    """Load one file chunk and generate paired augmented views of each window."""
    @staticmethod
    def _scan_file_info(args):
        """Read one file's path and complete-window count without loading features."""
        directory, filename, sequence_length = args
        file_path = os.path.join(directory, filename)
        
        if not os.path.exists(file_path):
            print(f"Warning: File {filename} from metadata not found. Skipping.")
            return None

        try:
            with h5py.File(file_path, 'r') as h5f:
                if 'data' not in h5f:
                    print(f"Warning: 'data' key not found in {filename}. Skipping.")
                    return None
                
                num_rows = h5f['data'].shape[0]
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
        num_workers: int = 0,
        file_info=None,
    ) -> None:
        self.directory = directory
        self.sequence_length = sequence_length
        self.metadata = metadata
        self.columns_to_noise = columns_to_noise
        self.mask_p = mask_p
        self.noise_level = noise_level
        self.num_workers = int(num_workers)
        self.inference_mode = True  # Window index views only; pretraining has no supervised labels.

        self.feature_plan = feature_plan
        self.ablation_strict = bool(ablation_strict)
        self.included_columns: Optional[list[str]] = None
        if self.feature_plan is not None:
            self.included_columns = list(self.feature_plan.included_columns)
        
        self.data_in_memory = DiskFeatureStore('pretraining') if self.num_workers > 0 else {}
        # This will map a global window index to a file and a local index within that file.
        self.window_map: list[tuple[str, int]] = []
        self.all_files_info: list[dict] = []
        self.total_windows: int = 0

        print("Scanning pretraining data based on metadata file...")
        
        scan_args = [(self.directory, filename, self.sequence_length) for filename in self.metadata['filename'].tolist()]
        file_results = file_info if file_info is not None else [self._scan_file_info(item) for item in scan_args]
        
        # Store the file info for all valid files. This doesn't load data, just metadata.
        self.all_files_info = [res for res in file_results if res is not None]
        
        self.total_windows = sum(info['num_windows'] for info in self.all_files_info)
        print(f"Found {len(self.all_files_info)} valid files with {self.total_windows} total windows.")

    def set_file_info(self, file_info_list: list[dict]) -> None:
        """Set the scanned file inventory and total available window count."""
        self.all_files_info = file_info_list
        self.total_windows = sum(info['num_windows'] for info in self.all_files_info)
        print(f"Dataset configured with {len(self.all_files_info)} valid files and {self.total_windows} total windows.")

    def load_data_for_files(self, file_info_list: list[dict]) -> None:
        """Replace cached features and the window index with the requested file chunk."""
        self.data_in_memory.clear()
        self.window_map = []
        segments = []

        print(f"Loading {len(file_info_list)} files into memory for the current chunk...")
        
        for file_info in file_info_list:
            file_path = file_info['path']
            try:
                with h5py.File(file_path, 'r') as h5f:
                    data_np = h5f['data'][:]
                    if not np.isfinite(data_np).all():
                        raise ValueError(f'Nonfinite pretraining features in {file_path}')
                    if len(data_np) - self.sequence_length + 1 != file_info['num_windows']:
                        raise ValueError(f'Pretraining frame inventory changed: {file_path}')

                    if self.included_columns is not None:
                        # Reuse HDF5 column parsing and name->index mapping from datasets.py
                        from .datasets import _read_headers_from_h5file, _build_keep_indices

                        file_headers = _read_headers_from_h5file(h5f, dataset_key='data')
                        keep_idx = _build_keep_indices(file_headers, self.included_columns)
                        data_np = data_np[:, keep_idx]

                    data_np = np.ascontiguousarray(data_np, dtype=np.float32)
                    if isinstance(self.data_in_memory, DiskFeatureStore):
                        self.data_in_memory.add(file_path, data_np)
                    else:
                        self.data_in_memory[file_path] = data_np
                
                segments.append({'path': file_path, 'count': file_info['num_windows'],
                                 'starts': None, 'source_start': 0})

            except Exception as e:
                if self.ablation_strict:
                    raise
                print(f"  Warning: Could not load {file_info['filename']}. Error: {e}")
        
        self.window_map = WindowInstructions(self, segments)
        print(f"  Successfully loaded {len(self.data_in_memory)} files.")
        
    def __len__(self) -> int:
        """Return the window count in the currently loaded file chunk."""
        return len(self.window_map)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return two independently augmented (T, F) views of a cached window."""
        if idx >= len(self.window_map):
            raise IndexError(f"Index {idx} is out of bounds for the current chunk of size {len(self.window_map)}.")

        instruction = self.window_map[idx]
        file_path, local_start_idx = instruction['input_file'], instruction['sequence_start']
        
        sequence_data = self.data_in_memory[file_path]
        window_data = sequence_data[local_start_idx : local_start_idx + self.sequence_length]
        
        sequence_tensor = torch.from_numpy(window_data).float()

        view1 = augment_features(sequence_tensor, self.mask_p, self.noise_level, self.columns_to_noise)
        view2 = augment_features(sequence_tensor, self.mask_p, self.noise_level, self.columns_to_noise)
        
        return view1, view2
