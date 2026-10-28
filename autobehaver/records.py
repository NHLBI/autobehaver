"""Write PosePred training metrics and ordered confusion matrices to CSV."""

from __future__ import annotations

import csv
import os
import shutil
from datetime import datetime
from typing import Optional, Union

import numpy as np
import torch
from torch.utils.data import DataLoader


__all__ = ['setup_records_file', 'setup_confusion_matrix_files', 'log_epoch_results', 'log_confusion_matrix', 'log_wandb_metrics']

def _record_headers(pose_pred, orient_pred, recon, pretrain_mode):
    if pretrain_mode:
        return ['Global Step', 'Epoch', 'Avg Loss (Interval)', 'Learning Rate', 'Time Elapsed (s)']
    headers = ['Epoch', 'Model Variant', 'LR Phase', 'SWA Active', 'SWA Snapshots Averaged',
               'Validation Available']
    if recon:
        headers += ['Raw Train Recon Loss Avg', 'Weighted Train Recon Loss Avg', 'Val Recon Loss Avg']
    if pose_pred:
        headers += ['Train Pose Loss Avg', 'Train Weighted Pose Loss Avg', 'Val Pose Loss Avg',
                    'Val Weighted Pose Loss Avg', 'Train Pose Accuracy', 'Val Pose Accuracy']
    if orient_pred:
        headers += ['Train Orient Loss Avg', 'Train Weighted Orient Loss Avg', 'Val Orient Loss Avg',
                    'Val Weighted Orient Loss Avg', 'Train Orient Accuracy', 'Val Orient Accuracy']
    headers += ['Train Total Loss Avg', 'Val Total Loss Avg']
    if pose_pred:
        headers += ['Val Pose Macro F1']
    if orient_pred:
        headers += ['Val Orient Macro F1']
    return headers + ['Epoch Time', 'Avg Time per Epoch', 'Forward Pass Time', 'Avg Forward Pass Time',
                      'Optimizer Step Time', 'Avg Optimizer Step Time', 'Epochs Since Last Improvement',
                      'Learning Rate']


def setup_records_file(records_path: str, start_epoch: int, pose_pred: bool, orient_pred: bool, recon: bool, pretrain_mode: bool = False, start_global_step: Optional[int] = None):
    """Back up, trim, and atomically publish the resolved metric schema before append.

    Existing rows retain common metric values. Newly enabled metrics remain blank
    in historical rows rather than borrowing values from another task.
    """
    from pathlib import Path
    import uuid
    path = Path(records_path)
    temporary = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        headers = _record_headers(pose_pred, orient_pred, recon, pretrain_mode)
        retained = []
        if path.exists():
            previous = path.parent / 'previous_records'
            previous.mkdir(exist_ok=True)
            stamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
            backup = previous / f'{path.stem}_prev_{stamp}_{uuid.uuid4().hex[:8]}{path.suffix}'
            shutil.copy2(path, backup)
            print(f'Created backup of existing record: {backup}')
            with path.open('r', newline='') as stream:
                reader = csv.DictReader(stream)
                old_headers = reader.fieldnames or []
                cutoff = int(start_global_step or 0) if pretrain_mode else int(start_epoch)
                for row in reader:
                    position = (row.get(headers[0]) or '').strip()
                    if position.isdigit() and int(position) <= cutoff:
                        retained.append([row.get(header, '') for header in headers])
                if old_headers and old_headers != headers:
                    print('Training record schema updated to the active tasks; previous schema retained in backup.')
        temporary = path.with_name('.records-' + uuid.uuid4().hex + '.tmp')
        with temporary.open('w', newline='') as stream:
            writer = csv.writer(stream)
            writer.writerow(headers)
            writer.writerows(retained)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        file = path.open('a', newline='')
        return file, csv.writer(file), headers
    except Exception as exc:
        print(f'Error in setup_records_file: {exc}')
        return None, None, None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def setup_confusion_matrix_files(val_dataloader: DataLoader, pose_conf_matrix_path: Optional[str], orient_conf_matrix_path: Optional[str], combined_conf_matrix_path: Optional[str],
                                 pose_pred: bool, orient_pred: bool) -> None:
    """Back up and initialize enabled confusion-matrix CSV files.

    Use class order from the underlying validation dataset, including the
    Cartesian pose/orientation order for the combined matrix.
    """
    try:
        original_dataset = val_dataloader.dataset.dataset

        class_names = {
            'pose': list(original_dataset.pose_dict.keys()) if hasattr(original_dataset, 'pose_dict') else [],
            'orient': list(original_dataset.orient_dict.keys()) if hasattr(original_dataset, 'orient_dict') else []
        }

        def create_confusion_matrix_file(filename, class_names):
            try:
                previous_records_dir = os.path.join(os.path.dirname(filename), "previous_records")
                if not os.path.exists(previous_records_dir):
                    os.makedirs(previous_records_dir)

                if os.path.exists(filename):
                    base, ext = os.path.splitext(filename)
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    backup_filename = os.path.join(previous_records_dir, f"{os.path.basename(base)}_prev_{timestamp}{ext}")
                    shutil.copy2(filename, backup_filename)
                    print(f"Created backup of existing confusion matrix file: {backup_filename}")

                with open(filename, 'w', newline='') as file:
                    writer = csv.writer(file)
                    headers = ['epoch'] + [f"{actual}_{predicted}" for actual in class_names for predicted in class_names]
                    writer.writerow(headers)
                    print(f"Created new confusion matrix file: {filename}")
            except Exception as e:
                print(f"Error creating confusion matrix file {filename}: {e}")

        if pose_pred and pose_conf_matrix_path:
            create_confusion_matrix_file(pose_conf_matrix_path, class_names['pose'])

        if orient_pred and orient_conf_matrix_path:
            create_confusion_matrix_file(orient_conf_matrix_path, class_names['orient'])

        if pose_pred and orient_pred and combined_conf_matrix_path:
            pose_class_names = sorted(list(original_dataset.pose_dict.keys()))
            orient_class_names = sorted(list(original_dataset.orient_dict.keys()))
            combined_class_names = [f"{p}-{o}" for p in pose_class_names for o in orient_class_names]
            create_confusion_matrix_file(combined_conf_matrix_path, combined_class_names)

    except Exception as e:
        print(f"Error in setup_confusion_matrix_files: {e}")


def log_epoch_results(file: object, writer: csv.writer, epoch: int, metrics: dict[str, Union[float, int, torch.Tensor]], 
                     headers: list[str], pose_pred: bool, orient_pred: bool, recon: bool) -> None:
    """Append epoch metrics in header order and flush the open record.

    Log zero-based ``epoch`` as epoch + 1. Include enabled task metrics and
    convert scalar tensors to Python numbers.
    """
    log_data = {
        'Epoch': epoch + 1,  
        'Model Variant': metrics.get('model_variant', 'base'),
        'LR Phase': metrics.get('lr_phase', 'main'),
        'SWA Active': int(bool(metrics.get('swa_active', False))),
        'SWA Snapshots Averaged': metrics.get('swa_n_averaged', 0),
        'Validation Available': int(bool(metrics.get('validation_available', True))),
        'Train Total Loss Avg': metrics['train_total_loss_avg'],
        'Val Total Loss Avg': metrics['val_total_loss_avg'],
        'Epoch Time': metrics['epoch_time'],
        'Avg Time per Epoch': metrics['avg_time_per_epoch'],
        'Learning Rate': metrics['current_lr'],
        'Forward Pass Time': metrics['time_forward_pass'],
        'Optimizer Step Time': metrics['time_optimizer_step'],
        'Avg Forward Pass Time': metrics['avg_time_forward_pass'],
        'Avg Optimizer Step Time': metrics['avg_time_optimizer_step'],
        'Epochs Since Last Improvement': metrics['epochs_since_improvement']
    }

    if recon:
        log_data.update({
            'Raw Train Recon Loss Avg': metrics['train_recon_loss_avg'],
            'Weighted Train Recon Loss Avg': metrics['train_weighted_recon_loss_avg'],
            'Val Recon Loss Avg': metrics['val_recon_loss_avg']
        })

    if pose_pred:
        log_data.update({
            'Train Pose Loss Avg': metrics['train_pose_loss_avg'],
            'Train Weighted Pose Loss Avg': metrics['train_weighted_pose_loss_avg'],
            'Val Pose Loss Avg': metrics['val_pose_loss_avg'],
            'Val Weighted Pose Loss Avg': metrics['val_weighted_pose_loss_avg'],
            'Train Pose Accuracy': metrics['train_pose_accuracy'],
            'Val Pose Accuracy': metrics['val_pose_accuracy'],
            'Val Pose Macro F1': metrics.get('val_pose_macro_f1', ''),
        })

    if orient_pred:
        log_data.update({
            'Train Orient Loss Avg': metrics['train_orient_loss_avg'],
            'Train Weighted Orient Loss Avg': metrics['train_weighted_orient_loss_avg'],
            'Val Orient Loss Avg': metrics['val_orient_loss_avg'],
            'Val Weighted Orient Loss Avg': metrics['val_weighted_orient_loss_avg'],
            'Train Orient Accuracy': metrics['train_orient_accuracy'],
            'Val Orient Accuracy': metrics['val_orient_accuracy'],
            'Val Orient Macro F1': metrics.get('val_orient_macro_f1', ''),
        })

    if not metrics.get('validation_available', True):
        for key in log_data:
            if key.startswith('Val '):
                log_data[key] = ''
    for key, value in log_data.items():
        if isinstance(value, torch.Tensor):
            log_data[key] = value.item()

    writer.writerow([log_data.get(header, '') for header in headers])
    file.flush()  


def log_confusion_matrix(conf_matrix: np.ndarray, class_names: Union[list[str], dict.keys], 
                        epoch: int, filename: str) -> None:
    """Append a flattened confusion matrix and epoch to CSV.

    Rows are actual classes and columns predicted classes in ``class_names``
    order. Require the complete square shape and matching existing headers;
    columns are named actual_predicted.
    """
    class_names = list(class_names)
    if tuple(conf_matrix.shape) != (len(class_names), len(class_names)):
        raise ValueError("Confusion matrix shape does not match the complete ordered class names")
    flattened = conf_matrix.flatten()
    
    headers = ['epoch'] + [f"{actual}_{predicted}" for actual in class_names for predicted in class_names]
    
    file_exists = os.path.isfile(filename)
    if file_exists:
        with open(filename, newline='') as existing:
            if next(csv.reader(existing), None) != headers:
                raise ValueError(f"Confusion CSV header does not match class order: {filename}")
    
    with open(filename, 'a', newline='') as file:
        writer = csv.writer(file)
        
        if not file_exists:
            writer.writerow(headers)
        
        writer.writerow([epoch] + list(flattened))


def log_wandb_metrics(epoch: int, metrics: dict[str, Union[float, int, torch.Tensor]], 
                     pose_pred: bool, orient_pred: bool, recon: bool) -> None:
    """Log enabled epoch metrics to an initialized optional Weights & Biases run.

    The CLI enables this explicitly and initializes offline logging. Other
    callers choose the run mode. Convert scalar tensors to numbers and log
    zero-based ``epoch`` as epoch + 1.
    """
    import wandb  # local import: optional dependency
    wandb_log_data = {
        'Epoch': epoch + 1,
        'Model Variant': metrics.get('model_variant', 'base'),
        'LR Phase': metrics.get('lr_phase', 'main'),
        'SWA Active': int(bool(metrics.get('swa_active', False))),
        'SWA Snapshots Averaged': metrics.get('swa_n_averaged', 0),
        'Train Total Loss Avg': metrics['train_total_loss_avg'],
        'Val Total Loss Avg': metrics['val_total_loss_avg'],
        'Epoch Time': metrics['epoch_time'],
        'Avg Time per Epoch': metrics['avg_time_per_epoch'],
        'Learning Rate': metrics['current_lr'],
        'Forward Pass Time': metrics['time_forward_pass'],
        'Optimizer Step Time': metrics['time_optimizer_step'],
        'Avg Forward Pass Time': metrics['avg_time_forward_pass'],
        'Avg Optimizer Step Time': metrics['avg_time_optimizer_step'],
        'Epochs Since Last Improvement': metrics['epochs_since_improvement']
    }
    if recon:
        wandb_log_data.update({
            'Raw Train Recon Loss Avg': metrics['train_recon_loss_avg'],
            'Weighted Train Recon Loss Avg': metrics['train_weighted_recon_loss_avg'],
            'Val Recon Loss Avg': metrics['val_recon_loss_avg']
        })
    if pose_pred:
        wandb_log_data.update({
            'Train Pose Loss Avg': metrics['train_pose_loss_avg'],
            'Train Weighted Pose Loss Avg': metrics['train_weighted_pose_loss_avg'],
            'Val Pose Loss Avg': metrics['val_pose_loss_avg'],
            'Val Weighted Pose Loss Avg': metrics['val_weighted_pose_loss_avg'],
            'Train Pose Accuracy': metrics['train_pose_accuracy'],
            'Val Pose Accuracy': metrics['val_pose_accuracy'],
            'Val Pose Macro F1': metrics.get('val_pose_macro_f1', ''),
        })
    if orient_pred:
        wandb_log_data.update({
            'Train Orient Loss Avg': metrics['train_orient_loss_avg'],
            'Train Weighted Orient Loss Avg': metrics['train_weighted_orient_loss_avg'],
            'Val Orient Loss Avg': metrics['val_orient_loss_avg'],
            'Val Weighted Orient Loss Avg': metrics['val_weighted_orient_loss_avg'],
            'Train Orient Accuracy': metrics['train_orient_accuracy'],
            'Val Orient Accuracy': metrics['val_orient_accuracy'],
            'Val Orient Macro F1': metrics.get('val_orient_macro_f1', ''),
        })
    for key, value in wandb_log_data.items():
        if isinstance(value, torch.Tensor):
            wandb_log_data[key] = value.item()
    wandb.log(wandb_log_data)


