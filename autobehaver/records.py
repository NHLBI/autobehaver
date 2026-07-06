"""Training record helpers.

Utilities for writing per-epoch/per-step metrics to CSV, tracking best validation metrics, and
saving lightweight summaries alongside checkpoints."""

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

def setup_records_file(records_path: str, start_epoch: int, pose_pred: bool, orient_pred: bool, recon: bool, pretrain_mode: bool = False, start_global_step: Optional[int] = None):
    """
    Setup the records file for logging training progress.

    This function manages the creation and maintenance of a CSV file used to log training metrics.
    It performs the following tasks:
    - Creates a backup of the existing records file, if it exists, in a 'previous_records' subdirectory.
    - Initializes a new records file with appropriate headers if no file exists.

    Parameters:
    - records_path: The path to the CSV file where records are logged.
    - start_epoch: The epoch number from which training is resumed.
    - pose_pred: Boolean indicating if pose prediction is enabled.
    - orient_pred: Boolean indicating if orientation prediction is enabled.
    - recon: Boolean indicating if reconstruction is enabled.
    - pretrain_mode: Boolean indicating if pretraining is enabled.

    Returns:
    - file: The file object for the records file.
    - writer: The CSV writer object for appending records.
    - headers: The list of headers used in the records file.
    """
    try:
        # Create a subdirectory for previous records if it doesn't exist
        previous_records_dir = os.path.join(os.path.dirname(records_path), "previous_records")
        if not os.path.exists(previous_records_dir):
            os.makedirs(previous_records_dir)

        # First, make a backup copy of the existing record if it exists
        if os.path.exists(records_path):
            base_path, ext = os.path.splitext(records_path)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = os.path.join(previous_records_dir, f"{os.path.basename(base_path)}_prev_{timestamp}{ext}")
            shutil.copy2(records_path, backup_path)
            print(f"Created backup of existing record: {backup_path}")

            # Trim the file to only include metrics up to the resume point.
            trimmed_rows = []
            with open(records_path, 'r', newline='') as file:
                reader = csv.reader(file)
                header = next(reader)
                trimmed_rows.append(header)
                for row in reader:
                    if not row or not row[0].strip().isdigit():
                        continue
                    if pretrain_mode:
                        row_step = int(row[0])
                        cutoff_step = int(start_global_step or 0)
                        if row_step <= cutoff_step:
                            trimmed_rows.append(row)
                        else:
                            break
                    else:
                        epoch_num = int(row[0])
                        if epoch_num <= start_epoch:
                            trimmed_rows.append(row)
                        else:
                            break  # Stop at the first row beyond the checkpoint

            # Overwrite the file with the trimmed rows
            with open(records_path, 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerows(trimmed_rows)
            if pretrain_mode:
                print(f"Trimmed record file to global step {int(start_global_step or 0)}")
            else:
                print(f"Trimmed record file to checkpoint epoch {start_epoch}")

        # If no file exists, create a new one with headers
        if not os.path.exists(records_path) or os.path.getsize(records_path) == 0:
            with open(records_path, 'w', newline='') as file:
                writer = csv.writer(file)
                # Define the full order of headers assuming all flags are true
                
                if pretrain_mode:
                    headers = [
                        'Global Step', 'Epoch', 'Avg Loss (Interval)', 'Learning Rate', 'Time Elapsed (s)'
                    ]
                else:
                    full_headers = [
                        'Epoch', 'Model Variant', 'LR Phase', 'SWA Active', 'SWA Snapshots Averaged',
                        'Raw Train Recon Loss Avg', 'Weighted Train Recon Loss Avg', 'Val Recon Loss Avg',
                        'Train Pose Loss Avg', 'Train Weighted Pose Loss Avg', 'Val Pose Loss Avg', 'Val Weighted Pose Loss Avg',
                        'Train Pose Accuracy', 'Val Pose Accuracy',
                        'Train Orient Loss Avg', 'Train Weighted Orient Loss Avg', 'Val Orient Loss Avg', 'Val Weighted Orient Loss Avg',
                        'Train Orient Accuracy', 'Val Orient Accuracy',
                        'Train Total Loss Avg', 'Val Total Loss Avg',
                        'Val Pose Macro F1', 'Val Orient Macro F1',
                        'Epoch Time', 'Avg Time per Epoch',
                        'Forward Pass Time', 'Avg Forward Pass Time',
                        'Optimizer Step Time', 'Avg Optimizer Step Time',
                        'Epochs Since Last Improvement',
                        'Learning Rate'
                    ]
                    
                    # Filter headers based on flags
                    headers = [header for header in full_headers if (
                        header in ['Epoch', 'Model Variant', 'LR Phase', 'SWA Active', 'SWA Snapshots Averaged',
                                   'Train Total Loss Avg', 'Val Total Loss Avg',
                                   'Epoch Time', 'Avg Time per Epoch',
                                   'Forward Pass Time', 'Avg Forward Pass Time',
                                   'Optimizer Step Time', 'Avg Optimizer Step Time',
                                   'Epochs Since Last Improvement',
                                   'Learning Rate'] or
                        (recon and header in ['Raw Train Recon Loss Avg', 'Weighted Train Recon Loss Avg', 'Val Recon Loss Avg']) or
                        (pose_pred and header in ['Train Pose Loss Avg', 'Train Weighted Pose Loss Avg', 'Val Pose Loss Avg', 'Val Weighted Pose Loss Avg',
                                                  'Train Pose Accuracy', 'Val Pose Accuracy', 'Val Pose Macro F1']) or
                        (orient_pred and header in ['Train Orient Loss Avg', 'Train Weighted Orient Loss Avg', 'Val Orient Loss Avg', 'Val Weighted Orient Loss Avg',
                                                    'Train Orient Accuracy', 'Val Orient Accuracy', 'Val Orient Macro F1'])
                    )]

                writer.writerow(headers)
                print(f"Created new record file: {records_path}")

        # Reopen the file in read mode to read headers
        with open(records_path, 'r', newline='') as file:
            reader = csv.reader(file)
            headers = next(reader)

        # Reopen the file in append mode for further writing
        file = open(records_path, 'a', newline='')
        writer = csv.writer(file)

        return file, writer, headers

    except Exception as e:
        print(f"Error in setup_records_file: {e}")
        return None, None, None


def setup_confusion_matrix_files(val_dataloader: DataLoader, pose_conf_matrix_path: Optional[str], orient_conf_matrix_path: Optional[str], combined_conf_matrix_path: Optional[str],
                                 pose_pred: bool, orient_pred: bool) -> None:
    """
    Initialize confusion matrix files for pose and orientation predictions.

    This function creates CSV files to log confusion matrices for pose and orientation predictions.
    It backs up existing files and initializes new ones with headers based on class names.

    Parameters:
    - val_dataloader: DataLoader containing the validation dataset with class mappings.
    - pose_conf_matrix_path: File path for the pose confusion matrix CSV.
    - orient_conf_matrix_path: File path for the orientation confusion matrix CSV.
    - pose_pred: Boolean indicating if pose prediction is enabled.
    - orient_pred: Boolean indicating if orientation prediction is enabled.
    """
    try:
        # Access the original dataset, which is wrapped by the Subset object
        original_dataset = val_dataloader.dataset.dataset

        # Extract class names from the original dataset
        class_names = {
            'pose': list(original_dataset.pose_dict.keys()) if hasattr(original_dataset, 'pose_dict') else [],
            'orient': list(original_dataset.orient_dict.keys()) if hasattr(original_dataset, 'orient_dict') else []
        }

        def create_confusion_matrix_file(filename, class_names):
            try:
                # Create a subdirectory for previous records if it doesn't exist
                previous_records_dir = os.path.join(os.path.dirname(filename), "previous_records")
                if not os.path.exists(previous_records_dir):
                    os.makedirs(previous_records_dir)

                if os.path.exists(filename):
                    # Backup existing file to the 'previous_records' directory
                    base, ext = os.path.splitext(filename)
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    backup_filename = os.path.join(previous_records_dir, f"{os.path.basename(base)}_prev_{timestamp}{ext}")
                    shutil.copy2(filename, backup_filename)
                    print(f"Created backup of existing confusion matrix file: {backup_filename}")

                # Create a new file with headers
                with open(filename, 'w', newline='') as file:
                    writer = csv.writer(file)
                    headers = ['epoch'] + [f"{actual}_{predicted}" for actual in class_names for predicted in class_names]
                    writer.writerow(headers)
                    print(f"Created new confusion matrix file: {filename}")
            except Exception as e:
                print(f"Error creating confusion matrix file {filename}: {e}")

        # Create confusion matrix files based on flags
        if pose_pred and pose_conf_matrix_path:
            create_confusion_matrix_file(pose_conf_matrix_path, class_names['pose'])

        if orient_pred and orient_conf_matrix_path:
            create_confusion_matrix_file(orient_conf_matrix_path, class_names['orient'])

        # Create combined confusion matrix file
        if pose_pred and orient_pred and combined_conf_matrix_path:
            pose_class_names = sorted(list(original_dataset.pose_dict.keys()))
            orient_class_names = sorted(list(original_dataset.orient_dict.keys()))
            combined_class_names = [f"{p}-{o}" for p in pose_class_names for o in orient_class_names]
            create_confusion_matrix_file(combined_conf_matrix_path, combined_class_names)

    except Exception as e:
        print(f"Error in setup_confusion_matrix_files: {e}")


def log_epoch_results(file: object, writer: csv.writer, epoch: int, metrics: dict[str, Union[float, int, torch.Tensor]], 
                     headers: list[str], pose_pred: bool, orient_pred: bool, recon: bool) -> None:
    """
    Logs comprehensive training metrics for a single epoch to a structured CSV file.
    
    This function orchestrates the systematic recording of all training and validation metrics
    from a completed epoch, ensuring consistent data logging across different training configurations.
    It dynamically constructs the log data based on enabled prediction heads, handles tensor-to-scalar
    conversion for numerical values, and maintains proper column ordering for downstream analysis.
    
    Logging Process:
    1. **Core Metrics Assembly**: Constructs base metrics common to all training modes
    2. **Conditional Metrics**: Adds reconstruction, pose, and orientation metrics based on enabled heads
    3. **Data Type Conversion**: Converts PyTorch tensors to Python scalars for CSV compatibility
    4. **Header-Ordered Writing**: Ensures data is written in the exact order specified by headers
    5. **Immediate Flushing**: Forces data to disk for real-time monitoring and crash recovery
    
    Logged Metrics Categories:
    - **Training Progress**: Epoch number, total losses, accuracy metrics
    - **Performance Timing**: Forward pass, optimizer step, and total epoch times
    - **Learning Dynamics**: Current learning rate and epochs since improvement
    - **Task-Specific**: Reconstruction losses, pose/orientation accuracies and F1 scores
    - **Class-Weighted**: Separate weighted loss tracking for imbalanced datasets
    
    Data Format Handling:
    - Automatically detects and converts torch.Tensor values to Python scalars
    - Handles missing metrics gracefully with empty string defaults
    - Maintains numerical precision for loss values and timing measurements
    - Ensures consistent formatting across different metric types
    
    Parameters:
    - file: Open file object for the CSV file where training records are appended.
    - writer: CSV writer object configured for the specific file and formatting.
    - epoch: Current epoch number (0-indexed) for identification in logs.
    - metrics: Dictionary containing all computed metrics from training and validation.
    - headers: List of column headers defining the exact order and structure of CSV output.
    - pose_pred: Whether pose classification metrics should be included in the log.
    - orient_pred: Whether orientation classification metrics should be included in the log.
    - recon: Whether reconstruction loss metrics should be included in the log.
    
    Returns:
    - None: Data is written directly to the CSV file with immediate disk flushing.
    
    Notes:
    - Supports flexible metric inclusion based on training configuration flags
    - Handles checkpoint resume scenarios
    - Essential for training monitoring, early stopping decisions, and post-training analysis
    - Handles both weighted and unweighted loss variants for comprehensive tracking
    """
    # Construct log data dynamically based on the provided metrics
    log_data = {
        'Epoch': epoch + 1,  # Increment epoch by 1 for logging
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

    # Add reconstruction-related metrics if reconstruction is enabled
    if recon:
        log_data.update({
            'Raw Train Recon Loss Avg': metrics['train_recon_loss_avg'],
            'Weighted Train Recon Loss Avg': metrics['train_weighted_recon_loss_avg'],
            'Val Recon Loss Avg': metrics['val_recon_loss_avg']
        })

    # Add pose prediction-related metrics if pose prediction is enabled
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

    # Add orientation prediction-related metrics if orientation prediction is enabled
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

    # Convert any tensor values to regular numbers for logging
    for key, value in log_data.items():
        if isinstance(value, torch.Tensor):
            log_data[key] = value.item()

    # Write the log data in the order of headers to the CSV file
    writer.writerow([log_data.get(header, '') for header in headers])
    file.flush()  # Ensure data is written to the file immediately


def log_confusion_matrix(conf_matrix: np.ndarray, class_names: Union[list[str], dict.keys], 
                        epoch: int, filename: str) -> None:
    """
    Records detailed confusion matrix data to CSV files for comprehensive classification analysis.
    
    This function provides systematic logging of confusion matrices across training epochs,
    enabling detailed error analysis, class-specific performance tracking, and model behavior
    assessment over time. The matrices are flattened and stored with descriptive headers for
    easy interpretation and downstream analysis with data science tools.
    
    Matrix Processing:
    1. **Flattening**: Converts 2D confusion matrix to 1D array for CSV row storage
    2. **Header Generation**: Creates descriptive column names in 'actual_predicted' format
    3. **File Management**: Creates new files with headers or appends to existing files
    4. **Data Writing**: Records epoch number alongside complete matrix data
    5. **Format Consistency**: Maintains uniform structure across all logged epochs
    
    Confusion Matrix Structure:
    - **Rows**: True/actual class labels as they appear in the dataset
    - **Columns**: Predicted class labels as output by the model
    - **Values**: Count of samples for each (actual, predicted) class pair
    - **Headers**: Formatted as '{actual_class}_{predicted_class}' for clarity
    
    Analysis Applications:
    - **Error Patterns**: Identify which classes are commonly confused with each other
    - **Class Performance**: Track per-class precision and recall trends over epochs
    - **Model Convergence**: Monitor reduction in off-diagonal confusion over training
    - **Dataset Insights**: Reveal inherent ambiguities or labeling inconsistencies
    
    File Format:
    - CSV format with epoch column followed by flattened matrix values
    - Headers written only for new files to avoid duplication
    - Chronological ordering enables time-series analysis of classification performance
    - Compatible with pandas, R, and other data analysis frameworks
    
    Parameters:
    - conf_matrix: 2D numpy array containing the confusion matrix with shape (n_classes, n_classes).
    - class_names: Collection of class names corresponding to matrix rows/columns (list or dict.keys).
    - epoch: Current epoch number for temporal tracking and identification.
    - filename: File path where confusion matrix data will be appended or created.
    
    Returns:
    - None: Data is written directly to the specified CSV file.
    
    Notes:
    - Automatically handles file creation with appropriate headers
    - Maintains chronological order for temporal analysis of model performance
    - Essential for understanding model behavior beyond simple accuracy metrics
    - Supports both pose and orientation classification tracking simultaneously
    """
    # Flatten the confusion matrix to a 1D array for easy logging
    flattened = conf_matrix.flatten()
    
    # Create column headers using class names, formatted as "actual_predicted"
    headers = ['epoch'] + [f"{actual}_{predicted}" for actual in class_names for predicted in class_names]
    
    # Check if the file exists to determine if headers need to be written
    file_exists = os.path.isfile(filename)
    
    # Open the file in append mode to add new data
    with open(filename, 'a', newline='') as file:
        writer = csv.writer(file)
        
        # Write headers if the file is new (i.e., it doesn't exist yet)
        if not file_exists:
            writer.writerow(headers)
        
        # Write the epoch number followed by the flattened confusion matrix data
        writer.writerow([epoch] + list(flattened))


def log_wandb_metrics(epoch: int, metrics: dict[str, Union[float, int, torch.Tensor]], 
                     pose_pred: bool, orient_pred: bool, recon: bool) -> None:
    """
    Uploads comprehensive training metrics to Weights & Biases for real-time experiment tracking.
    
    This function synchronizes training progress with the Weights & Biases platform, enabling
    real-time monitoring, experiment comparison, and collaborative model development. It processes
    and uploads all relevant metrics while handling tensor-to-scalar conversion and conditional
    metric inclusion based on training configuration.
    
    WandB Integration Benefits:
    - **Real-time Monitoring**: Live tracking of training progress from anywhere
    - **Experiment Comparison**: Side-by-side analysis of different hyperparameter settings
    - **Team Collaboration**: Shared visibility into training runs across research teams
    - **Automated Visualization**: Built-in plots for losses, accuracies, and learning rates
    - **Hyperparameter Tracking**: Automatic correlation between config and performance
    
    Metric Processing Pipeline:
    1. **Base Metrics Assembly**: Core training metrics common to all configurations
    2. **Conditional Addition**: Task-specific metrics based on enabled prediction heads
    3. **Data Type Conversion**: PyTorch tensor conversion to WandB-compatible scalars
    4. **Platform Upload**: Structured data transmission to WandB servers
    5. **Dashboard Update**: Automatic refresh of experiment tracking dashboard
    
    Logged Metric Categories:
    - **Training Dynamics**: Epoch progress, total losses, learning rate schedules
    - **Performance Metrics**: Accuracy scores, F1 measures for classification tasks
    - **Timing Information**: Execution times for forward pass and optimizer steps
    - **Task-Specific Losses**: Reconstruction, pose, and orientation objectives
    - **Early Stopping**: Epochs since improvement for convergence monitoring
    
    Platform Features:
    - Automatic plot generation for numerical metrics over time
    - Hyperparameter correlation analysis with final performance
    - Model artifact storage and versioning capabilities
    - Integration with popular ML frameworks and cloud platforms
    
    Parameters:
    - epoch: Current epoch number (0-indexed) for temporal tracking on WandB dashboard.
    - metrics: Dictionary containing all computed metrics from training and validation phases.
    - pose_pred: Whether to include pose classification metrics in the upload.
    - orient_pred: Whether to include orientation classification metrics in the upload.
    - recon: Whether to include reconstruction loss metrics in the upload.
    
    Returns:
    - None: Metrics are uploaded directly to the active WandB experiment run.
    
    Prerequisites:
    - Active WandB session initialized with wandb.init()
    - Valid WandB API key and network connectivity
    - Global wandb_log flag set to True to enable logging
    
    Notes:
    - Metrics are uploaded asynchronously to avoid blocking training
    - Tensor values are automatically converted to Python scalars for compatibility
    - Conditional metric inclusion prevents undefined values in the dashboard
    - Essential for monitoring long-running training jobs and comparing experiments
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
    # Convert any tensor values to regular numbers for logging
    for key, value in wandb_log_data.items():
        if isinstance(value, torch.Tensor):
            wandb_log_data[key] = value.item()
    wandb.log(wandb_log_data)


