"""Loss functions used by Autobehaver.

Includes supervised losses (pose/orientation classification), optional reconstruction loss, and
self-supervised losses used during pretraining (InfoNCE and Barlow Twins)."""

from __future__ import annotations

from collections import Counter
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = ['calculate_class_weights', 'mse_loss', 'masked_mse_loss', 'info_nce_loss', 'barlow_twins_loss', 'pose_only_criterion']

def calculate_class_weights(
    full_dataset,
    train_indices: List[int],
    val_indices: List[int],
    pose_pred: bool,
    orient_pred: bool,
    max_class_weight: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Calculate per-instruction sample weights to handle class imbalance.

    IMPORTANT: The class-frequency statistics are computed **only from the training split**
    (i.e., from `train_indices`). This avoids leaking validation-set information into the
    loss weighting.

    The returned tensors are indexed by the *global* instruction index (the index into
    `full_dataset.dataset_instructions`) which is what `KeypointsDataset.__getitem__`
    returns as `idx`. This matches the existing training/validation loops that do:

        batch_weights = class_weights[indices]

    If a (pose, orient) label tuple appears in validation but not in training, it will
    default to weight=1.0 (the model cannot learn a class it never sees in training).
    """

    n_instructions = len(full_dataset.dataset_instructions)

    # Fall back to uniform weights if there is nothing to compute.
    if n_instructions == 0 or len(train_indices) == 0:
        uniform = torch.ones(n_instructions, dtype=torch.float32)
        return uniform, uniform

    # Build (pose, orient) label tuples for TRAIN split only
    train_label_tuples = []
    for idx in train_indices:
        instr = full_dataset.dataset_instructions[idx]
        train_label_tuples.append(
            (
                instr['pose_label'] if pose_pred else None,
                instr['orient_label'] if orient_pred else None
            )
        )

    # Count frequency of each label tuple in TRAIN split
    label_counts = Counter(train_label_tuples)
    total_samples = sum(label_counts.values())
    num_classes = max(1, len(label_counts))

    # Inverse-frequency weights with clipping
    class_weights_dict = {
        label: min(total_samples / (num_classes * count), max_class_weight)
        for label, count in label_counts.items()
    }

    # Build a FULL-LENGTH weight vector (indexed by global instruction index)
    weights_full = torch.ones(n_instructions, dtype=torch.float32)
    for i, instr in enumerate(full_dataset.dataset_instructions):
        label = (
            instr['pose_label'] if pose_pred else None,
            instr['orient_label'] if orient_pred else None
        )
        weights_full[i] = float(class_weights_dict.get(label, 1.0))

    # Print a brief summary (top 30 weights) for diagnostics
    print("Class Weights Summary (computed from TRAIN split only):")
    for label, weight in sorted(class_weights_dict.items(), key=lambda kv: kv[1], reverse=True)[:30]:
        print(f"Pose: {label[0]}, Orient: {label[1]}, Class Weight: {weight}")
    if len(class_weights_dict) > 30:
        print(f"... ({len(class_weights_dict) - 30} more classes not shown)")

    # Optional: warn if VAL contains unseen labels
    if val_indices:
        val_label_tuples = set()
        for idx in val_indices:
            instr = full_dataset.dataset_instructions[idx]
            val_label_tuples.add(
                (
                    instr['pose_label'] if pose_pred else None,
                    instr['orient_label'] if orient_pred else None
                )
            )
        unseen_in_train = [lbl for lbl in val_label_tuples if lbl not in class_weights_dict]
        if unseen_in_train:
            print(
                f"WARNING: {len(unseen_in_train)} (pose, orient) label tuples appear in VAL but not in TRAIN. "
                "Those samples will use weight=1.0 and the model cannot learn them from this split."
            )

    # Keep existing API: return train and val tensors (same indexing and values).
    return weights_full, weights_full


def mse_loss(input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Computes the Mean Squared Error (MSE) loss between input and target tensors.
    
    This function calculates the element-wise squared difference between the input
    and target tensors, then returns the mean of all squared differences. Commonly
    used for regression tasks and reconstruction loss in autoencoders.
    
    Parameters:
    - input: The predicted tensor values.
    - target: The ground truth tensor values.
    
    Returns:
    - torch.Tensor: The mean squared error loss as a scalar tensor.
    """
    squared_diff = (input - target) ** 2
    loss = squared_diff.mean()
    return loss


def masked_mse_loss(pred: torch.Tensor, target: torch.Tensor, keep_mask: torch.Tensor) -> torch.Tensor:
    """Mean-squared-error computed **only on masked elements**.

    This is designed for denoising / masked-reconstruction objectives:
    - The model input is corrupted (noise and/or masking).
    - The target is the *original clean* sequence.
    - We only compute reconstruction loss on elements that were actually masked
      (i.e., keep_mask == False).

    Parameters:
        pred: Model reconstruction output, shape (...).
        target: Clean target tensor, same shape as pred.
        keep_mask: Boolean mask, same shape as pred, where True=kept/unmasked and False=masked.

    Returns:
        Scalar tensor loss (0.0 if nothing was masked in this batch).
    """
    if pred.shape != target.shape:
        raise ValueError(f"masked_mse_loss: pred and target must match. Got {pred.shape} vs {target.shape}")
    if keep_mask.shape != pred.shape:
        raise ValueError(
            f"masked_mse_loss: keep_mask must match pred/target. Got {keep_mask.shape} vs {pred.shape}"
        )

    masked = ~keep_mask
    if masked.sum().item() == 0:
        # Nothing masked: return a scalar 0 on the right device/dtype.
        return torch.zeros((), device=pred.device, dtype=torch.float32)

    # Do the math in float32 for numerical stability under autocast.
    diff2 = (pred.float() - target.float()) ** 2
    return diff2[masked].mean()


def info_nce_loss(view1: torch.Tensor, view2: torch.Tensor, temperature: float = 0.1) -> torch.Tensor:
    """
    Calculates the InfoNCE loss for contrastive learning.
    
    Parameters:
    - view1: Tensor of embeddings for the first view (batch_size x proj_dim)
    - view2: Tensor of embeddings for the second view (batch_size x proj_dim)
    - temperature: The temperature scaling factor for the similarity scores.

    Returns:
    - The InfoNCE loss.
    """
    # Normalize the embeddings to have unit length
    view1 = F.normalize(view1, dim=1)
    view2 = F.normalize(view2, dim=1)

    # Concatenate the two views to create a batch of size 2*B
    embeddings = torch.cat([view1, view2], dim=0)
    
    # Calculate the cosine similarity matrix (2*B x 2*B)
    similarity_matrix = torch.matmul(embeddings, embeddings.T) / temperature

    # The labels are the indices of the positive pairs.
    # For a batch of size B, the positive for sample i is at i+B, and for i+B is at i.
    batch_size = view1.shape[0]
    labels = torch.arange(batch_size, device=view1.device)
    labels = torch.cat([labels + batch_size, labels])

    # Mask out self-similarity on the diagonal.
    # If we keep the diagonal, each sample's similarity to itself is always maximal (after L2-normalization it's 1/temperature)
    # and it becomes an unavoidable 'negative' in the softmax, which can make the objective ill-posed.
    neg_inf = -torch.finfo(similarity_matrix.dtype).max
    similarity_matrix.fill_diagonal_(neg_inf)

    # We use CrossEntropyLoss, which combines LogSoftmax and NLLLoss.
    # This is equivalent to the InfoNCE loss formula.
    loss = F.cross_entropy(similarity_matrix, labels)
    
    return loss


def barlow_twins_loss(z1: torch.Tensor, z2: torch.Tensor, lambda_param: float = 5e-3) -> torch.Tensor:
    """
    Calculates the Barlow Twins loss.
    This loss function encourages the embeddings of augmented views of the same sample
    to be similar, while minimizing the redundancy between the components of the embeddings.
    
    Parameters:
    - z1: Tensor of embeddings for the first view (batch_size x proj_dim)
    - z2: Tensor of embeddings for the second view (batch_size x proj_dim)
    - lambda_param: The weight for the redundancy reduction term.

    Returns:
    - The Barlow Twins loss.
    """
    batch_size, proj_dim = z1.shape

    # 1. Normalization
    # Normalize each embedding vector to have zero mean and unit standard deviation.
    z1_norm = (z1 - z1.mean(0)) / z1.std(0)
    z2_norm = (z2 - z2.mean(0)) / z2.std(0)

    # 2. Cross-correlation matrix
    # Compute the cross-correlation matrix between the two sets of embeddings.
    # The resulting matrix will have shape (proj_dim, proj_dim).
    c = torch.matmul(z1_norm.T, z2_norm) / batch_size

    # 3. Loss calculation
    # The loss has two components:
    # - Invariance term: Pushes the diagonal elements of `c` to 1.
    # - Redundancy reduction term: Pushes the off-diagonal elements of `c` to 0.
    
    # Invariance term
    on_diag = torch.diagonal(c)
    invariance_loss = ((on_diag - 1) ** 2).sum()
    
    # Redundancy reduction term
    off_diag = c.fill_diagonal_(0)
    redundancy_loss = (off_diag ** 2).sum()
    
    # Combine the two terms with the lambda parameter
    loss = invariance_loss + lambda_param * redundancy_loss
    
    return loss


def pose_only_criterion(outputs: tuple[torch.Tensor, ...], targets: torch.Tensor) -> torch.Tensor:
    """
    Criterion function for learning rate finder that extracts pose predictions and computes cross-entropy loss.
    
    This function is used specifically with torch-lr-finder to evaluate model performance during
    learning rate range testing. It extracts the pose predictions from the model outputs and
    computes the cross-entropy loss against the ground truth targets.
    
    Parameters:
    - outputs: Tuple containing model outputs, where the first element is pose logits.
    - targets: Ground truth pose labels as a tensor.
    
    Returns:
    - torch.Tensor: Cross-entropy loss between predicted and actual pose labels.
    """
    pose_logits = outputs[0]
    targets = targets.view(-1)  # Ensure targets is 1D for CrossEntropyLoss
    return nn.CrossEntropyLoss()(pose_logits, targets)


