"""Compute training-window class weights and reconstruction/pretraining losses."""

from __future__ import annotations

from collections import Counter
from typing import List, Tuple

import torch
import torch.nn.functional as F


__all__ = ['calculate_class_weights', 'masked_mse_loss', 'info_nce_loss', 'barlow_twins_loss']

def calculate_class_weights(
    full_dataset,
    train_indices: List[int],
    val_indices: List[int],
    pose_pred: bool,
    orient_pred: bool,
    max_class_weight: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute clipped inverse-frequency weights from training windows only.

    Estimate frequencies for joint enabled pose/orientation labels using
    ``train_indices``. Both returned tensors use global instruction indices,
    so validation uses the same training-derived weights. Unseen validation
    label combinations receive weight 1.0.
    """

    n_instructions = len(full_dataset.dataset_instructions)

    if n_instructions == 0 or len(train_indices) == 0:
        uniform = torch.ones(n_instructions, dtype=torch.float32)
        return uniform, uniform

    from .window_index import WindowInstructions
    if isinstance(full_dataset.dataset_instructions, WindowInstructions):
        import numpy as np
        instructions = full_dataset.dataset_instructions
        pose_codes = instructions.label_codes('pose') if pose_pred else np.zeros(n_instructions, dtype=np.int32)
        orient_codes = instructions.label_codes('orient') if orient_pred else np.zeros(n_instructions, dtype=np.int32)
        stride = max(1, len(full_dataset.orient_dict)) if orient_pred else 1
        joint = pose_codes.astype(np.int64) * stride + orient_codes
        keys, counts = np.unique(joint[np.asarray(train_indices, dtype=np.int64)], return_counts=True)
        per_class = np.minimum(len(train_indices) / (len(keys) * counts), max_class_weight)
        locations = np.searchsorted(keys, joint)
        safe_locations = np.minimum(locations, len(keys) - 1)
        known = (locations < len(keys)) & (keys[safe_locations] == joint)
        weights = np.ones(n_instructions, dtype=np.float32)
        weights[known] = per_class[safe_locations[known]]
        if len(val_indices):
            unseen = np.unique(joint[np.asarray(val_indices, dtype=np.int64)][~known[np.asarray(val_indices, dtype=np.int64)]])
            if len(unseen):
                print(f'WARNING: {len(unseen)} label tuples in VAL are absent from TRAIN; weight=1.0.')
        print(f'Class weights computed from TRAIN: {len(keys)} label tuples, {len(train_indices)} windows.')
        tensor = torch.from_numpy(weights)
        return tensor, tensor

    # Estimate joint label frequencies from training windows only.
    train_label_tuples = []
    for idx in train_indices:
        instr = full_dataset.dataset_instructions[idx]
        train_label_tuples.append(
            (
                instr['pose_label'] if pose_pred else None,
                instr['orient_label'] if orient_pred else None
            )
        )

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

    print("Class Weights Summary (computed from TRAIN split only):")
    for label, weight in sorted(class_weights_dict.items(), key=lambda kv: kv[1], reverse=True)[:30]:
        print(f"Pose: {label[0]}, Orient: {label[1]}, Class Weight: {weight}")
    if len(class_weights_dict) > 30:
        print(f"... ({len(class_weights_dict) - 30} more classes not shown)")

    if len(val_indices):
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

    # Both vectors share training-derived weights and global window indexing.
    return weights_full, weights_full


def masked_mse_loss(pred: torch.Tensor, target: torch.Tensor, keep_mask: torch.Tensor) -> torch.Tensor:
    """Return mean squared error over masked elements only.

    All tensors must have the same shape. ``keep_mask`` is Boolean with True
    for unmasked elements; targets are the original clean features. Compute in
    float32 under autocast and return scalar zero when nothing is masked.
    """
    if pred.shape != target.shape:
        raise ValueError(f"masked_mse_loss: pred and target must match. Got {pred.shape} vs {target.shape}")
    if keep_mask.shape != pred.shape:
        raise ValueError(
            f"masked_mse_loss: keep_mask must match pred/target. Got {keep_mask.shape} vs {pred.shape}"
        )

    masked = ~keep_mask
    if masked.sum().item() == 0:
        return pred.float()[masked].sum()

    # Do the math in float32 for numerical stability under autocast.
    diff2 = (pred.float() - target.float()) ** 2
    return diff2[masked].mean()


def info_nce_loss(view1: torch.Tensor, view2: torch.Tensor, temperature: float = 0.1) -> torch.Tensor:
    """Compute InfoNCE for two (B, D) batches of paired-view embeddings.

    Normalize embeddings, pair corresponding rows across views, and exclude
    self-similarity from the contrastive denominator. Return a scalar loss.
    """
    view1 = F.normalize(view1, dim=1)
    view2 = F.normalize(view2, dim=1)

    embeddings = torch.cat([view1, view2], dim=0)
    
    similarity_matrix = torch.matmul(embeddings, embeddings.T) / temperature

    # The labels are the indices of the positive pairs.
    # For a batch of size B, the positive for sample i is at i+B, and for i+B is at i.
    batch_size = view1.shape[0]
    labels = torch.arange(batch_size, device=view1.device)
    labels = torch.cat([labels + batch_size, labels])

    # Exclude self-similarity from the contrastive denominator.
    neg_inf = -torch.finfo(similarity_matrix.dtype).max
    similarity_matrix.fill_diagonal_(neg_inf)

    # Cross-entropy uses the paired-view embedding as the positive target.
    loss = F.cross_entropy(similarity_matrix, labels)
    
    return loss


def barlow_twins_loss(z1: torch.Tensor, z2: torch.Tensor, lambda_param: float = 5e-3) -> torch.Tensor:
    """Compute Barlow Twins loss for matching (B, D) projected embeddings.

    Require B >= 2 and finite, nonconstant dimensions. Standardize each
    dimension across the batch, then penalize departures from a unit
    correlation diagonal and weight off-diagonal redundancy by ``lambda_param``.
    """
    batch_size, proj_dim = z1.shape
    if z1.shape != z2.shape or batch_size < 2:
        raise ValueError("Barlow Twins requires matching embeddings and at least two observations")

    # Standardize each projected dimension across the batch.
    sd1, sd2 = z1.std(0, unbiased=False), z2.std(0, unbiased=False)
    if not torch.isfinite(z1).all() or not torch.isfinite(z2).all() or (sd1 <= 0).any() or (sd2 <= 0).any():
        raise ValueError("Barlow Twins embeddings contain nonfinite or constant dimensions")
    z1_norm = (z1 - z1.mean(0)) / sd1
    z2_norm = (z2 - z2.mean(0)) / sd2

    # Compute the cross-correlation matrix between the two sets of embeddings.
    # The resulting matrix will have shape (proj_dim, proj_dim).
    c = torch.matmul(z1_norm.T, z2_norm) / batch_size

    # The loss has two components:
    # - Invariance term: Pushes the diagonal elements of `c` to 1.
    # - Redundancy reduction term: Pushes the off-diagonal elements of `c` to 0.
    
    on_diag = torch.diagonal(c)
    invariance_loss = ((on_diag - 1) ** 2).sum()
    
    off_diag = c.fill_diagonal_(0)
    redundancy_loss = (off_diag ** 2).sum()
    
    loss = invariance_loss + lambda_param * redundancy_loss
    
    return loss


