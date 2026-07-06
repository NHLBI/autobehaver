"""Augmentation utilities for sequence features.

Implements runtime augmentations used during training and optional test-time augmentation during
inference, including additive Gaussian noise and elementwise masking."""

from __future__ import annotations

from typing import Tuple, Union

import torch


__all__ = ['augment_features', 'select_columns_for_augmentation']


def augment_features(
    features: torch.Tensor,
    mask_p: float,
    noise_level: float,
    columns_to_noise: list[int],
    *,
    return_mask: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """Augment a (B, T, F) feature tensor.

    Uses additive Gaussian noise followed by masking. Applying masking last
    prevents noise from being added on top of masked values.

    Parameters:
        features: Tensor of shape (B, T, F) or (T, F).
        mask_p: Probability of masking each element (set to 0.0 to disable masking).
        noise_level: Std-dev of Gaussian noise (set to 0.0 to disable noise).
        columns_to_noise: Feature columns to apply Gaussian noise to.
        return_mask: If True, also returns a boolean mask where True=kept, False=masked.

    Returns:
        If return_mask is False: augmented features tensor.
        If return_mask is True: (augmented_features, keep_mask_bool).
    """

    was_2d = features.dim() == 2
    if was_2d:
        features = features.unsqueeze(0)

    if features.dim() != 3:
        raise ValueError(f"Expected 2D or 3D tensor, but got {features.dim()} dimensions")

    will_noise = (noise_level is not None and noise_level > 0.0 and columns_to_noise is not None and len(columns_to_noise) > 0)
    will_mask = (mask_p is not None and mask_p > 0.0)

    if will_noise or will_mask:
        features = features.clone()

    if will_noise:
        b, t, _ = features.shape
        noise = torch.randn(b, t, len(columns_to_noise), device=features.device) * noise_level
        features[..., columns_to_noise] += noise

    if will_mask:
        keep_mask = torch.rand(features.shape, device=features.device) > mask_p
        features = features * keep_mask
    else:
        keep_mask = torch.ones(features.shape, device=features.device, dtype=torch.bool)

    if was_2d:
        features = features.squeeze(0)
        keep_mask = keep_mask.squeeze(0)

    if return_mask:
        return features, keep_mask
    return features


def select_columns_for_augmentation(headers: list[str]) -> list[int]:
    """Select feature columns for augmentation.

    Current policy:
      - Add Gaussian noise to all feature columns.
      - Apply masking across all feature columns.

    Parameters:
        headers: List of feature names (already decoded to Python ``str``).

    Returns:
        list[int]: Indices of columns that should have Gaussian noise added.
    """

    return list(range(len(headers)))
