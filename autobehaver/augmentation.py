"""Apply Gaussian noise and elementwise masking to sequence features."""

from __future__ import annotations

from typing import Tuple, Union

import torch


__all__ = ['augment_features', 'select_columns_for_augmentation', 'augment_inference_features', 'TTA_RANDOM_ALGORITHM']

TTA_RANDOM_ALGORITHM = 'pcg64-original-frame-block-v1'
_TTA_BLOCK_SIZE = 64


def augment_inference_features(sequences, columns_to_noise, noise_level, *, seed,
                               recording_identities, source_frames, model_identity, pass_index):
    """Add seeded Gaussian noise, independent of file order and batch boundaries.

    Each recording/model/pass uses fixed blocks of 64 original center frames.
    Generate only blocks touched by this batch, so memory is bounded and neither
    a per-window generator nor a whole-recording noise array is required.
    """
    if not columns_to_noise or noise_level == 0:
        return sequences
    noise = inference_noise(len(sequences), int(sequences.shape[1]), columns_to_noise,
                            seed=seed, recording_identities=recording_identities,
                            source_frames=source_frames, model_identity=model_identity,
                            pass_index=pass_index)
    return apply_inference_noise(sequences, noise, columns_to_noise, noise_level)


def inference_noise(rows, sequence_length, columns_to_noise, *, seed,
                    recording_identities, source_frames, model_identity, pass_index,
                    pin_memory=False):
    """Generate the original PCG64 values in a CPU buffer, optionally pinned.

    Independent generators release the GIL in NumPy's native normal generator,
    allowing bounded preparation threads without copying arrays between processes.
    The key, block shape, dtype and indexing match pcg64-original-frame-block-v1.
    """
    import hashlib
    import json
    import numpy as np

    if len(recording_identities) != rows or len(source_frames) != rows:
        raise ValueError('TTA requires one recording identity and original center frame per window.')
    groups = {}
    for row, (recording, frame) in enumerate(zip(recording_identities, source_frames)):
        frame = int(frame)
        if frame < 0:
            raise ValueError('TTA original center frames must be nonnegative.')
        block, offset = divmod(frame, _TTA_BLOCK_SIZE)
        groups.setdefault((recording, block), []).append((row, offset))
    shape = (rows, sequence_length, len(columns_to_noise))
    tensor = torch.empty(shape, dtype=torch.float32, pin_memory=pin_memory)
    noise = tensor.numpy()
    for (recording, block), rows in groups.items():
        key = json.dumps([TTA_RANDOM_ALGORITHM, int(seed), recording, model_identity,
                          int(pass_index), block, shape[1], list(columns_to_noise)],
                         separators=(',', ':')).encode('utf-8')
        block_seed = int.from_bytes(hashlib.sha256(key).digest(), 'little')
        generator = np.random.Generator(np.random.PCG64(block_seed))
        block_noise = generator.standard_normal((_TTA_BLOCK_SIZE, *shape[1:]), dtype=np.float32)
        destinations, offsets = zip(*rows)
        noise[list(destinations)] = block_noise[list(offsets)]
    return tensor


def apply_inference_noise(sequences, noise, columns_to_noise, noise_level):
    """Apply prepared noise with the original arithmetic and column semantics."""
    augmented = sequences.clone()
    added = noise.to(device=sequences.device, dtype=sequences.dtype, non_blocking=noise.is_pinned())
    augmented[:, :, columns_to_noise] += added * float(noise_level)
    return augmented


def augment_features(
    features: torch.Tensor,
    mask_p: float,
    noise_level: float,
    columns_to_noise: list[int],
    *,
    return_mask: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """Augment a feature tensor of shape (B, T, F) or (T, F).

    Add Gaussian noise to ``columns_to_noise``, then mask individual elements
    to zero so noise cannot restore masked values. ``noise_level`` is the noise
    standard deviation in feature units. Return the augmented tensor, or
    (tensor, keep_mask) when ``return_mask`` is true; True means unmasked.
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
    """Return all feature-column indices for Gaussian noise.

    Elementwise masking also applies across all columns.
    """

    return list(range(len(headers)))
