"""Frozen rc32 PCG64 inference reference. Do not update with runtime refactors."""
import torch
TTA_RANDOM_ALGORITHM = "pcg64-original-frame-block-v1"
_TTA_BLOCK_SIZE = 64

def augment_inference_features(sequences, columns_to_noise, noise_level, *, seed,
                               recording_identities, source_frames, model_identity, pass_index):
    """Add seeded Gaussian noise, independent of file order and batch boundaries.

    Each recording/model/pass uses fixed blocks of 64 original center frames.
    Generate only blocks touched by this batch, so memory is bounded and neither
    a per-window generator nor a whole-recording noise array is required.
    """
    import hashlib
    import json
    import numpy as np

    if not columns_to_noise or noise_level == 0:
        return sequences
    if len(recording_identities) != len(sequences) or len(source_frames) != len(sequences):
        raise ValueError('TTA requires one recording identity and original center frame per window.')
    groups = {}
    for row, (recording, frame) in enumerate(zip(recording_identities, source_frames)):
        frame = int(frame)
        if frame < 0:
            raise ValueError('TTA original center frames must be nonnegative.')
        block, offset = divmod(frame, _TTA_BLOCK_SIZE)
        groups.setdefault((recording, block), []).append((row, offset))
    shape = (len(sequences), int(sequences.shape[1]), len(columns_to_noise))
    noise = np.empty(shape, dtype=np.float32)
    for (recording, block), rows in groups.items():
        key = json.dumps([TTA_RANDOM_ALGORITHM, int(seed), recording, model_identity,
                          int(pass_index), block, shape[1], list(columns_to_noise)],
                         separators=(',', ':')).encode('utf-8')
        block_seed = int.from_bytes(hashlib.sha256(key).digest(), 'little')
        generator = np.random.Generator(np.random.PCG64(block_seed))
        block_noise = generator.standard_normal((_TTA_BLOCK_SIZE, *shape[1:]), dtype=np.float32)
        destinations, offsets = zip(*rows)
        noise[list(destinations)] = block_noise[list(offsets)]
    augmented = sequences.clone()
    added = torch.from_numpy(noise).to(device=sequences.device, dtype=sequences.dtype)
    augmented[:, :, columns_to_noise] += added * float(noise_level)
    return augmented
