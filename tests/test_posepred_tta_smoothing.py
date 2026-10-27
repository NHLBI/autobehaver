"""Deterministic inference noise and frozen-output bout-smoothing checks."""
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest
import torch

from autobehaver.augmentation import augment_inference_features, TTA_RANDOM_ALGORITHM
from autobehaver.inference import run_inference, smooth_bouts_vectorized
from tests.test_posepred_inference_audit import _make_supervised_checkpoint, _write_h5
from tests._frozen_smoothing_v1 import smooth_bouts_vectorized as frozen_smoothing


def test_seeded_tta_is_invariant_to_batch_order_and_only_noises_selected_columns():
    data = torch.arange(129 * 5 * 4, dtype=torch.float32).reshape(129, 5, 4) / 100
    frames = np.arange(50, 179)
    identities = ['recording-a'] * 70 + ['recording-b'] * 59
    kwargs = dict(seed=42, model_identity='model-a', pass_index=2)
    whole = augment_inference_features(data, [1, 3], .1,
        recording_identities=identities, source_frames=frames, **kwargs)
    order = np.random.default_rng(8).permutation(len(data))
    reordered = torch.empty_like(data)
    for batch in np.array_split(order, 17):
        reordered[batch] = augment_inference_features(data[batch], [1, 3], .1,
            recording_identities=[identities[i] for i in batch], source_frames=frames[batch], **kwargs)
    assert torch.equal(whole, reordered)
    assert torch.equal(whole[:, :, [0, 2]], data[:, :, [0, 2]])
    assert not torch.equal(whole, data)
    for override in ({'seed': 43}, {'model_identity': 'model-b'}, {'pass_index': 3}):
        changed = augment_inference_features(data, [1, 3], .1, recording_identities=identities,
            source_frames=frames, **{**kwargs, **override})
        assert not torch.equal(changed, whole)


def test_native_seeded_inference_preserves_values_across_file_order_and_batch_size(tmp_path):
    torch.set_num_threads(1)
    torch.manual_seed(17)
    inputs = tmp_path / 'i'
    inputs.mkdir()
    files = [inputs / f'{name}.h5' for name in ('a', 'b')]
    for path in files:
        _write_h5(path, ['x', 'y'], rows=13)
    checkpoint = tmp_path / 'm.pth'
    _make_supervised_checkpoint(checkpoint, columns=['x', 'y'], pose_dict={'rest': 0, 'walk': 1})
    outputs = []
    for name, batch, seed, order in [('one', 3, 42, files), ('two', 5, 42, files[::-1]), ('three', 3, 43, files)]:
        output = tmp_path / name
        outputs.append(output)
        run_inference(str(inputs), [str(checkpoint)], 0, batch, 2, 5, 1, 1,
            use_tta=True, tta_noise_level=.1, tta_seed=seed, output_dir=str(output),
            explicit_files=[str(path) for path in order], use_inferences_subdir=False)
    for name in ('a', 'b'):
        a, b, c = [pd.read_csv(output / f'{name}_inference.csv') for output in outputs]
        pd.testing.assert_frame_equal(a, b, rtol=2e-6, atol=2e-7)
        assert not np.array_equal(a.filter(like='-Prob').to_numpy(), c.filter(like='-Prob').to_numpy(), equal_nan=True)
        meta = json.loads((outputs[0] / f'{name}_inference.csv.provenance.json').read_text())
        assert meta['settings']['tta_seed'] == 42
        assert meta['settings']['tta_random_algorithm'] == TTA_RANDOM_ALGORITHM
        with h5py.File(outputs[0] / 'inference_details/cv/m01' / f'{name}_cv.h5') as handle:
            assert handle.attrs['tta_seed'] == 42
            assert handle.attrs['tta_random_algorithm'] == TTA_RANDOM_ALGORITHM


@pytest.mark.parametrize('threshold', [0, 1, 2, 5, 13])
def test_smoothing_after_gap_bridging_matches_frozen_valid_label_rules(threshold):
    generator = np.random.default_rng(903)
    examples = [[], [None] * 4, [''] * 5,
        ['a', 'a', 'b', 'c', 'Jump', '', 'b', 'a', 'a'],
        ['a', 'b', 'c'] * 70, ['a'] * 1000 + ['b', 'c'] * 20]
    choices = np.array(['a', 'b', 'c', 'Jump', '', None], dtype=object)
    examples += [generator.choice(choices, size=int(generator.integers(1, 240))) for _ in range(20)]
    for labels in examples:
        n = len(labels)
        probs = pd.DataFrame(generator.random((n, 3)), columns=['Pose-a-Prob', 'Pose-b-Prob', 'Pose-c-Prob'])
        if n:
            probs.iloc[::7] = 0  # Equal-probability tie handling.
            probs.iloc[::11] = np.nan
            probs.index = np.arange(n) * 3 + 100
        from autobehaver.label_gaps import fill_short_label_gaps
        bridged, _ = fill_short_label_gaps(labels)
        expected = frozen_smoothing(bridged, probs, threshold, 'Pose', exempt_labels={'Jump'})
        expected, _ = fill_short_label_gaps(expected)
        actual = smooth_bouts_vectorized(labels, probs, threshold, 'Pose', exempt_labels={'Jump'})
        pd.testing.assert_series_equal(pd.Series(actual), pd.Series(expected))
