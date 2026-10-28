import numpy as np
import pandas as pd
import pytest

from autobehaver.postinf.mmgen.cli import build_parser
from autobehaver.postinf.mmgen.core import MMGenConfig
from autobehaver.postinf.mmgen.worker_io import _align_poscalc_to_inference


def test_explicit_normalization_offset_preserves_window_centers():
    raw = pd.DataFrame({'source_row': np.arange(80)})
    predictions = pd.DataFrame({'Frame Number': np.arange(1, 71)})
    _, aligned = _align_poscalc_to_inference(predictions, raw, edge_skip=5,
                                            source_name='test', source_frame_offset=5)
    assert aligned.source_row.tolist() == list(range(5, 75))
    assert aligned.source_row.iloc[20] == 25
    assert aligned.source_row.iloc[49] == 54


def test_sparse_frame_timeline_is_rejected_before_bout_computation():
    raw = pd.DataFrame({'source_row': np.arange(80)})
    df = pd.DataFrame({'Frame Number': [6, 26, 75]})
    # Sparse frames cannot be treated as adjacent in durations/transitions.
    # Reject the timeline instead of silently closing its missing intervals.
    with pytest.raises(ValueError, match='consecutive'):
        _align_poscalc_to_inference(df, raw, edge_skip=5, source_name='test')


@pytest.mark.parametrize('frames', [[1, 76], [0, 1], [2, 2], [3, 2], [1, 1.5]])
def test_explicit_offset_never_falls_back_on_invalid_coordinates(frames):
    with pytest.raises(ValueError, match='Invalid Frame Number'):
        _align_poscalc_to_inference(pd.DataFrame({'Frame Number': frames}),
                                   pd.DataFrame({'x': np.arange(80)}), edge_skip=5,
                                   source_name='test', source_frame_offset=5,
                                   allow_unsafe_fallback=True)


def test_offset_requires_frame_numbers():
    with pytest.raises(ValueError, match='requires'):
        _align_poscalc_to_inference(pd.DataFrame({'label': [1]}),
                                   pd.DataFrame({'x': range(8)}), edge_skip=5,
                                   source_name='test', source_frame_offset=5)


def test_cli_config_runtime_chain():
    args = build_parser().parse_args(['--source_frame_offset', '5'])
    cfg = MMGenConfig(inference_csv_dir='i', poscalc_dir='p', output_dir='o',
                      source_frame_offset=args.source_frame_offset)
    assert cfg.runtime_settings.source_frame_offset == 5
    assert build_parser().parse_args([]).source_frame_offset == 0
    with pytest.raises(ValueError, match='nonnegative'):
        MMGenConfig(inference_csv_dir='i', poscalc_dir='p', output_dir='o', source_frame_offset=-1)
