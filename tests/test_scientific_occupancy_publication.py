"""Missing spatial files and failed exports cannot certify completed results."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from autobehaver.atomic_io import sha256_file
from autobehaver.postinf.mmgen import occupancy as o


def accumulator():
    acc = o._OccupancyAccumulator(heatmap_bins=50,
        heatmap_range=((0,600),(0,600)), quiver_grid_size=20,
        kde_max_points=500, seed=0)
    col = acc.behavior_cols[0]
    acc.add_kde(col, 'Walking', 'G', np.linspace(.1,.9,100))
    return acc


def test_radial_save_failure_propagates_and_closes_figure(tmp_path, monkeypatch):
    def fail(*args, **kwargs):
        raise OSError('simulated full disk')
    monkeypatch.setattr(o.plt, 'savefig', fail)
    with pytest.raises(OSError, match='full disk'):
        o._generate_radial_distribution_plots_from_acc(accumulator(), str(tmp_path))
    assert not o.plt.get_fignums()


def test_missing_transformed_chunk_is_not_silently_omitted(tmp_path):
    with pytest.raises(RuntimeError, match='Missing transformed'):
        o._write_csv_parts_to_target([str(tmp_path/'missing.csv')], str(tmp_path/'out.csv'))


def test_full_spatial_receipt_hashes_only_current_applicable_outputs(tmp_path, monkeypatch):
    target = tmp_path/'temp_transformed_positional_data.csv'
    plot = tmp_path/'9_Occupancy_Plots'/'current.png'
    plot.parent.mkdir()
    stale = plot.parent/'stale.png'; stale.write_bytes(b'old unrelated export')
    def stream(*args, **kwargs):
        pd.DataFrame({'r_norm':[.4,.8]}).to_csv(target,index=False)
        return accumulator()
    def radial(*args, **kwargs):
        plot.write_bytes(b'current native output')
        return [str(plot)]
    monkeypatch.setattr(o, '_stream_transform_write_and_accumulate', stream)
    monkeypatch.setattr(o, '_generate_radial_distribution_plots_from_acc', radial)
    monkeypatch.setattr(o, '_generate_2d_heatmaps_from_acc', lambda *a, **k: [])
    result = o.calculate_and_save_occupancy_analysis('unused','unused',str(tmp_path),
        save_outputs=True,generate_quiver_plots=False)
    receipt = json.loads((tmp_path/'MMGEN_OCCUPANCY_OUTPUTS.json').read_text())
    assert Path(result) == target
    assert receipt['plot_counts'] == {'radial':1,'heatmaps':0,'quiver':0}
    assert receipt['artifacts'] == {
        'temp_transformed_positional_data.csv':sha256_file(target),
        '9_Occupancy_Plots/current.png':sha256_file(plot)}
    assert not receipt['quiver_requested']
