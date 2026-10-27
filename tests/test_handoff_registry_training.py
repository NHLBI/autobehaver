"""Tiny native tests for external storage and true training resume continuity."""
from pathlib import Path
import shutil

import h5py
import numpy as np
import pandas as pd
import pytest
import torch

from autobehaver.registry import (RegistryValidationError, read_registry, relpath_str,
                                 resolve_registry_path, upsert_registry_rows,
                                 validate_registry_dataframe)
from autobehaver.training_identity import (build_training_data_identity,
                                          validate_training_data_identity)


def test_external_registry_location_survives_existing_row_and_returns_to_project(tmp_path):
    project = tmp_path / 'project'
    project.mkdir()
    registry = project / 'registry.csv'
    inside = project / 'crop.mp4'
    external = tmp_path / 'external' / 'crop.mp4'
    upsert_registry_rows(csv_path=registry, rows=[{'recording_id':'r', 'active':True}])
    upsert_registry_rows(csv_path=registry, rows=[{'recording_id':'r',
        'cropped_movie_relpath':relpath_str(external, project)}])
    row = read_registry(registry).iloc[0]
    assert resolve_registry_path(project, row, 'cropped_movie_relpath') == external.resolve()
    assert row['cropped_movie_storage_root'] != '.'
    upsert_registry_rows(csv_path=registry, rows=[{'recording_id':'r', 'cropped_movie_relpath':'crop.mp4'}])
    row = read_registry(registry).iloc[0]
    assert row['cropped_movie_storage_root'] == '.'
    assert resolve_registry_path(project, row, 'cropped_movie_relpath') == inside.resolve()


def test_registry_cross_drive_does_not_collapse_to_basename(tmp_path, monkeypatch):
    import autobehaver.registry as registry_module
    def different_drive(*args):
        raise ValueError('different drive')
    monkeypatch.setattr(registry_module.os.path, 'relpath', different_drive)
    source = tmp_path / 'other' / 'folder' / 'crop.mp4'
    encoded = relpath_str(source, tmp_path / 'project')
    row = {'recording_id':'r', 'cropped_movie_relpath':str(encoded),
           'cropped_movie_storage_root':encoded.storage_root}
    assert str(encoded) != source.name
    validate_registry_dataframe(pd.DataFrame([row]))
    assert resolve_registry_path(tmp_path / 'project', row, 'cropped_movie_relpath') == source.resolve()
    row['cropped_movie_storage_root'] = str(tmp_path)
    with pytest.raises(RegistryValidationError):
        validate_registry_dataframe(pd.DataFrame([row]))


@pytest.mark.parametrize('path', ['../outside.mp4', '/outside.mp4', 'C:outside.mp4'])
def test_registry_rejects_ambiguous_or_escaping_paths(path):
    with pytest.raises(RegistryValidationError):
        validate_registry_dataframe(pd.DataFrame([{'recording_id':'r','cropped_movie_relpath':path}]))


def _training_files(directory):
    metadata = pd.DataFrame({'filename':['a.h5', 'b.h5'],
                             'labels':['a_labels.h5','b_labels.h5'], 'set':['train','val']})
    for index, row in metadata.iterrows():
        with h5py.File(directory / row.filename, 'w') as handle:
            handle.create_dataset('data', data=np.arange(36, dtype='float32').reshape(9,4) + index)
            handle.attrs['columns'] = np.array(['f0','f1','f2','f3'],dtype='S')
        with h5py.File(directory / row.labels, 'w') as handle:
            handle.create_dataset('Behavior', data=np.array(['rest','walk']*4+['rest'],dtype='S'))
    return metadata


def test_training_content_identity_allows_moved_identical_labels_and_root(tmp_path):
    metadata = _training_files(tmp_path)
    identity = build_training_data_identity(tmp_path, metadata, sequence_length=5,pose_pred=True,orient_pred=False)
    moved = tmp_path / 'copy'
    moved.mkdir()
    for path in tmp_path.glob('*.h5'):
        shutil.copyfile(path, moved / path.name)
    current_metadata = metadata.copy()
    shutil.copyfile(moved/'a_labels.h5', moved/'renamed.h5')
    current_metadata.loc[0,'labels'] = 'renamed.h5'
    current = build_training_data_identity(moved,current_metadata,sequence_length=5,pose_pred=True,orient_pred=False)
    validate_training_data_identity(identity,current)


@pytest.mark.parametrize('changed_file,dataset', [('a.h5','data'),('a_labels.h5','Behavior')])
def test_native_training_resume_rejects_changed_data_and_restart_retains_checkpoint_split(tmp_path,changed_file,dataset):
    from autobehaver._training_core import prepare_data
    metadata = _training_files(tmp_path)
    initial = prepare_data(str(tmp_path),metadata,5,num_workers=0,pose_pred=True,orient_pred=False,defined_split=True)
    checkpoint = tmp_path/'resume.pth'
    torch.save({'checkpoint_kind':'supervised','pose_dict':initial[2], **initial[-1]},checkpoint)
    unchanged = prepare_data(str(tmp_path),metadata,5,num_workers=0,pose_pred=True,orient_pred=False,
                             defined_split=True,pretrained_model_path=str(checkpoint))
    assert unchanged[-1]['training_data_identity'] == initial[-1]['training_data_identity']
    with h5py.File(tmp_path/changed_file,'r+') as handle:
        if dataset == 'data':
            handle[dataset][0,0] += 1
        else:
            handle[dataset][0] = b'walk'
    with pytest.raises(ValueError,match='Training inputs changed'):
        prepare_data(str(tmp_path),metadata,5,num_workers=0,pose_pred=True,orient_pred=False,
                     defined_split=True,pretrained_model_path=str(checkpoint))
    metadata['set'] = ['val','train']
    restarted = prepare_data(str(tmp_path),metadata,5,num_workers=0,pose_pred=True,orient_pred=False,
        defined_split=True,pretrained_model_path=str(checkpoint),restart_optimizer=True)
    assert restarted[-1]['train_filenames'] == ['a.h5']
    assert restarted[-1]['val_filenames'] == ['b.h5']
    for prepared in (initial, unchanged, restarted):
        store = prepared[0].dataset.data_in_memory
        if hasattr(store, 'close'):
            store.close()


def test_historical_training_checkpoint_stays_readable(caplog):
    with caplog.at_level('WARNING'):
        validate_training_data_identity(None, {'files':[]})
    assert 'Historical checkpoint' in caplog.text
