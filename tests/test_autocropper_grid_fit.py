import numpy as np
import pytest
from autobehaver.preinf.autocropper.detection import calculate_theoretical_centers
from autobehaver.preinf.autocropper.grid_fit import fit_grid, circle_support

TEMPLATE = calculate_theoretical_centers(2016, 1520, 3, 2)
REAL = np.array([[320,1024,274], [962,380,272], [1600,1018,272],
                 [1600,380,272], [958,1028,278], [318,382,274]])


def test_extra_circles_do_not_displace_real_grid():
    candidates = np.vstack([REAL, [758,1190,286], [1394,1202,270]])
    coords, report = fit_grid(candidates, TEMPLATE, 1, (1520,2016), [.99,1,.94,1,.96,1,.57,.38])
    assert report['selected_candidates'] == dict(chamber1=5,chamber2=1,chamber3=3,chamber4=0,chamber5=4,chamber6=2)
    assert coords['chamber1'] == (318,382) and report['inferred_chambers'] == []
    assert report['max_residual_px'] < 5


@pytest.mark.parametrize('subset', [[0,1,2,3,4], [0,1,2,3], [0,1,2]])
def test_missing_circles_are_inferred_from_noncollinear_evidence(subset):
    coords, report = fit_grid(REAL[subset], TEMPLATE, 1, (1520,2016))
    complete, _ = fit_grid(REAL, TEMPLATE, 1, (1520,2016))
    assert len(report['inferred_chambers']) == 6-len(subset)
    assert max(np.linalg.norm(np.array(coords[k])-complete[k]) for k in coords) < 10


def test_grid_follows_translation_and_rotation():
    ref = np.array(list(TEMPLATE.values()), float)
    angle = np.deg2rad(5)
    matrix = np.array([[np.cos(angle),np.sin(angle)],[-np.sin(angle),np.cos(angle)]])
    moved = (ref - ref.mean(axis=0)) @ matrix + [1000,760]
    circles = np.c_[moved, np.full(6,265)]
    coords, report = fit_grid(circles, TEMPLATE, 1, (1520,2016))
    assert np.max(np.abs(np.array(list(coords.values()))-moved)) < 1
    assert not report['inferred_chambers']


def test_reject_collinear_underconstrained_and_out_of_frame():
    for candidates in (REAL[[1,3,5]], REAL[:2], REAL + [1500,0,0]):
        with pytest.raises(ValueError):
            fit_grid(candidates, TEMPLATE, 1, (1520,2016))


def test_reject_equally_supported_alternative_grid():
    with pytest.raises(ValueError, match='Ambiguous'):
        fit_grid(np.vstack([REAL,REAL+[25,0,0]]), TEMPLATE, 1, (1520,2016))


def test_no_boundary_support_is_not_confidence():
    scores, edges = circle_support(np.zeros((100,100),np.uint8), np.array([[50,50,20]]))
    assert scores.tolist() == [0] and not edges.any()


def test_recovery_reuse_requires_pinned_producer_inputs_settings_and_bytes(tmp_path, monkeypatch):
    from autobehaver.preinf.autocropper import contract
    from autobehaver.completion import write_receipt
    output=tmp_path/'crop.mp4';output.write_bytes(b'original pixels')
    monkeypatch.setattr(contract,'video_shape',lambda path:(10,600,600))
    old=dict(stage='crop-v1',inputs=['inputhash'],settings={'chamber':'chamber1'},code='a'*64)
    new=dict(old,code='b'*64)
    write_receipt(output,old)
    assert not contract.valid_crop('movie',output,new)

    monkeypatch.setenv('AUTOBEHAVER_CROP_REUSE_CODES','a'*64)
    assert contract.valid_crop('movie',output,new)
    assert not contract.valid_crop('movie',output,dict(new,inputs=['changed']))
    assert not contract.valid_crop('movie',output,dict(new,settings={'chamber':'chamber2'}))
    output.write_bytes(b'changed pixels')
    assert not contract.valid_crop('movie',output,new)


def test_producer_source_is_verified_before_runtime_hashing(tmp_path):
    import hashlib
    from autobehaver.preinf.autocropper.recovery_reuse import producer_codes
    from autobehaver.completion import code_digest
    files = {}
    for name in ('detection.py','cropping.py','contract.py'):
        path = tmp_path/name; path.write_text('def f():\n    return 1\n')
        files[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest = {'sources': [{'root': str(tmp_path), 'files': files}]}
    assert producer_codes(manifest) == [code_digest([tmp_path/name for name in files])]
    (tmp_path/'detection.py').write_text('def f():\n    return 2\n')
    with pytest.raises(ValueError, match='source changed'):
        producer_codes(manifest)
