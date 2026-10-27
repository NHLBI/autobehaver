import pandas as pd
import pytest
from unittest.mock import patch
from autobehaver.postinf.mmgen.integrity import verification_chunks, all_table_inventories, frame_fingerprint
from autobehaver.postinf.mmgen.hdf_store import open_store, hdf_safe_append, HDF_KEYS


def test_chunk_boundaries_preserve_fingerprints_and_detect_corruption(tmp_path):
    size=140123
    frame=pd.DataFrame({'source_file':['a.csv' if i%3 else 'b.csv' for i in range(size)],
                        'value':[float(i%100) for i in range(size)],
                        'category':['long-string' if i%5 else None for i in range(size)]})
    key=HDF_KEYS['postural']
    expected={name:{key:frame_fingerprint(part)} for name,part in frame.groupby('source_file',sort=False)}
    path=tmp_path/'rows.h5'
    with open_store(str(path),'w') as store:
        hdf_safe_append(store,key,frame)
        result=all_table_inventories(store,fingerprints=expected)
        assert result['a.csv'][key]==(frame.source_file=='a.csv').sum()
        sizes=[len(c) for c in verification_chunks(store,key)]
        assert max(sizes)>4096 and max(sizes)<=65536
        assert sum(sizes)==size
        expected['a.csv'][key]['sha256']='0'*64
        with pytest.raises(RuntimeError,match='scientific cells changed'):
            all_table_inventories(store,fingerprints=expected)


def test_small_memory_budget_reduces_blocks(tmp_path):
    frame=pd.DataFrame({'source_file':['a.csv']*8000,'text':['x'*1024]*8000})
    path=tmp_path/'wide.h5';key=HDF_KEYS['postural']
    with open_store(str(path),'w') as store:
        hdf_safe_append(store,key,frame)
        with patch('autobehaver.preinf.execution.memory_limit',return_value=(('test',16*1024**2),[])):
            sizes=[len(c) for c in verification_chunks(store,key)]
        assert sizes[0]==1024
        assert max(sizes[1:])<1024
        assert sum(sizes)==len(frame)


def test_concurrent_input_identities_equal_serial_and_reject_changed_native_output(tmp_path):
    import json
    from pathlib import Path
    from autobehaver.atomic_io import sha256_file
    from autobehaver.postinf.mmgen.integrity import source_identities
    inf=tmp_path/'inf';pos=tmp_path/'pos';inf.mkdir();pos.mkdir()
    files=[]
    for i in range(12):
        name=f'recording{i}_inference.csv';files.append(name)
        (inf/name).write_text('Frame Number,Pose\n1,Standing\n')
        partner=pos/f'recording{i}.csv';partner.write_text('x,y\n1,2\n')
        sidecar={'schema':'autobehaver.posepred_inference.provenance.v1', 'status':'complete',
                 'outputs':[{'path':name,'sha256':sha256_file(inf/name)}],
                 'source_positional_sha256':sha256_file(partner)}
        Path(str(inf/name)+'.provenance.json').write_text(json.dumps(sidecar))
    kwargs=dict(inference_csv_dir=inf,poscalc_dir=pos,inf_tag='_inference')
    with patch('autobehaver.preinf.execution.cpu_limit',return_value=1):
        serial=source_identities(files,**kwargs)
    with patch('autobehaver.preinf.execution.cpu_limit',return_value=4):
        parallel=source_identities(files,**kwargs)
    assert parallel==serial and list(parallel)==sorted(files)
    (inf/files[0]).write_text('Frame Number,Pose\n1,Walking\n')
    with pytest.raises(ValueError,match='does not match'):
        source_identities(files,**kwargs)
