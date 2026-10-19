import hashlib
import json
import numpy as np
import pytest
from weatherai.official import assets


def test_orbax_literal_module_dots_and_integrity(tmp_path, monkeypatch):
    pytest.importorskip('tensorstore')
    output=tmp_path/'genfocal/super_resolution'
    output.mkdir(parents=True)
    tensor=output/'00000.npy'
    np.save(tensor,np.array([1.,2.],dtype=np.float32))
    name='DStack_0.84xres10x20.dblock0.Dense_0.bias'
    record={'name':name,'file':tensor.name,'sha256':hashlib.sha256(tensor.read_bytes()).hexdigest()}
    prefix,tree_prefix=assets.GENFOCAL['super_resolution']
    (output/'manifest.json').write_text(json.dumps({'source':prefix,'tree_prefix':tree_prefix,'tensors':[record]}))
    keys=['ema_state','ema','DStack_0','84xres10x20.dblock0','Dense_0','bias']
    metadata=output/'metadata.json'
    metadata.write_text(json.dumps({'tree_metadata':{'leaf':{'key_metadata':[{'key':k} for k in keys]}}}))
    monkeypatch.setattr(assets,'download',lambda *a,**k:(metadata,''))
    tree,_=assets.genfocal_parameters('super_resolution',tmp_path)
    np.testing.assert_array_equal(tree['DStack_0']['84xres10x20.dblock0']['Dense_0']['bias'],[1,2])
    tensor.write_bytes(b'corrupted')
    with pytest.raises(ValueError,match='checksum mismatch'):
        assets.genfocal_parameters('super_resolution',tmp_path)


def test_main_entries_are_official_and_lite_is_explicit():
    from weatherai import GenFocal,PBC,Translocation,GenFocal_lite,PBC_lite,Translocation_lite
    for cls in (GenFocal,PBC,Translocation):
        assert cls.implementation=='official'
    with pytest.raises(ValueError,match='no universal'):
        PBC.from_pretrained()
    for factory in (GenFocal_lite,PBC_lite,Translocation_lite):
        assert getattr(factory(),'implementation',None)!='official'
