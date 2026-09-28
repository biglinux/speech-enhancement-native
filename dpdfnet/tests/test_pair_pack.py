"""Independent tests of layout conversion, exact integer arithmetic and ABI guards."""
import copy
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from pack_matrices import transform, repack_array, convert


def fixture(rows=192, cols=64):
    rng = np.random.default_rng(621751)
    q = rng.integers(-127, 128, (rows, cols), dtype=np.int8)
    scales = np.linspace(0.001, 0.01, rows, dtype="<f4")
    tail = np.arange(17, dtype="<f4").tobytes()
    blob = q.tobytes() + scales.tobytes() + tail
    matrix = dict(kind="w8a16", rows=rows, cols=cols,
                  weight=dict(dtype="i8", offset=0, len=q.size),
                  scales=dict(dtype="f32", offset=q.size, len=rows))
    m = dict(schema=1, architecture="dpdfnet-48hr-v1", weights_sha256=hashlib.sha256(blob).hexdigest(),
             weight_bytes=len(blob), matrix=matrix,
             bias=dict(dtype="f32",offset=q.size+4*rows,len=17))
    return m, blob, q


@pytest.mark.parametrize("rows,cols", [(8,2),(24,6),(192,64),(768,256),(8,1024)])
def test_packing_mapping_and_roundtrip(rows,cols):
    m,blob,q=fixture(rows,cols)
    p,b,r=transform(m,blob)
    view=np.frombuffer(b[:q.size],np.int8)
    for row in range(rows):
        for j in range(cols):
            at=(row//8)*(cols*8)+(j//2)*16+(row%8)*2+j%2
            assert view[at]==q[row,j]
    assert b[q.size:]==blob[q.size:]
    assert len(b)==len(blob) and m["schema"]==1 and p["schema"]==2
    old,restored,_=transform(p,b,True)
    assert restored==blob and old==m
    assert r["changed_weight_bytes"]==q.size


@pytest.mark.parametrize("cols",[2,6,64,256,1024])
def test_pair_sums_are_exact_with_int64_oracle(cols):
    rows=24
    _,_,w=fixture(rows,cols)
    rng=np.random.default_rng(cols)
    for x in (rng.integers(-16383,16384,(4,cols),dtype=np.int16), np.full((4,cols),16383,np.int16)):
        p=repack_array(w).reshape(rows//8,cols//2,8,2).astype(np.int32)
        want=x.astype(np.int64)@w.astype(np.int64).T
        got=np.zeros((4,rows),np.int32)
        for k in range(4):
            sums=(p*x[k].reshape(1,cols//2,1,2)).sum(axis=(1,3),dtype=np.int64)
            got[k]=sums.reshape(-1)
        np.testing.assert_array_equal(got,want)
    assert 127*16383*cols<=np.iinfo(np.int32).max


def test_checksum_layout_bounds_and_overlap_rejections():
    m,b,_=fixture()
    with pytest.raises(ValueError): transform(m,b[:-1])
    bad=copy.deepcopy(m); bad["matrix"]["layout"]="unknown"
    with pytest.raises(ValueError): transform(bad,b)
    bad=copy.deepcopy(m); bad["matrix"]["layout"]="pair-output8-v1"
    with pytest.raises(ValueError): transform(bad,b)
    bad=copy.deepcopy(m); bad["alias"]=dict(dtype="f32",offset=0,len=4)
    with pytest.raises(ValueError): transform(bad,b)
    bad=copy.deepcopy(m); bad["matrix"]["cols"]=1026
    with pytest.raises(ValueError): transform(bad,b)


def test_conversion_is_idempotent_and_refuses_overwrite(tmp_path):
    m,b,_=fixture()
    src=tmp_path/"source";src.mkdir()
    (src/"manifest.json").write_text(json.dumps(m));(src/"weights.bin").write_bytes(b)
    dest=tmp_path/"packed";convert(src,dest)
    p,b1=json.loads((dest/"manifest.json").read_text()),(dest/"weights.bin").read_bytes()
    p2,b2,r=transform(p,b1)
    assert p2==p and b2==b1 and not r["changed_matrices"]
    with pytest.raises(FileExistsError): convert(src,dest)
    assert (src/"weights.bin").read_bytes()==b
