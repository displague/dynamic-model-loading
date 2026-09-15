import copy
import json
from pathlib import Path
import numpy as np
import pytest
from dynamic_model_loading.sparse_screen import validate_config
from dynamic_model_loading.sparse_analysis import audit_pages, compare, decision, bounded_cuda


def fixture():
    c=dict(input_ids=[17],started=1.,finished=5.)
    calls=[c]
    episode=dict(condition='sparse',episode='episode-3')
    tensors={f'activity.{i+1}':np.zeros((1,1024),dtype=np.uint8) for i in range(24)}
    pages=[dict(call=i+1,layer=i,tokens=1,episode='episode-3',condition='sparse',
        started=2.+i*.1,selection_finished=2.02+i*.1,acquisition_finished=2.04+i*.1,compute_finished=2.08+i*.1,
        pages=[],extents=[],weight_h2d_bytes=0,activity_d2h_bytes=8193) for i in range(24)]
    return pages,tensors,calls,episode


def test_replay_all_zero():
    stats=audit_pages(*fixture())
    assert stats['h2d_bytes']==0 and stats['d2h_bytes']==24*8193


@pytest.mark.parametrize('mutation',['page','bytes','layer','clock','activity','missing','overlap'])
def test_replay_rejects_corruption(mutation):
    pages,tensors,calls,ep=fixture()
    if mutation=='page': pages[0]['pages']=[0]
    if mutation=='bytes': pages[0]['activity_d2h_bytes']=8192
    if mutation=='layer': pages[0]['layer']=1
    if mutation=='clock': pages[0]['compute_finished']=6.
    if mutation=='activity': tensors['activity.1'][0,0]=128
    if mutation=='missing': pages.pop()
    if mutation=='overlap': pages[1]['started']=pages[0]['started']
    with pytest.raises(ValueError): audit_pages(pages,tensors,calls,ep)


def test_numerical_is_per_position_not_diluted_by_larger_other_row():
    a=np.array([[1.,2.],[1e10,2e10]])
    b=a.copy(); b[0,0]+=.01
    with pytest.raises(ValueError): compare(a,b)
    assert compare(a,a)['relative_l2']==0


def test_gate_requires_both_real_traffic_and_latency():
    rows=dict(stream=dict(h2d_bytes=100,wall_seconds=10),sparse=dict(h2d_bytes=80,wall_seconds=11))
    assert decision(rows)['checks']==dict(Htraffic=True,Hruntime=False)
    rows['sparse']['wall_seconds']=9
    assert all(decision(rows)['checks'].values())


def test_inclusive_gate_boundaries():
    rows=dict(stream=dict(h2d_bytes=100,wall_seconds=100),sparse=dict(h2d_bytes=90,wall_seconds=95))
    assert all(decision(rows)['checks'].values())


def test_allocator_peak_enforces_cap_between_samples():
    receipt=dict(allocated_bytes=5000*2**20,reserved_bytes=6000*2**20,
                 peak_allocated_bytes=16000*2**20,peak_reserved_bytes=16000*2**20)
    with pytest.raises(ValueError): bounded_cuda(receipt)
    receipt.update(peak_allocated_bytes=15000*2**20,peak_reserved_bytes=15000*2**20)
    bounded_cuda(receipt)


def test_frozen_config():
    cfg=json.loads(Path('configs/sparse-down-screen.json').read_text())
    validate_config(cfg)
    cfg['page_width']=1
    with pytest.raises(ValueError): validate_config(cfg)
