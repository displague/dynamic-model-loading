from collections import OrderedDict
from types import SimpleNamespace
import pytest
import torch
from dynamic_model_loading.retained_rows import RetainedRows,retention_plan


def test_plan_overflow_consumes_hits_before_reusing_slots():
    state=OrderedDict([(1,0),(2,1)])
    hits,misses,inserts=retention_plan(state,[1,3,4,5],2)
    assert hits==[(1,0)] and misses==[3,4,5]
    assert list(state)==[4,5] and inserts==[(4,0),(5,1)]


@pytest.mark.parametrize('dtype',[torch.float32,torch.float16])
@pytest.mark.parametrize('capacity',[0,1,3,7])
def test_retention_matches_original_dense_with_interlayer_stale_rows(dtype,capacity):
    torch.manual_seed(37)
    layers=[SimpleNamespace(fc2=torch.nn.Linear(7,3,dtype=dtype)) for _ in range(2)]
    weights=[l.fc2.weight.detach().clone() for l in layers]; rows=[]
    bank=RetainedRows(layers,capacity=capacity,device='cpu',record=rows.append)
    for condition in ('packet','retained','packet'):
        bank.begin('test',condition)
        for index,active in [(0,[0,1,4]),(1,[1,3,4]),(0,[1,4]),(1,list(range(7))),(0,[])]:
            x=torch.zeros((2,7),dtype=dtype); x[:,active]=2
            expected=torch.nn.functional.linear(x,weights[index],layers[index].fc2.bias)
            torch.testing.assert_close(layers[index].fc2(x),expected,rtol=0,atol=0)
            assert len(bank.maps[index])<=capacity
        assert bank.workspace.T.is_contiguous()
    assert not any(bank.maps)


def test_nonfinite_rejected_and_episode_reset():
    layers=[SimpleNamespace(fc2=torch.nn.Linear(7,3))]
    bank=RetainedRows(layers,device='cpu',capacity=2); bank.begin('a','retained')
    with pytest.raises(ValueError): bank.forward(0,torch.full((1,7),float('nan')))
    bank.forward(0,torch.ones(1,7)); assert len(bank.maps[0])==2
    bank.begin('b','retained'); assert not bank.maps[0]


@pytest.mark.parametrize('active',[[2,1],[1,1],[-1]])
def test_bad_active_ids(active):
    with pytest.raises(ValueError): retention_plan(OrderedDict(),active,2)


def test_screen_import_and_audit_reject_missing_layers():
    from dynamic_model_loading.retention_screen import audit_pages
    with pytest.raises(ValueError): audit_pages([],{},[dict(input_ids=[1])],'packet',1024)


def test_exact_gate_boundary_and_single_byte_failure():
    from dynamic_model_loading.retention_screen import decision
    conditions=dict(packet=dict(h2d_bytes=16810000,wall_seconds=100.),
        retained=dict(h2d_bytes=15129000,wall_seconds=95.))
    assert all(decision(conditions)['checks'].values())
    conditions['retained']['h2d_bytes']+=1
    assert not decision(conditions)['checks']['Hacquisition']
    conditions['retained']['wall_seconds']+=.00001
    assert not decision(conditions)['checks']['Hruntime']


@pytest.mark.parametrize('tamper',[False,True])
def test_full_size_single_call_raw_replay(tamper):
    import numpy as np
    from dynamic_model_loading.retention_screen import audit_pages
    calls=[dict(input_ids=[4],started=1.,finished=3.)]; arrays={}; pages=[]
    for i in range(24):
        a=np.zeros((1,1024),dtype=np.uint8); a[0,0]=128; arrays[f'activity.{i+1}']=a
        pages.append(dict(call=i+1,layer=i,condition='retained',tokens=1,active=[0],hits=[],misses=[0],
            inserts=[[0,0]],residency=[[0,0]],weight_h2d_bytes=8192,metadata_h2d_bytes=24,
            activity_d2h_bytes=8193,started=1.,selection_finished=1.1,acquisition_finished=1.2,finished=2.))
    if tamper:
        pages[0]['metadata_h2d_bytes']=8
        with pytest.raises(ValueError): audit_pages(pages,arrays,calls,'retained',1024)
    else:
        stats=audit_pages(pages,arrays,calls,'retained',1024)
        assert stats['weight_h2d_bytes']==24*8192 and stats['metadata_h2d_bytes']==24*24
