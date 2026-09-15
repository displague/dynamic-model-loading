import json

import numpy as np
import pytest
import torch
from transformers.cache_utils import DynamicCache

from dynamic_model_loading.adapters import extract_ffns
from dynamic_model_loading.decision_field import CONFIG,FROZEN,branches,fingerprint,kv_storage_bytes,validate_config
from dynamic_model_loading.decision_field_analysis import frame_metrics,summarize_frames,validate_branch_rows,validate_page_rows,validate_parent_reference,validate_phases,SLAB
from dynamic_model_loading.fault_generation import step
from dynamic_model_loading.scoped_precision import ScopedPrecision
from test_progressive_precision import tiny_model


def test_frozen_decision_field():
    validate_config(json.loads(CONFIG.read_text(encoding='utf-8')))
    with pytest.raises(ValueError):
        validate_config(dict(FROZEN,sites=[0,7,14,21]))
    assert len(branches())==12


def test_additive_and_interacting_downstream_decisions():
    base = np.array([2.,0.,-3.])
    vectors = {name:base+len(sites)*np.array([-.5,.75,0.]) for name,sites in branches()}
    vectors['target'] = vectors['all8'].copy()
    metrics = frame_metrics(vectors)
    assert metrics['opportunity']
    assert all(p['argmax_match'] and p['interaction_norm']==0 for p in metrics['pairs'])
    vectors['1+8'] = np.array([100.,0.,0.])
    changed = frame_metrics(vectors)
    assert not changed['pairs'][0]['argmax_match']
    assert summarize_frames([changed])['aggregate_interaction_ratio']>0
    vectors['base'][0] = np.nan
    with pytest.raises(ValueError):
        frame_metrics(vectors)


def test_scoped_physical_execution_and_exact_old_kv():
    model = tiny_model()
    events = []
    pager = ScopedPrecision(extract_ffns(model),device='cpu',group=4,slots=1,page_sink=events.append)
    cache = DynamicCache(config=model.config)
    pager.selected = {0,1}
    for token in (1,3):
        step(model,torch.tensor([[token]]),cache,pager)
    before,_ = fingerprint(cache,2)
    events.clear()
    pager.selected = {1}
    step(model,torch.tensor([[5]]),cache,pager)
    storage = kv_storage_bytes(cache)
    cache.crop(2)
    assert fingerprint(cache,2)[0]==before
    assert kv_storage_bytes(cache)==storage
    assert [r['key'] for r in events]==[[1,0],[1,1]]
    events.clear()
    pager.selected = set()
    step(model,torch.tensor([[5]]),cache,pager)
    assert not events and fingerprint(cache,2)[0]==before
    with torch.inference_mode():
        cache.layers[0].keys[:,:,0] += 1
    assert fingerprint(cache,2)[0]!=before
    pager.selected = {-1}
    with pytest.raises(ValueError):
        pager.begin_token()
    pager.restore()


def valid_rows():
    return [dict(document='d',split='fit',position=pos,branch=name,sites=sites,
        consumed_token=pos+4,base_length=4+pos,end_length=4+pos+int(name=='base'),
        prefix_before='a'*64,prefix_after='a'*64,kv_fingerprint_d2h_bytes=57344*(4+pos),
        draft_kv_bytes=57344*(4+pos+int(name=='base')),target_kv_bytes=57344*(5+pos),
        draft_kv_storage_bytes=57344*(5+pos),target_kv_storage_bytes=57344*(5+pos),
        h2d_bytes=2*len(sites)*SLAB,correction_scalar_d2h_bytes=8*len(sites),wall_seconds=.1) for pos in range(4) for name,sites in branches()]


@pytest.mark.parametrize('field,value',[('h2d_bytes',0),('end_length',99),('prefix_after','b'*64),
    ('consumed_token',100),('wall_seconds',float('nan')),('kv_fingerprint_d2h_bytes',0)])
def test_branch_audit_rejects_drift(field,value):
    rows = valid_rows()
    validate_branch_rows(rows,[('d','fit')],{'d':list(range(8))})
    rows[0][field] = value
    with pytest.raises(ValueError):
        validate_branch_rows(rows,[('d','fit')],{'d':list(range(8))})


def test_page_audit_missing_or_fake_traffic():
    rows = [dict(phase='mechanics',document=None,position=None,branch=None,token=i+1,key=[i,s],
                 kind='load',slot=1,bytes=SLAB) for i in range(28) for s in (0,1)]
    assert validate_page_rows(rows,[])['h2d_bytes']==56*SLAB
    with pytest.raises(ValueError):
        validate_page_rows(rows[:-1],[])
    rows[0]['kind']='hit'
    with pytest.raises(ValueError):
        validate_page_rows(rows,[])


def test_changed_parent_reference_cannot_be_blessed_by_new_raw_hashes(tmp_path):
    import hashlib
    path = tmp_path/'parent-mechanics.safetensors'
    path.write_bytes(b'frozen-reference')
    expected = {'mechanics.safetensors':hashlib.sha256(path.read_bytes()).hexdigest()}
    validate_parent_reference(tmp_path,expected)
    path.write_bytes(b'replaced-reference')
    (tmp_path/'files.json').write_text(json.dumps({path.name:hashlib.sha256(path.read_bytes()).hexdigest()}))
    with pytest.raises(ValueError,match='frozen representation parent'):
        validate_parent_reference(tmp_path,expected)


def phase_rows():
    data = [dict(phase='model_loading'),dict(phase='construction_and_equality',equal_layers=[True]*28,
        construction_h2d_bytes=650280960,snapshot_d2h_bytes=650280960),
        dict(phase='mechanics',relative_l2=[0.]*28,correction_scalar_d2h_bytes=224,
             input_h2d_bytes=28*1536*4,output_d2h_bytes=28*1536*4),
        dict(phase='prefill',document='d',kv_bytes=2*4*57344,correction_scalar_d2h_bytes=4*28*8)]
    data.extend(dict(phase='frame',document='d',split='fit',position=p,
        initial_fingerprint_d2h_bytes=57344*(4+p),target_logit_d2h_bytes=151936*4,target_seconds=.1) for p in range(4))
    return [dict(**r,wall_seconds=1.,cuda={'peak_allocated_bytes':200},extra_cuda_peak_bytes=100) for r in data]


@pytest.mark.parametrize('index,field',[(1,'construction_h2d_bytes'),(1,'snapshot_d2h_bytes'),
    (2,'input_h2d_bytes'),(2,'output_d2h_bytes'),(3,'kv_bytes'),
    (4,'initial_fingerprint_d2h_bytes'),(4,'target_logit_d2h_bytes'),(4,'target_seconds')])
def test_missing_and_wrong_phase_accounting(index,field):
    rows = phase_rows()
    validate_phases(rows,[('d','fit')],100)
    rows[index][field]=0
    with pytest.raises(ValueError):
        validate_phases(rows,[('d','fit')],100)
    del rows[index][field]
    with pytest.raises(KeyError):
        validate_phases(rows,[('d','fit')],100)
